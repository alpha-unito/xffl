"""DNN training utility methods"""

import logging
import math
import time
from logging import Logger, getLogger
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from wandb.wandb_run import Run

from xffl.custom.types import PathLike
from xffl.distributed.aggregation import bucket_optimized_coalesced_
from xffl.distributed.distributed import DistributedState
from xffl.learning.modelling import save_fsdp_model

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def _move_opt_to(opt, device):
    for param in opt.state.values():
        for k, v in param.items():
            if torch.is_tensor(v):
                param[k] = v.to(device, non_blocking=True)


def distributed_training(
    model: nn.Module | FullyShardedDataParallel,
    optimizer: Optimizer,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    state: DistributedState,
    federated_batches: Optional[int] = None,
    validate: bool = True,
    epochs: int = 1,
    save_path: Optional[PathLike] = None,
    output_model_name: Optional[str] = None,
    lr_scheduler: Optional[LRScheduler] = None,
    wandb_run: Optional[Run] = None,
    criterion=None,
    fedopt=False,
) -> Dict[str, float]:
    """Generic training cycle for FSDP models

    :param model: Model to train
    :type model: nn.Module
    :param optimizer: Model's optimizer
    :type optimizer: Optimizer
    :param train_dataloader: Training dataset data loader
    :type train_dataloader: DataLoader
    :param eval_dataloader: Validation dataset data loader
    :type eval_dataloader: DataLoader
    :param state: Instantiated distributed state
    :type state: DistributedState
    :param federated_batches: Number of training batched to process between two federated averaging
    :type federated_batches: Optional[int]
    :param validate: Activate validation, defaults to True
    :type validate: bool, optional
    :param epochs: Number of epochs to train, defaults to 1
    :type epochs: int, optional
    :param save_path: Path where to save the trained model, defaults to None
    :type save_path: Optional[PathLike], optional
    :param output_model_name: Name to use for the saved trained model, defaults to None
    :type output_model_name: Optional[str], optional
    :param lr_scheduler: Learning rate scheduler, defaults to None
    :type lr_scheduler: Optional[LRScheduler], optional
    :param wandb_run: WandB run if wandb logging is desired, defaults to None
    :type wandb_run: Optional[wandb.Run], optional
    :return: Dictionary of metrics names and achieved values
    :rtype: Dict[str, float]
    """

    # TODO: if fp16 the gradients should be rescaled
    train_perp: List[float] = []
    train_loss: List[float] = []
    val_perp: List[float] = []
    val_loss: List[float] = []

    train_step_perplexity: List[float] = []
    train_step_loss: List[float] = []
    val_step_loss: List[float] = []
    val_step_perplexity: List[float] = []

    epoch_times: List[float] = []
    checkpoint_times: List[float] = []
    results: Dict[str, float] = {}
    best_val_loss: float = float("inf")

    agg_lr_sched = None
    if fedopt:
        _move_opt_to(optimizer, "cpu")

        origin_state_dict = {
            k: v.detach().clone().to(device="cpu", non_blocking=True)
            for k, v in model.state_dict().items()
        }
        grads = [
            (torch.zeros_like(p, device="cpu") if p.requires_grad else None)
            for p in model.parameters()
        ]
        optimizer_agg = torch.optim.AdamW(
            model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, fused=True
        )

        from torch.optim.lr_scheduler import LambdaLR

        def get_cosine_schedule(optimizer, total_steps, warmup_frac=0.1):
            """
            Scheduler stile LLaMA3.1 semplificato: warmup -> cosine decay.

            Args:
                optimizer: torch.optim.Optimizer
                total_steps (int): passi totali (es. 128)
                lr_max (float): learning rate massimo
                warmup_frac (float): frazione di warmup (default 5%)
            """
            warmup_steps = int(total_steps * warmup_frac)
            decay_steps = total_steps - warmup_steps

            def lr_lambda(step):
                if step < warmup_steps:
                    # warmup lineare
                    return step / max(1, warmup_steps)
                else:
                    # decadimento coseno
                    progress = (step - warmup_steps) / max(1, decay_steps)
                    return 0.5 * (1 + math.cos(math.pi * progress))

            return LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step))

        agg_lr_sched = get_cosine_schedule(
            optimizer_agg,
            total_steps=len(train_dataloader),
        )
        _move_opt_to(optimizer_agg, "cpu")
        _move_opt_to(optimizer, state.current_device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    if torch.distributed.is_initialized():
        dist.barrier(device_ids=[state.node_local_rank])
    for epoch in range(epochs):
        epoch_start_time: float = time.perf_counter()
        logger.info(f"Starting epoch {epoch}/{epochs}")

        model.train()

        total_loss: float = 0.0
        total_length: int = len(train_dataloader)
        pbar: tqdm = tqdm(
            colour="blue",
            desc=f"Training Epoch: {epoch+1}",
            total=total_length,
            dynamic_ncols=True,
            disable=state.rank != 0,
        )

        step: int
        batch: Dict[str, Any]
        for step, batch in enumerate(train_dataloader):
            if logging.root.level == logging.DEBUG:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.perf_counter()

            if isinstance(batch, dict):
                for key in batch.keys():
                    batch[key] = batch[key].to(
                        device=state.current_device,
                        non_blocking=True,
                    )
                loss: torch.Tensor = model(**batch).loss
            else:
                data, target = batch
                data, target = data.to(
                    device=state.current_device,
                    non_blocking=True,
                ), target.to(
                    device=state.current_device,
                    non_blocking=True,
                )
                loss: torch.Tensor = criterion(model(data), target)

            if logging.root.level == logging.DEBUG:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                batch_time = time.perf_counter() - start_time
                start_time = time.perf_counter()

            loss.backward()

            if logging.root.level == logging.DEBUG:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                back_time = time.perf_counter() - start_time
                start_time = time.perf_counter()

            optimizer.step()  # TODO: average optimizer?
            optimizer.zero_grad(set_to_none=True)
            if lr_scheduler:
                lr_scheduler.step()
                if agg_lr_sched:
                    agg_lr_sched.step()
            pbar.update(1)

            if logging.root.level == logging.DEBUG:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                optimizer_time = time.perf_counter() - start_time
                start_time = time.perf_counter()

            if state.is_federated_scaling_setup() and (
                (step + 1) % federated_batches == 0 or step + 1 == total_length
            ):
                with torch.no_grad():
                    bucket_optimized_coalesced_(
                        model=model,
                        state=state,
                        use_multiple_cuda_streams=False,
                        use_contiguous_memory=False,
                    )

                    if fedopt:
                        _move_opt_to(optimizer, "cpu")

                        current_state_dict = {
                            k: v.detach().clone().to(device="cpu")
                            for k, v in model.state_dict().items()
                        }

                        for g, p_origin, p_current in zip(
                            grads,
                            origin_state_dict.values(),
                            current_state_dict.values(),
                            strict=True,
                        ):
                            if g is not None:
                                g.copy_(p_origin.data - p_current.data).to(
                                    device=state.current_device
                                )

                        if state.rank == 0:
                            total_norm = sum(
                                (g.norm().item() if g is not None else torch.tensor(0))
                                for g in grads
                            )
                            print(f"Norma dei cambiamenti: {total_norm}")

                        model.load_state_dict(origin_state_dict, strict=True)
                        for p, g in zip(model.parameters(), grads, strict=True):
                            if p.requires_grad:
                                p.grad = g.detach().to(device=state.current_device)

                        if state.rank == 0:
                            pre_clipping_norm = torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=1.0
                            )
                            print(f"Pre-clipping norm: {pre_clipping_norm}")

                            total_norm = torch.sqrt(
                                sum(
                                    (
                                        p.grad.norm() ** 2
                                        if p.grad is not None
                                        else torch.tensor(0)
                                    )
                                    for p in model.parameters()
                                )
                            )
                            print(f"Norma dei gradienti: {total_norm}")

                        _move_opt_to(optimizer_agg, state.current_device)

                        optimizer_agg.step()
                        if state.rank == 0:
                            diff = sum(
                                (p.detach().clone().to("cpu") - p_before).norm().item()
                                for p, p_before in zip(
                                    model.parameters(), current_state_dict.values()
                                )
                            )
                            print("Total change after optimizer.step():", diff)

                        optimizer_agg.zero_grad(set_to_none=True)
                        # if agg_lr_sched:
                        #    agg_lr_sched.step()
                        _move_opt_to(optimizer_agg, "cpu")

                        origin_state_dict = {
                            k: v.detach().clone().to(device="cpu")
                            for k, v in model.state_dict().items()
                        }

                        _move_opt_to(optimizer, state.current_device)

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()

            if logging.root.level == logging.DEBUG:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                comm_time = time.perf_counter() - start_time
                start_time = time.perf_counter()

            total_loss += loss.detach().float()
            train_step_perplexity.append(float(torch.exp(loss.detach().float())))
            train_step_loss.append(loss.detach().float().item())

            if wandb_run:
                wandb_run.log(
                    {
                        "train/epoch": epoch + 1,
                        "train/step": epoch * total_length + step,
                        "train/loss": train_step_loss[-1],
                        "train/perplexity": train_step_perplexity[-1],
                    }
                )

            pbar.set_description(
                f"Training Epoch: {epoch + 1}/{epochs}, step {step}/{total_length} completed (loss: {train_step_loss[-1]:.4f})",
                refresh=False,
            )

            if logging.root.level == logging.DEBUG:
                pbar.set_postfix(
                    ordered_dict={
                        "F": f"{batch_time:.2f}",
                        "B": f"{back_time:.2f}",
                        "O": f"{optimizer_time:.2f}",
                        "A": f"{comm_time:.2f}",
                        "Other": f"{(time.perf_counter() - start_time):.2f}",
                    },
                    refresh=False,
                )

            if logging.root.level == logging.DEBUG:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

        pbar.close()
        epoch_times.append(time.perf_counter() - epoch_start_time)

        # TODO: fix this
        # if dist.is_initialized():
        #    dist.all_reduce(
        #        tensor=total_loss,  # .to(state.current_device),
        #        op=dist.ReduceOp.SUM,
        #    )

        train_epoch_loss: torch.Tensor = (
            (total_loss / total_length)
            / state.federated_local_size[state.federated_rank]
            if state.is_federated_scaling_setup()
            else (total_loss / total_length) / state.world_size
        )
        train_perplexity: torch.Tensor = torch.exp(train_epoch_loss)

        train_perp.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        if validate:
            eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity, eval_acc = (
                fsdp_evaluation(
                    model=model,
                    eval_dataloader=eval_dataloader,
                    state=state,
                    wandb_run=wandb_run,
                    criterion=criterion,
                )
            )

            val_step_loss.extend(temp_val_loss)
            val_step_perplexity.extend(temp_step_perplexity)

            if output_model_name:
                checkpoint_start_time = time.perf_counter()
                save_fsdp_model(
                    model=model,
                    optimizer=optimizer,
                    path=save_path,
                    name=output_model_name,
                    rank=state.rank,
                    epoch=epoch,
                )
                checkpoint_times.append(time.perf_counter() - checkpoint_start_time)

            if eval_epoch_loss < best_val_loss:
                best_val_loss: torch.Tensor = eval_epoch_loss
                # if state.rank == 0:
                #    logger.info(f"Best eval loss on epoch {epoch+1} is {best_val_loss}")
            val_loss.append(float(best_val_loss))
            val_perp.append(float(eval_ppl))

        if state.rank == 0:
            if validate:
                logger.info(
                    f"Epoch {epoch+1}:\n\ttrain_perplexity={train_perplexity:.4f},\n\ttrain_epoch_loss={train_epoch_loss:.4f},\n\tepoch time {epoch_times[-1]:.2f}s,\n\taccuracy={eval_acc:.2f}%"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1}:\n\ttrain_perplexity={train_perplexity:.4f},\n\ttrain_epoch_loss={train_epoch_loss:.4f},\n\tepoch time {epoch_times[-1]:.2f}s"
                )

    results["avg_epoch_time"] = sum(epoch_times) / len(epoch_times)
    results["avg_train_perp"] = sum(train_perp) / len(train_perp)
    results["avg_train_loss"] = sum(train_loss) / len(train_loss)
    if validate:
        results["avg_eval_perp"] = sum(val_perp) / len(val_perp)
        results["avg_eval_loss"] = sum(val_loss) / len(val_loss)
        results["val_acc"] = eval_acc

    return results


def fsdp_evaluation(
    model: nn.Module,
    eval_dataloader: DataLoader,
    state: DistributedState,
    wandb_run: Optional[Run] = None,
    criterion=None,
) -> Tuple[torch.Tensor, torch.Tensor, List[float], List[float]]:
    """Generic evaluation cycle for FSDP models

    :param model: Model to evaluate
    :type model: nn.Module
    :param eval_dataloader: Validation dataset data loader
    :type eval_dataloader: DataLoader
    :param state: Instantiated distributed state
    :type state: DistributedState
    :param wandb_run: WandB run if wandb logging is desired, defaults to None
    :type wandb_run: Optional[wandb.Run], optional
    :return: perplexity, epoch loss, step loss, step perplexity
    :rtype: Tuple[float, float, float, float]
    """
    model.eval()

    val_step_loss: List[float] = []
    val_step_perplexity: List[float] = []
    eval_loss: float = 0.0
    correct: int = 0

    with torch.no_grad():
        batch: Dict[str, Any]
        for _, batch in enumerate(
            tqdm(
                eval_dataloader,
                colour="green",
                desc="evaluating Epoch",
                dynamic_ncols=True,
                disable=state.rank != 0,
            )
        ):
            if isinstance(batch, dict):
                for key in batch.keys():
                    batch[key] = batch[key].to(
                        device=torch.cuda.current_device(), non_blocking=True
                    )

                outputs = model(**batch)
                loss: torch.Tensor = outputs.loss

                eval_loss += loss.detach().float()
                val_step_loss.append(loss.detach().float().item())
                val_step_perplexity.append(float(torch.exp(loss.detach().float())))
            else:
                data, target = batch
                data, target = data.to(
                    device=state.current_device,
                    non_blocking=True,
                ), target.to(
                    device=state.current_device,
                    non_blocking=True,
                )

                output: torch.Tensor = model(data)
                loss: float = criterion(output, target)
                eval_loss += loss
                pred: int = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        if torch.cuda.device_count() > 1:
            dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    eval_epoch_loss: torch.Tensor = (
        eval_loss / len(eval_dataloader)
    ) / state.world_size
    eval_ppl: torch.Tensor = torch.exp(eval_epoch_loss)
    eval_acc: float = (100.0 * correct) / (
        len(eval_dataloader) * eval_dataloader.batch_size
    )

    # if state.node_local_rank == 0:
    #    logger.info(f" {eval_ppl=} {eval_epoch_loss=}")

    if wandb_run:
        wandb_run.log(
            {
                "eval/perplexity": eval_ppl,
                "eval/loss": eval_epoch_loss,
                "eval/accuracy": eval_acc,
            },
            commit=False,
        )

    return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity, eval_acc
