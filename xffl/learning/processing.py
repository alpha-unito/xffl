"""DNN training utility methods"""

import logging
import os
import time
from logging import Logger, getLogger
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor, nn, tensor
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from wandb.wandb_run import Run

from xffl.custom.types import PathLike
from xffl.distributed.aggregation import bucket_optimized_coalesced_
from xffl.distributed.distributed import DistributedState
from xffl.learning.modelling import save_model

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def distributed_training(
    model: FullyShardedDataParallel,
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
    gradient_clipping=None,
    accumulation_steps=None,
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
    best_val_loss: Tensor = tensor(float("inf"))

    if torch.distributed.is_initialized():
        dist.barrier(device_ids=[state.node_local_rank])
    for epoch in range(epochs):
        epoch_start_time: float = time.perf_counter()
        logger.info(f"Starting epoch {epoch}/{epochs}")

        model.train()

        total_loss: Tensor = tensor(0.0)
        total_length: int = len(train_dataloader)
        pbar: tqdm = tqdm(
            colour="blue",
            desc=f"Training Epoch: {epoch+1}",
            total=total_length,
            dynamic_ncols=True,
            disable=state.rank != 0,
        )

        if (
            save_path is not None
            and output_model_name is not None
            and (not os.path.exists(save_path) or not os.path.isdir(save_path))
        ):
            logger.warning(
                f"Impossible saving the trained model with save dir {save_path}: saving disabled."
            )
            save_path, output_model_name = None, None

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
                # loss: torch.Tensor = model(
                #     input_ids=batch["input_ids"],
                #     attention_mask=batch["attention_mask"],
                #     labels=batch["input_ids"],
                # ).loss
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

            if accumulation_steps is not None:
                loss = loss / accumulation_steps
            loss.backward()

            if logging.root.level == logging.DEBUG:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                back_time = time.perf_counter() - start_time
                start_time = time.perf_counter()

            if (
                accumulation_steps is not None
                and (step + 1) % accumulation_steps == 0
                or (step + 1) == len(train_dataloader)
            ):
                if gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clipping
                    )
                optimizer.step()
                optimizer.zero_grad()

                if lr_scheduler:
                    lr_scheduler.step()
            pbar.update(1)

            if logging.root.level == logging.DEBUG:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                optimizer_time = time.perf_counter() - start_time
                start_time = time.perf_counter()

            with torch.no_grad():
                assert federated_batches is not None

                if state.is_federated_scaling_setup() and (
                    (step + 1) % federated_batches == 0 or step + 1 == total_length
                ):
                    bucket_optimized_coalesced_(
                        model=model,
                        state=state,
                        use_multiple_cuda_streams=True,
                        use_contiguous_memory=True,
                    )

            if logging.root.level == logging.DEBUG:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                comm_time = time.perf_counter() - start_time
                start_time = time.perf_counter()

            if accumulation_steps is not None:
                loss = loss * accumulation_steps
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
                        "train/learning rate": (
                            lr_scheduler.get_lr() if lr_scheduler is not None else None
                        ),
                        "train/gradient_accumulation": accumulation_steps,
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

        assert state.federated_local_size is not None
        assert state.federated_rank is not None
        assert state.world_size is not None

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

            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                # if state.rank == 0:
                #    logger.info(f"Best eval loss on epoch {epoch+1} is {best_val_loss}")
            val_loss.append(float(best_val_loss))
            val_perp.append(float(eval_ppl))

        if output_model_name is not None and save_path is not None:
            assert state.rank is not None
            checkpoint_start_time = time.perf_counter()
            save_model(
                model=model,
                optimizer=optimizer,
                path=save_path,
                name=output_model_name,
                rank=state.rank,
                epoch=epoch,
            )
            checkpoint_times.append(time.perf_counter() - checkpoint_start_time)

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
) -> Tuple[Tensor, Tensor, List[float], List[float], float]:
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
    eval_loss: Tensor = tensor(0.0)
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
            loss: Tensor

            if isinstance(batch, dict):
                for key in batch.keys():
                    batch[key] = batch[key].to(
                        device=torch.cuda.current_device(), non_blocking=True
                    )

                outputs = model(**batch)
                loss = outputs.loss

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
                loss = criterion(output, target)
                eval_loss += loss
                pred: int = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        if torch.cuda.device_count() > 1:
            dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    assert state.world_size is not None
    assert eval_dataloader.batch_size is not None

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
