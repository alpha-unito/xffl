"""DNN training utility methods"""

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
from xffl.distributed.aggregation import (
    layer_by_layer_aggregation,
    sync_federated_averaging,
)
from xffl.distributed.distributed import DistributedState
from xffl.learning.modelling import save_fsdp_model

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


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
        )

        step: int
        batch: Dict[str, Any]
        for step, batch in enumerate(train_dataloader):
            # logger.warning(f"[RANK {state.rank}]:1")
            if state.rank == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
            for key in batch.keys():
                batch[key] = batch[key].to(
                    device=(
                        torch.cuda.current_device()
                        if torch.cuda.is_available()
                        else "cpu"
                    ),
                    non_blocking=True,
                )

            loss: torch.Tensor = model(**batch).loss

            # logger.warning(f"[RANK {state.rank}]2")
            if state.rank == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                batch_time = time.perf_counter() - start_time
                start_time = time.perf_counter()

            loss.backward()

            # logger.warning(f"[RANK {state.rank}]:3")
            if state.rank == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                back_time = time.perf_counter() - start_time
                start_time = time.perf_counter()

            optimizer.step()  # TODO: average optimizer?
            optimizer.zero_grad()

            # logger.warning(f"[RANK {state.rank}]:4")
            if state.rank == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                optimizer_time = time.perf_counter() - start_time
                start_time = time.perf_counter()

            with torch.no_grad():
                if state.is_federated_scaling_setup() and (
                    (step + 1) % federated_batches == 0 or step + 1 == total_length
                ):
                    layer_by_layer_aggregation(model=model, state=state)

                # logger.warning(f"[RANK {state.rank}]:6")
                if state.rank == 0:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    comm_time = time.perf_counter() - start_time
                    start_time = time.perf_counter()

                total_loss += loss.detach().float()
                train_step_perplexity.append(float(torch.exp(loss.detach().float())))
                train_step_loss.append(loss.detach().float().item())

                pbar.update(1)
                pbar.set_description(
                    f"Training Epoch: {epoch + 1}/{epochs}, step {step}/{total_length} completed (loss: {train_step_loss[-1]:.4f})"
                )

                if wandb_run:
                    wandb_run.log(
                        {
                            "train/epoch": epoch + 1,
                            "train/step": epoch * total_length + step,
                            "train/loss": train_step_loss[-1],
                            "train/perplexity": train_step_perplexity[-1],
                        }
                    )

            if state.rank == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                logger.debug(
                    f"[RANK {state.rank}]: Forward: {batch_time:.2f}, Backward: {back_time:.2f}, Optimizer: {optimizer_time:.2f}, Averaging: {comm_time:.2f}, Metrics update: {(time.perf_counter() - start_time):.2f}"
                )

        pbar.close()
        epoch_times.append(time.perf_counter() - epoch_start_time)

        if dist.is_initialized():
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM, group=state.federation)

        train_epoch_loss: torch.Tensor = (
            (total_loss / total_length)
            / state.federated_local_size[state.federated_rank]
            if state.is_federated_scaling_setup()
            else (total_loss / total_length) / state.world_size
        )
        train_perplexity: torch.Tensor = torch.exp(train_epoch_loss)

        train_perp.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        if lr_scheduler:
            lr_scheduler.step()

        if validate:
            eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = (
                fsdp_evaluation(
                    model=model,
                    eval_dataloader=eval_dataloader,
                    state=state,
                    wandb_run=wandb_run,
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
                if state.rank == 0:
                    logger.info(f"Best eval loss on epoch {epoch+1} is {best_val_loss}")
            val_loss.append(float(best_val_loss))
            val_perp.append(float(eval_ppl))

        if state.rank == 0:
            logger.info(
                f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_times[-1]}s"
            )

    results["avg_epoch_time"] = sum(epoch_times) / len(epoch_times)
    results["avg_train_perp"] = sum(train_perp) / len(train_perp)
    results["avg_train_loss"] = sum(train_loss) / len(train_loss)
    if validate:
        results["avg_eval_perp"] = sum(val_perp) / len(val_perp)
        results["avg_eval_loss"] = sum(val_loss) / len(val_loss)

    return results


def fsdp_evaluation(
    model: nn.Module,
    eval_dataloader: DataLoader,
    state: DistributedState,
    wandb_run: Optional[Run] = None,
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

    batch: Dict[str, Any]
    for _, batch in enumerate(
        tqdm(
            eval_dataloader,
            colour="green",
            desc="evaluating Epoch",
            dynamic_ncols=True,
        )
    ):
        for key in batch.keys():
            batch[key] = batch[key].to(
                device=torch.cuda.current_device(), non_blocking=True
            )

        with torch.no_grad():
            outputs = model(**batch)
            loss: torch.Tensor = outputs.loss

            eval_loss += loss.detach().float()
            val_step_loss.append(loss.detach().float().item())
            val_step_perplexity.append(float(torch.exp(loss.detach().float())))

    if torch.cuda.device_count() > 1:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    eval_epoch_loss: torch.Tensor = (
        eval_loss / len(eval_dataloader)
    ) / state.world_size
    eval_ppl: torch.Tensor = torch.exp(eval_epoch_loss)

    if state.node_local_rank == 0:
        logger.info(f" {eval_ppl=} {eval_epoch_loss=}")

    if wandb_run:
        wandb_run.log(
            {
                "eval/perplexity": eval_ppl,
                "eval/loss": eval_epoch_loss,
            },
            commit=False,
        )

    return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity
