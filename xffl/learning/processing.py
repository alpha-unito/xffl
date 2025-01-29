"""DNN training utility methods"""

import time
from logging import Logger, getLogger
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from wandb.wandb_run import Run

from xffl.custom.types import PathLike
from xffl.learning.modelling import save_FSDP_model

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def fsdp_training(
    model: nn.Module,
    optimizer: Optimizer,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    rank: int,
    local_rank: int,
    world_size: int,
    validate: bool = True,
    epochs: int = 1,
    save_path: Optional[PathLike] = None,
    model_name: Optional[str] = None,
    save_model: Optional[bool] = None,
    lr_scheduler: Optional[LRScheduler] = None,
    wandb_run: Optional[Run] = None,
    verbose: Optional[bool] = None,
) -> Dict[str, float]:
    """Genreric training cycle for FSDP models

    :param model: Model to train
    :type model: nn.Module
    :param optimizer: Model's optimizer
    :type optimizer: Optimizer
    :param train_dataloader: Training dataset data loader
    :type train_dataloader: DataLoader
    :param eval_dataloader: Validation dataset data loader
    :type eval_dataloader: DataLoader
    :param rank: Global rank of the calling process
    :type rank: int
    :param local_rank: Local rank of the calling process
    :type local_rank: int
    :param world_size: World size of the processes taking part to the FSDP training
    :type world_size: int
    :param validate: Activate validation, defaults to True
    :type validate: bool, optional
    :param epochs: Number of epochs to train, defaults to 1
    :type epochs: int, optional
    :param save_path: Path where to save the trained model, defaults to None
    :type save_path: Optional[PathLike], optional
    :param model_name: Name to use for the saved trained model, defaults to None
    :type model_name: Optional[str], optional
    :param save_model: If to save the model, defaults to None
    :type save_model: Optional[bool], optional
    :param lr_scheduler: Learning rate scheduler, defaults to None
    :type lr_scheduler: Optional[LRScheduler], optional
    :param wandb_run: WandB run if wandb logging is desired, defaults to None
    :type wandb_run: Optional[wandb.Run], optional
    :param precision: Precision to use for PyTorch's automatic mixed precision context manager, defaults to None
    :type precision: Optional[torch.dtype], optional
    :param verbose: Enable verbose output, defaults to None
    :type verbose: Optional[bool], optional
    :return: Dictionary of metrics names and achieved values
    :rtype: Dict[str, float]
    """

    # TODO: if fp16 the gradients should be rescaled
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss = []

    train_step_perplexity = []
    train_step_loss = []
    val_step_loss = []
    val_step_perplexity = []

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")

    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch}/{epochs}")
        epoch_start_time = time.perf_counter()

        model.train()
        total_loss = 0.0
        total_length = len(train_dataloader)
        pbar = tqdm(
            colour="blue",
            desc=f"Training Epoch: {epoch+1}",
            total=total_length,
            dynamic_ncols=True,
        )
        for step, batch in enumerate(train_dataloader):
            for key in batch.keys():
                batch[key] = batch[key].to(device=local_rank, non_blocking=True)
            loss = model(**batch).loss

            total_loss += loss.detach().float()
            train_step_loss.append(loss.detach().float().item())
            train_step_perplexity.append(float(torch.exp(loss.detach().float())))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.update(1)

            if wandb_run and rank == 0:
                wandb_run.log(
                    {
                        "train/epoch": epoch + 1,
                        "train/step": epoch * len(train_dataloader) + step,
                        "train/loss": loss.detach().float(),
                    }
                )

            pbar.set_description(
                f"Training Epoch: {epoch+1}/{epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})"
            )

        pbar.close()

        epoch_end_time = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_end_time)

        if torch.cuda.device_count():
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)

        train_epoch_loss = (total_loss / len(train_dataloader)) / world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        if lr_scheduler:
            lr_scheduler.step()

        if validate:
            eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = (
                fsdp_evaluation(
                    model, eval_dataloader, local_rank, world_size, wandb_run
                )
            )

            val_step_loss.extend(temp_val_loss)
            val_step_perplexity.extend(temp_step_perplexity)

            if save_model:
                checkpoint_start_time = time.perf_counter()
                save_FSDP_model(
                    model=model,
                    path=save_path,
                    name=model_name,
                    rank=rank,
                    epoch=epoch,
                    verbose=True,
                )
                # TODO: possible barrier?
                checkpoint_end_time = time.perf_counter() - checkpoint_start_time
                checkpoint_times.append(checkpoint_end_time)

            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                if verbose and rank == 0:
                    logger.info(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
            val_loss.append(float(best_val_loss))
            val_prep.append(float(eval_ppl))

        if rank == 0:
            logger.info(
                f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s"
            )

    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    avg_train_prep = sum(train_prep) / len(train_prep)
    avg_train_loss = sum(train_loss) / len(train_loss)
    if validate:
        avg_eval_prep = sum(val_prep) / len(val_prep)
        avg_eval_loss = sum(val_loss) / len(val_loss)

    results["avg_train_prep"] = avg_train_prep
    results["avg_train_loss"] = avg_train_loss
    if validate:
        results["avg_eval_prep"] = avg_eval_prep
        results["avg_eval_loss"] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time

    return results


def fsdp_evaluation(
    model: nn.Module,
    eval_dataloader: DataLoader,
    local_rank: int,
    world_size: int,
    wandb_run: Optional[Run] = None,
) -> Tuple[float, float, List[float], List[float]]:
    """Genreric evaluation cycle for FSDP models

    :param model: Model to evaluate
    :type model: nn.Module
    :param eval_dataloader: Validation dataset data loader
    :type eval_dataloader: DataLoader
    :param local_rank: Local rank of the calling process
    :type local_rank: int
    :param world_size: World size of the processes taking part to the FSDP training
    :type world_size: int
    :param wandb_run: WandB run if wandb logging is desired, defaults to None
    :type wandb_run: Optional[wandb.Run], optional
    :return: perplexity, epoch loss, step loff, step perplexity
    :rtype: Tuple[float, float, float, float]
    """
    model.eval()
    val_step_loss = []
    val_step_perplexity = []
    eval_loss = 0.0
    for _, batch in enumerate(
        tqdm(
            eval_dataloader,
            colour="green",
            desc="evaluating Epoch",
            dynamic_ncols=True,
        )
    ):
        for key in batch.keys():
            batch[key] = batch[key].to(device=f"cuda:{local_rank}", non_blocking=True)

        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss

            eval_loss += loss.detach().float()
            val_step_loss.append(loss.detach().float().item())
            val_step_perplexity.append(float(torch.exp(loss.detach().float())))

    if torch.cuda.device_count() > 1:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_epoch_loss = eval_epoch_loss / world_size
    eval_ppl = torch.exp(eval_epoch_loss)

    if local_rank == 0:
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
