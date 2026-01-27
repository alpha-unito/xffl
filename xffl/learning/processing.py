"""Training utilities"""

import logging
import os
import time
from logging import Logger, getLogger
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor, nn, tensor
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from wandb.wandb_run import Run

from xffl.custom.config import XFFLConfig
from xffl.custom.types import PathLike
from xffl.distributed.aggregation import bucket_optimized_coalesced_
from xffl.distributed.distributed import DistributedState
from xffl.learning.modelling import save_model
from xffl.learning.utils import (
    cuda_reset_memory_stats_and_empty_cache,
    cuda_sync,
    cuda_sync_and_empty_cache,
    resolve_param,
)

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""

# --------------------------------------------------------------------------- #
#                               Helper functions                              #
# --------------------------------------------------------------------------- #


def _move_opt_to(
    optimizer: Optimizer, device: torch.device | str, state: DistributedState
) -> None:
    if state.device_type != "cpu":
        for param in optimizer.state.values():
            for k, v in param.items():
                if torch.is_tensor(v):
                    param[k] = v.to(device, non_blocking=True)


def _get_train_function(batch: Mapping[Any, Any] | Tuple[Any, Any]) -> Callable:

    train_function: Callable

    def attention_mask_training(
        model: nn.Module,
        batch: Dict[str, Any],
        state: DistributedState,
        criterion: nn.Module,
    ) -> Tensor:
        for key in batch.keys():
            batch[key] = batch[key].to(
                device=state.current_device,
                non_blocking=True,
            )

        return model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["input_ids"],
        ).loss

    def dict_training(
        model: nn.Module,
        batch: Dict[str, Any],
        state: DistributedState,
        criterion: nn.Module,
    ) -> Tensor:
        for key in batch.keys():
            batch[key] = batch[key].to(
                device=state.current_device,
                non_blocking=True,
            )
        return model(**batch).loss

    def criterion_training(
        model: nn.Module,
        batch: Tuple[Any, Any],
        state: DistributedState,
        criterion: nn.Module,
    ) -> Tensor:
        data, target = batch
        data, target = data.to(
            device=state.current_device,
            non_blocking=True,
        ), target.to(
            device=state.current_device,
            non_blocking=True,
        )
        return criterion(model(data), target)

    if isinstance(batch, dict):
        if all(key in batch for key in ["input_ids", "attention_mask"]):
            train_function = attention_mask_training
        else:
            train_function = dict_training
    else:
        train_function = criterion_training

    return train_function


def _fedopt_setup(
    model: nn.Module,
    optimizer: Optimizer,
    fedopt_optimizer: Optimizer,
    state: DistributedState,
) -> Tuple[Mapping[str, Tensor], List[Optional[Tensor]]]:
    assert state.current_device is not None

    _move_opt_to(optimizer=optimizer, device="cpu", state=state)

    origin_state_dict: Mapping[str, Tensor] = {
        k: v.detach().clone().to(device="cpu", non_blocking=True)
        for k, v in model.state_dict().items()
    }
    grads: List[Optional[Tensor]] = [
        (torch.zeros_like(p, device="cpu") if p.requires_grad else None)
        for p in model.parameters()
    ]

    _move_opt_to(optimizer=fedopt_optimizer, device="cpu", state=state)
    _move_opt_to(optimizer=optimizer, device=state.current_device, state=state)

    cuda_sync_and_empty_cache()

    return origin_state_dict, grads


def _fedopt_step(
    model: nn.Module,
    optimizer: Optimizer,
    fedopt_optimizer: Optimizer,
    fedopt_lr_scheduler: Optional[LRScheduler],
    origin_state_dict: Mapping[str, Tensor],
    grads: List[Optional[Tensor]],
    state: DistributedState,
) -> Mapping[str, Tensor]:
    assert state.current_device is not None

    _move_opt_to(optimizer=optimizer, device="cpu", state=state)

    current_state_dict: Mapping[str, Tensor] = {
        k: v.detach().clone().to(device="cpu") for k, v in model.state_dict().items()
    }

    # Pseudo-gradient calculation
    g: Optional[Tensor]
    p_origin: Tensor
    p_current: Tensor
    for g, p_origin, p_current in zip(
        grads,
        origin_state_dict.values(),
        current_state_dict.values(),
        strict=True,
    ):
        if g is not None:
            g.copy_(p_origin.data - p_current.data).to(device=state.current_device)

    if logging.root.level == logging.DEBUG and state.rank == 0:
        total_norm: Tensor = tensor(
            sum((g.norm().item() if g is not None else torch.tensor(0)) for g in grads)
        )
        print(f"Norma dei cambiamenti: {total_norm}")

    # Loading last round weights and new pseudo-gradients
    model.load_state_dict(origin_state_dict, strict=True)
    for p, g in zip(model.parameters(), grads, strict=True):
        if p.requires_grad:
            assert g is not None
            p.grad = g.detach().to(device=state.current_device)

    if logging.root.level == logging.DEBUG and state.rank == 0:
        pre_clipping_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        print(f"Pre-clipping norm: {pre_clipping_norm}")

        total_norm: Tensor = torch.sqrt(
            tensor(
                sum(
                    (p.grad.norm() ** 2 if p.grad is not None else torch.tensor(0))
                    for p in model.parameters()
                )
            )
        )
        print(f"Norma dei gradienti: {total_norm}")

    # FedOpt optimizer step
    _move_opt_to(optimizer=fedopt_optimizer, device=state.current_device, state=state)
    fedopt_optimizer.step()
    fedopt_optimizer.zero_grad(set_to_none=True)
    if fedopt_lr_scheduler:
        fedopt_lr_scheduler.step()

    if logging.root.level == logging.DEBUG and state.rank == 0:
        diff = sum(
            (p.detach().clone().to("cpu") - p_before).norm().item()
            for p, p_before in zip(model.parameters(), current_state_dict.values())
        )
        print("Total change after optimizer.step():", diff)

    origin_state_dict = {
        k: v.detach().clone().to(device="cpu") for k, v in model.state_dict().items()
    }

    _move_opt_to(optimizer=fedopt_optimizer, device="cpu", state=state)
    _move_opt_to(optimizer=optimizer, device=state.current_device, state=state)

    cuda_sync_and_empty_cache()

    return origin_state_dict


def _get_average_train_loss(
    total_loss: Tensor, total_length: int, state: DistributedState
) -> Tensor:
    assert state.world_size is not None

    dist.all_reduce(
        tensor=total_loss,
        op=dist.ReduceOp.SUM,
    )

    if state.is_federated_scaling_setup():
        assert state.federated_local_size is not None
        assert state.federated_rank is not None
        train_epoch_loss: torch.Tensor = (
            total_loss / total_length
        ) / state.federated_local_size[state.federated_rank]
    else:
        train_epoch_loss: torch.Tensor = (total_loss / total_length) / state.world_size

    return train_epoch_loss


# --------------------------------------------------------------------------- #
#                               Training methods                              #
# --------------------------------------------------------------------------- #


def distributed_training(
    model: nn.Module | FullyShardedDataParallel,
    state: DistributedState,
    optimizer: Optimizer,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    wandb_run: Optional[Run] = None,
    fedopt: Optional[bool] = None,
    validate: Optional[bool] = None,
    epochs: Optional[int] = None,
    federated_batches: Optional[int] = None,
    save_path: Optional[PathLike] = None,
    output_model_name: Optional[str] = None,
    lr_scheduler: Optional[LRScheduler] = None,
    fedopt_lr_scheduler: Optional[LRScheduler] = None,
    fedopt_optimizer: Optional[Optimizer] = None,
    criterion: Optional[nn.Module] = None,
    gradient_clipping: Optional[float] = None,
    accumulation_steps: Optional[int] = None,
    config: Optional[XFFLConfig] = None,
) -> Mapping[str, float]:
    """Generic training cycle for FSDP models.

    The parameters can be provided both directly and through an XFFL configuration.
    In case both are provided, the firsts take the precedence.

    :param model: Model to train
    :type model: nn.Module | FullyShardedDataParallel
    :param state: xFFL distributed state
    :type state: DistributedState
    :param optimizer: Model's optimizer
    :type optimizer: Optimizer
    :param train_dataloader: Training set data loader
    :type train_dataloader: DataLoader
    :param val_dataloader: Validation set data loader, defaults to None
    :type val_dataloader: DataLoader
    :param fedopt: Activate FedOpt, defaults to None
    :type fedopt: bool, optional
    :param validate: Activate validation, defaults to None
    :type validate: bool, optional
    :param epochs: Number of epochs to train, defaults to None
    :type epochs: int, optional
    :param federated_batches: Number of training batched to process between two federated averaging, defaults to None
    :type federated_batches: Optional[int]
    :param save_path: Path where to save the trained model, defaults to None
    :type save_path: Optional[PathLike], optional
    :param output_model_name: Name to use for the saved trained model, defaults to None
    :type output_model_name: Optional[str], optional
    :param lr_scheduler: Learning rate scheduler, defaults to None
    :type lr_scheduler: Optional[LRScheduler], optional
    :param fedopt_lr_scheduler: Learning rate scheduler for the FedOpt optimizer, defaults to None
    :type fedopt_lr_scheduler: Optional[LRScheduler], optional
    :param fedopt_optimizer: FedOpt optimizer, defaults to None
    :type fedopt_optimizer: Optional[Optimizer], optional
    :param wandb_run: WandB run if wandb logging is desired, defaults to None
    :type wandb_run: Optional[wandb.Run], optional
    :param criterion: Loss function, defaults to None
    :type criterion: Optional[Callable], optional
    :param gradient_clipping: Gradient clipping value, defaults to None
    :type gradient_clipping: Optional[float], optional
    :param accumulation_steps: Gradient accumulation steps, defaults to None
    :type accumulation_steps: Optional[int], optional
    :param config: XFFL configuration
    :type config: Optional[XFFLConfig], defaults to None
    :return: Dictionary of metrics names and achieved values
    :rtype: Mapping[str, float]
    """

    # Parameters resolution
    _fedopt: Optional[bool] = resolve_param(value=fedopt, config=config, attr="fedopt")
    if _fedopt and not state.is_federated_scaling_setup():
        logger.warning(
            "FedOpt is active, but FederatedScaling is not setup: running FedOpt with a single model replica."
        )
    _validate: Optional[bool] = resolve_param(
        value=validate, config=config, attr="validate"
    )
    if _validate and not val_dataloader:
        logger.warning(
            "Model validation is active, but no validation set dataloader is found. Validation will not be performed."
        )
    _epochs: Optional[int] = resolve_param(value=epochs, config=config, attr="epochs")
    if _epochs is None or _epochs < 1:
        logger.error(
            f"Selected number of epochs ({_epochs}) not acceptable - defaulting to 1."
        )
        _epochs = 1
    _federated_batches: Optional[int] = resolve_param(
        value=federated_batches, config=config, attr="federated_batches"
    )
    if _federated_batches and not state.is_federated_scaling_setup():
        logger.warning(
            "Federated batches is setup, but FederatedScaling is not: running FederatedScaling with a single model replica."
        )
    _save_path: Optional[PathLike] = resolve_param(
        value=save_path, config=config, attr="save_path"
    )
    if _save_path is not None and (
        not os.path.exists(_save_path) or not os.path.isdir(_save_path)
    ):
        logger.warning(
            f"Impossible saving the trained model with save dir {_save_path}: saving disabled."
        )
        _save_path = None
    _output_model_name: Optional[str] = resolve_param(
        value=output_model_name, config=config, attr="output_model_name"
    )
    if _save_path is not None and _output_model_name is None:
        logger.warning("No output model name specified - defaulting to 'output'.")
        _output_model_name = "output"
    elif _save_path is None and _output_model_name is not None:
        logger.error(
            "Output model name is provided, but save path is not - disabling model saving."
        )
        _output_model_name = None
    _lr_scheduler: Optional[LRScheduler] = resolve_param(
        value=lr_scheduler, config=config, attr="lr_scheduler"
    )
    _fedopt_optimizer: Optional[Optimizer] = resolve_param(
        value=fedopt_optimizer, config=config, attr="fedopt_optimizer"
    )
    if _fedopt_optimizer and not _fedopt:
        logger.warning(
            "A FedOpt optimizer is specified, but FedOpt is not setup; it will be ignored."
        )
        _fedopt_optimizer = None
    elif _fedopt and not _fedopt_optimizer:
        logger.error(
            "FedOpt is active, but no FedOpt optimizer is specified; FedOpt will be disabled."
        )
        _fedopt = None
    _fedopt_lr_scheduler: Optional[LRScheduler] = resolve_param(
        value=fedopt_lr_scheduler, config=config, attr="fedopt_lr_scheduler"
    )
    if _fedopt_lr_scheduler and not (_fedopt and _fedopt_optimizer):
        logger.error(
            "A FedOpt learning rate scheduler is specified, but FedOpt and/or the FedOpt optimizer are not setup; it will be ignored."
        )
        _fedopt_lr_scheduler = None
    _criterion: Optional[nn.Module] = resolve_param(
        value=criterion, config=config, attr="criterion"
    )
    _gradient_clipping: Optional[float] = resolve_param(
        value=gradient_clipping, config=config, attr="gradient_clipping"
    )
    if _gradient_clipping and _gradient_clipping <= 0.0:
        logger.error(
            f"Gradient clipping is set to {_gradient_clipping}, which is not acceptable. Defaulting to 1.0."
        )
        _gradient_clipping = 1.0
    _accumulation_steps: Optional[int] = resolve_param(
        value=accumulation_steps, config=config, attr="accumulation_steps"
    )
    if _accumulation_steps and _accumulation_steps < 1:
        logger.error(
            f"Gradient accumulation steps is set to {_accumulation_steps}, which is not acceptable. Defaulting to 1."
        )
        _accumulation_steps = 1

    # Clear GPU cache and reset peak memory stats
    cuda_reset_memory_stats_and_empty_cache()

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

    optimizer_step: int = 0
    aggregation: int = 0

    # FedOpt setup
    if _fedopt:
        assert _fedopt_optimizer is not None

        fedopt_origin_state_dict: Mapping[str, Tensor]
        fedopt_grads: List[Optional[Tensor]]
        fedopt_origin_state_dict, fedopt_grads = _fedopt_setup(
            model=model,
            optimizer=optimizer,
            fedopt_optimizer=_fedopt_optimizer,
            state=state,
        )

    # Epoch training cycle
    for epoch in range(_epochs):
        epoch_start_time: float = time.perf_counter()
        logger.info(f"Starting epoch {epoch}/{_epochs}")

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

        step: int
        batch: Dict[str, Any]
        optimizer.zero_grad(set_to_none=True)
        train_function: Callable = _get_train_function(next(iter(train_dataloader)))

        # Batch training cycle
        for step, batch in enumerate(train_dataloader):
            if logging.root.level == logging.DEBUG:
                cuda_sync()
                start_time = time.perf_counter()

            # Forward
            loss: torch.Tensor = train_function(
                model=model, batch=batch, state=state, criterion=_criterion
            )

            if logging.root.level == logging.DEBUG:
                cuda_sync()
                batch_time = time.perf_counter() - start_time
                start_time = time.perf_counter()

            # Backward
            if _accumulation_steps is not None:
                loss /= _accumulation_steps
            loss.backward()

            if logging.root.level == logging.DEBUG:
                cuda_sync()
                back_time = time.perf_counter() - start_time
                start_time = time.perf_counter()

            # Optimization
            if (
                _accumulation_steps is None
                or (step + 1) % _accumulation_steps == 0
                or (step + 1) == total_length
            ):
                if _gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), _gradient_clipping
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if _lr_scheduler:
                    _lr_scheduler.step()
                optimizer_step += 1

            # TQDM update
            pbar.update(1)

            if logging.root.level == logging.DEBUG:
                cuda_sync()
                optimizer_time = time.perf_counter() - start_time
                start_time = time.perf_counter()

            # FederatedScaling
            if state.is_federated_scaling_setup():
                assert _federated_batches is not None
                if (step + 1) % _federated_batches == 0 or (step + 1) == total_length:

                    # Aggregation
                    with torch.no_grad():
                        bucket_optimized_coalesced_(
                            model=model,
                            state=state,
                            use_multiple_cuda_streams=False,
                            use_contiguous_memory=True,
                        )
                        if _fedopt:
                            assert _fedopt_optimizer is not None

                            fedopt_origin_state_dict = _fedopt_step(
                                model=model,
                                optimizer=optimizer,
                                fedopt_optimizer=_fedopt_optimizer,
                                fedopt_lr_scheduler=_fedopt_lr_scheduler,
                                origin_state_dict=fedopt_origin_state_dict,
                                grads=fedopt_grads,
                                state=state,
                            )
                        aggregation += 1

            if logging.root.level == logging.DEBUG:
                cuda_sync()
                comm_time = time.perf_counter() - start_time
                start_time = time.perf_counter()

            # Logging
            if _accumulation_steps is not None:
                loss *= _accumulation_steps

            total_loss += loss.detach().float()
            train_step_perplexity.append(float(torch.exp(loss.detach().float())))
            train_step_loss.append(loss.detach().float().item())

            # WandB
            if wandb_run:
                metrics: Mapping[str, Any] = {
                    "train/epoch": epoch + 1,
                    "train/step": epoch * total_length + step,
                    "train/loss": train_step_loss[-1],
                    "train/perplexity": train_step_perplexity[-1],
                    "train/optimizer step": optimizer_step,
                    "train/learning rate": (
                        _lr_scheduler.get_lr()
                        if _lr_scheduler is not None
                        else optimizer.param_groups[0]["lr"]
                    ),
                }
                if state.is_federated_scaling_setup():
                    metrics["train/aggregation"] = aggregation
                    if _fedopt:
                        assert _fedopt_lr_scheduler is not None
                        assert _fedopt_optimizer is not None

                        metrics["train/FedOpt learning rate"] = (
                            (
                                _fedopt_lr_scheduler.get_lr()
                                if _lr_scheduler is not None
                                else _fedopt_optimizer.param_groups[0]["lr"]
                            ),
                        )
                wandb_run.log(metrics)

            pbar.set_description(
                f"Training Epoch: {epoch + 1}/{_epochs}, step {step+1}/{total_length} completed (loss: {train_step_loss[-1]:.4f})",
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

        pbar.close()
        epoch_times.append(time.perf_counter() - epoch_start_time)

        train_epoch_loss = _get_average_train_loss(
            total_loss=total_loss, total_length=total_length, state=state
        )
        train_perplexity: torch.Tensor = torch.exp(train_epoch_loss)
        train_perp.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        # Validation
        if _validate:
            assert val_dataloader is not None

            val_ppl, val_epoch_loss, temp_val_loss, temp_step_perplexity, eval_acc = (
                validation(
                    model=model,
                    eval_dataloader=val_dataloader,
                    state=state,
                    wandb_run=wandb_run,
                    criterion=_criterion,
                )
            )
            val_step_loss.extend(temp_val_loss)
            val_step_perplexity.extend(temp_step_perplexity)

            if val_epoch_loss < best_val_loss:
                best_val_loss: torch.Tensor = val_epoch_loss
                if state.rank == 0:
                    logger.info(f"Best eval loss on epoch {epoch+1} is {best_val_loss}")
            val_loss.append(float(best_val_loss))
            val_perp.append(float(val_ppl))

        # Model saving
        if _output_model_name is not None and _save_path is not None:
            assert state.rank is not None

            checkpoint_start_time = time.perf_counter()
            save_model(
                model=model,
                optimizer=optimizer,
                path=_save_path,
                name=_output_model_name,
                rank=state.rank,
                epoch=epoch,
            )
            checkpoint_times.append(time.perf_counter() - checkpoint_start_time)
            if logging.root.level == logging.DEBUG and state.rank == 0:
                logger.debug(f"Checkpoint time: {checkpoint_times[-1]}")

        if state.rank == 0:
            if _validate:
                logger.info(
                    f"Epoch {epoch+1}:\n\ttrain_perplexity={train_perplexity:.4f},\n\ttrain_epoch_loss={train_epoch_loss:.4f},\n\tepoch time {epoch_times[-1]:.2f}s,\n\taccuracy={eval_acc:.2f}%"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1}:\n\ttrain_perplexity={train_perplexity:.4f},\n\ttrain_epoch_loss={train_epoch_loss:.4f},\n\tepoch time {epoch_times[-1]:.2f}s"
                )

    # Results dictionary # TODO: not sure about this
    results["Epoch time (avg)"] = sum(epoch_times) / len(epoch_times)
    results["Train perplexity (avg)"] = sum(train_perp) / len(train_perp)
    results["Train loss (avg)"] = sum(train_loss) / len(train_loss)
    if _validate:
        results["Validation perplexity (avg)"] = sum(val_perp) / len(val_perp)
        results["Validation loss (avg)"] = sum(val_loss) / len(val_loss)
        if eval_acc is not None:
            results["Validation accuracy"] = eval_acc

    return results


def validation(
    model: nn.Module,
    eval_dataloader: DataLoader,
    state: DistributedState,
    wandb_run: Optional[Run] = None,
    criterion: Optional[nn.Module] = None,
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
            loss: torch.Tensor | float

            if isinstance(batch, dict):
                for key in batch.keys():
                    batch[key] = batch[key].to(
                        device=torch.cuda.current_device(), non_blocking=True
                    )

                outputs = model(**batch)
                loss = outputs.loss
                assert isinstance(loss, Tensor)

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
