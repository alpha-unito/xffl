"""Training utilities"""

import logging
import os
import time
from logging import Logger, getLogger
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn, tensor
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from wandb.wandb_run import Run

from xffl.custom.config import XFFLConfig
from xffl.distributed.aggregation import (
    bucket_optimized_coalesced_,
    get_average_distributed_loss,
)
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
    """Moves an optimizer to the desired device.
    If training is happening on CPU this method has no effect.

    :param optimizer: Optimizer to be moved
    :type optimizer: Optimizer
    :param device: Device on which to move the optimizer
    :type device: torch.device | str
    :param state: xFFL distributed state
    :type state: DistributedState
    """
    if state.device_type != "cpu":
        for param in optimizer.state.values():
            for k, v in param.items():
                if torch.is_tensor(v):
                    param[k] = v.to(device, non_blocking=True)


def _get_processing_function(
    batch: Any,
) -> Callable[[nn.Module, Any, DistributedState], Tuple[Tensor, Tensor]]:
    """Returns the appropriate processing function for the provided batch data type.

    :param batch: An exemplary data batch from a dataloader
    :type batch: Mapping[Any, Any] | Tuple[Any, Any]
    :return: A function with parameters model, batch, state, returning the processing output and eventual target
    :rtype: Callable[[nn.Module, Any, DistributedState], Tuple[Tensor, Tensor]]
    """

    train_function: Callable

    def attention_mask_processing(
        model: nn.Module,
        batch: Any,
        state: DistributedState,
    ) -> Tuple[Any, Optional[Tensor]]:

        for key in batch.keys():
            batch[key] = batch[key].to(
                device=state.current_device,
                non_blocking=True,
            )
        output: Any = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["input_ids"],
        )
        return output, None

    def dict_processing(
        model: nn.Module,
        batch: Dict[str, Any],
        state: DistributedState,
    ) -> Tuple[Any, Optional[Tensor]]:
        for key in batch.keys():
            batch[key] = batch[key].to(
                device=state.current_device,
                non_blocking=True,
            )
        output: Any = model(**batch)
        return output, None

    def criterion_processing(
        model: nn.Module,
        batch: Tuple[Any, Any],
        state: DistributedState,
    ) -> Tuple[Any, Optional[Tensor]]:
        data: Tensor
        target: Tensor
        data, target = batch
        data, target = data.to(
            device=state.current_device,
            non_blocking=True,
        ), target.to(
            device=state.current_device,
            non_blocking=True,
        )
        output: Tensor = model(data)
        return output, target

    if isinstance(batch, dict):
        if all(key in batch for key in ["input_ids", "attention_mask"]):
            train_function = attention_mask_processing
        else:
            train_function = dict_processing
    else:
        train_function = criterion_processing

    return train_function


def _fedopt_setup(
    model: nn.Module,
    optimizer: Optimizer,
    fedopt_optimizer: Optimizer,
    state: DistributedState,
) -> Tuple[Mapping[str, Tensor], List[Optional[Tensor]]]:
    """Sets up the necessary FedOpt information.
    Records the initial parameter configuration and creates a list of empty pseudo-gradients.

    :param model: Model to train
    :type model: nn.Module
    :param optimizer: Inner (local) optimizer
    :type optimizer: Optimizer
    :param fedopt_optimizer: Outer (server) optimizer
    :type fedopt_optimizer: Optimizer
    :param state: xFFL distributed state
    :type state: DistributedState
    :return: Starting state dict and empty pseudo-gradients
    :rtype: Tuple[Mapping[str, Tensor], List[Optional[Tensor]]]
    """
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
    """Executes a FedOpt optimization step.

    :param model: Model to train
    :type model: nn.Module
    :param optimizer: Inner (local) optimizer
    :type optimizer: Optimizer
    :param fedopt_optimizer: Outer (server) optimizer
    :type fedopt_optimizer: Optimizer
    :param fedopt_lr_scheduler: Outer (server) optimizer learning rate scheduler
    :type fedopt_lr_scheduler: Optional[LRScheduler]
    :param origin_state_dict: Previous FedOpt step state dict
    :type origin_state_dict: Mapping[str, Tensor]
    :param grads: Empty pseudo-gradients
    :type grads: List[Optional[Tensor]]
    :param state: xFFL distributed state
    :type state: DistributedState
    :return: _description_
    :rtype: Mapping[str, Tensor]
    """
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
            (p.detach().clone().to(device="cpu") - p_before).norm().item()
            for p, p_before in zip(model.parameters(), current_state_dict.values())
        )
        print("Total change after optimizer.step():", diff)

    new_state_dict: Mapping[str, Tensor] = {
        k: v.detach().clone().to(device="cpu") for k, v in model.state_dict().items()
    }

    _move_opt_to(optimizer=fedopt_optimizer, device="cpu", state=state)
    _move_opt_to(optimizer=optimizer, device=state.current_device, state=state)

    cuda_sync_and_empty_cache()

    return new_state_dict


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
    epochs: Optional[int] = None,
    federated_batches: Optional[int] = None,
    save_path: Optional[Path] = None,
    output_model_name: Optional[str] = None,
    lr_scheduler: Optional[Callable] = None,
    fedopt_lr_scheduler: Optional[LRScheduler] = None,
    fedopt_optimizer: Optional[Optimizer] = None,
    criterion: Optional[nn.Module] = None,
    gradient_clipping: Optional[float] = None,
    gradient_accumulation: Optional[int] = None,
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
    :type save_path: Optional[Path], optional
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
    :param gradient_accumulation: Gradient accumulation steps, defaults to None
    :type gradient_accumulation: Optional[int], optional
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
    _save_path: Optional[Path] = resolve_param(
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
    __lr_scheduler: Optional[Callable] = resolve_param(
        value=lr_scheduler, config=config, attr="lr_scheduler"
    )
    _lr_scheduler: Optional[LRScheduler] = (
        __lr_scheduler(
            optimizer=optimizer, total_steps=len(train_dataloader), config=config
        )
        if __lr_scheduler is not None
        else __lr_scheduler
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
    _gradient_accumulation: Optional[int] = resolve_param(
        value=gradient_accumulation, config=config, attr="gradient_accumulation"
    )
    if _gradient_accumulation and _gradient_accumulation < 1:
        logger.error(
            f"Gradient accumulation steps is set to {_gradient_accumulation}, which is not acceptable. Defaulting to 1."
        )
        _gradient_accumulation = 1

    # Clear GPU cache and reset peak memory stats
    cuda_reset_memory_stats_and_empty_cache()

    # TODO: if fp16 the gradients should be rescaled
    train_perp: List[float] = []
    train_loss: List[float] = []
    val_perp: List[float] = []
    val_loss: List[float] = []
    val_acc: List[float] = []
    epoch_times: List[float] = []
    val_epoch_times: List[float] = []
    checkpoint_times: List[float] = []
    results: Dict[str, float] = {}

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
    epoch: int
    optimizer.zero_grad(set_to_none=True)
    train_function: Callable = _get_processing_function(next(iter(train_dataloader)))
    for epoch in range(_epochs):
        epoch_start_time: float = time.perf_counter()
        if state.rank == 0:
            logger.info(f" --- Starting epoch {epoch+1}/{_epochs} --- ")

        train_step_perplexity: List[float] = []
        train_step_loss: List[float] = []
        val_step_loss: List[float] = []
        val_step_perplexity: List[float] = []

        model.train()

        train_epoch_loss: Tensor = tensor(0.0)
        total_length: int = len(train_dataloader)
        pbar: tqdm = tqdm(
            colour="blue",
            desc=f"Training Epoch: {epoch+1}",
            total=total_length,
            dynamic_ncols=True,
            disable=state.rank != 0,
        )

        # Batch training cycle
        step: int
        batch: Dict[str, Any]
        for step, batch in enumerate(train_dataloader):
            if logging.root.level == logging.DEBUG:
                cuda_sync()
                step_start_time: float = time.perf_counter()
                start_time: float = time.perf_counter()

            # Forward
            output: Any
            target: Tensor
            output, target = train_function(model=model, batch=batch, state=state)
            if _criterion:
                loss: Tensor = _criterion(output, target)
            else:
                loss: Tensor = output.loss

            if logging.root.level == logging.DEBUG:
                cuda_sync()
                batch_time: float = time.perf_counter() - start_time
                start_time = time.perf_counter()

            # Backward
            if _gradient_accumulation is not None:
                loss /= _gradient_accumulation
            loss.backward()

            if logging.root.level == logging.DEBUG:
                cuda_sync()
                back_time: float = time.perf_counter() - start_time
                start_time = time.perf_counter()

            # Optimization
            if (
                _gradient_accumulation is None
                or (step + 1) % _gradient_accumulation == 0
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
                optimizer_time: float = time.perf_counter() - start_time
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
                            use_contiguous_memory=False,
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
                comm_time: float = time.perf_counter() - start_time
                start_time = time.perf_counter()

            # Logging
            if _gradient_accumulation is not None:
                loss *= _gradient_accumulation

            train_epoch_loss += loss.detach().float().cpu()
            train_step_perplexity.append(float(torch.exp(loss.detach().float().cpu())))
            train_step_loss.append(loss.detach().float().cpu().item())

            pbar.set_description(
                f"Training Epoch: {epoch + 1}/{_epochs},   step {step+1}/{total_length} completed (loss: {train_step_loss[-1]:.4f})",
                refresh=True,
            )

            if logging.root.level == logging.DEBUG:
                assert batch_time
                assert back_time
                assert optimizer_time
                assert comm_time

                cuda_sync()
                overall_step_time: float = time.perf_counter() - step_start_time
                other_step_time: float = overall_step_time - (
                    batch_time + back_time + optimizer_time + comm_time
                )

            if logging.root.level == logging.DEBUG:
                pbar.set_postfix(
                    ordered_dict={
                        "F": f"{batch_time:.2f}",
                        "B": f"{back_time:.2f}",
                        "O": f"{optimizer_time:.2f}",
                        "A": f"{comm_time:.2f}",
                        "M": f"{other_step_time:.2f}",
                        "T": f"{overall_step_time:.2f}",
                    },
                    refresh=True,
                )

            # WandB
            if wandb_run:
                metrics: Mapping[str, Any] = {
                    "train/Step": epoch * total_length + step,
                    "train/Step loss": train_step_loss[-1],
                    "train/Step perplexity": train_step_perplexity[-1],
                    "train/Optimizer step": optimizer_step,
                    "train/Learning rate": (
                        _lr_scheduler.get_lr()
                        if _lr_scheduler is not None
                        else optimizer.param_groups[0]["lr"]
                    ),
                }
                if state.is_federated_scaling_setup():
                    metrics["train/Aggregation step"] = aggregation
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
                if logging.root.level == logging.DEBUG:

                    metrics.update(
                        {
                            "train/Forward time": batch_time,
                            "train/Backward time": back_time,
                            "train/Optimizer time": optimizer_time,
                            "train/Aggregation time": comm_time,
                            "train/Other time": other_step_time,
                            "train/Overall step time": overall_step_time,
                        }
                    )
                wandb_run.log(metrics)

        pbar.close()
        epoch_times.append(time.perf_counter() - epoch_start_time)

        _train_epoch_loss: Tensor = get_average_distributed_loss(
            loss=train_epoch_loss, total_length=total_length, state=state
        )
        train_epoch_perplexity: Tensor = torch.exp(_train_epoch_loss)
        train_perp.append(float(train_epoch_perplexity))
        train_loss.append(float(_train_epoch_loss))

        if wandb_run:
            metrics: Mapping[str, Any] = {
                "train/Epoch": epoch + 1,
                "train/Epoch loss": _train_epoch_loss,
                "train/Epoch perplexity": train_epoch_perplexity,
                "train/Epoch time": epoch_times[-1],
            }
            wandb_run.log(
                metrics,
                commit=False,
            )

        # Validation
        if val_dataloader is not None:
            (
                val_epoch_loss,
                val_epoch_perplexity,
                _val_step_loss,
                _val_step_perplexity,
                val_epoch_time,
                _val_acc,
            ) = validation(
                model=model,
                val_dataloader=val_dataloader,
                state=state,
                epoch=epoch,
                epochs=_epochs,
                wandb_run=wandb_run,
                criterion=_criterion,
            )
            val_step_loss.extend(_val_step_loss)
            val_step_perplexity.extend(_val_step_perplexity)
            val_epoch_times.append(val_epoch_time)

            val_loss.append(float(val_epoch_loss))
            val_perp.append(float(val_epoch_perplexity))
            if _val_acc is not None:
                val_acc.append(_val_acc)

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
            log_message: str = (
                f"Epoch {epoch+1}:\n\t"
                + f"Train time:\t\t{epoch_times[-1]:.2f}s\n\t"
                + f"Train loss:\t\t{_train_epoch_loss:.2f}\n\t"
                + f"Train perplexity:\t{train_epoch_perplexity:.2f}\n\t"
            )
            if val_dataloader is not None:
                log_message = (
                    log_message
                    + f"Validation time:\t{val_epoch_time:.2f}s\n\t"
                    + f"Validation loss:\t{val_epoch_loss:.2f}\n\t"
                    + f"Validation perplexity:\t{val_epoch_perplexity:.2f}\n\t"
                )
                if _val_acc is not None:
                    log_message += f"Validation accuracy:\t{_val_acc:.2f}%"
            logger.info(log_message + "\n")

    # Results dictionary
    results["Epoch time (avg):\t\t"] = sum(epoch_times) / len(epoch_times)
    results["Train loss (avg):\t\t"] = sum(train_loss) / len(train_loss)
    results["Train perplexity (avg):\t"] = sum(train_perp) / len(train_perp)
    if val_dataloader is not None:
        results["Validation epoch time (avg):\t"] = sum(val_epoch_times) / len(
            val_epoch_times
        )
        results["Validation loss (avg):\t\t"] = sum(val_loss) / len(val_loss)
        results["Validation perplexity (avg):\t"] = sum(val_perp) / len(val_perp)
        if _val_acc is not None:
            results["Validation accuracy (avg):\t"] = sum(val_acc) / len(val_acc)

    return results


def validation(
    model: nn.Module,
    val_dataloader: DataLoader,
    state: DistributedState,
    epoch: int,
    epochs: int,
    wandb_run: Optional[Run] = None,
    criterion: Optional[nn.Module] = None,
) -> Tuple[Tensor, Tensor, List[float], List[float], float, Optional[float]]:
    """Generic evaluation cycle for FSDP models

    :param model: Model to evaluate
    :type model: nn.Module
    :param val_dataloader: Validation dataset data loader
    :type val_dataloader: DataLoader
    :param state: Instantiated distributed state
    :type state: DistributedState
    :param wandb_run: WandB run if wandb logging is desired, defaults to None
    :type wandb_run: Optional[wandb.Run], optional
    :param criterion: Loss function, defaults to None
    :type criterion: Optional[Callable], optional
    :return: Total epoch loss, total epoch perplexity, per-step loss, per-step perplexity, overall accuracy
    :rtype: Tuple[Tensor, Tensor, List[float], List[float], Optional[float]]
    """

    model.eval()

    val_step_loss: List[float] = []
    val_step_perplexity: List[float] = []
    val_epoch_loss: Tensor = tensor(0.0)
    correct: Optional[int] = None

    val_function: Callable = _get_processing_function(next(iter(val_dataloader)))
    total_length: int = len(val_dataloader)

    pbar: tqdm = tqdm(
        val_dataloader,
        colour="green",
        total=total_length,
        desc=f"Validation Epoch: {epoch+1}",
        dynamic_ncols=True,
        disable=state.rank != 0,
    )

    with torch.no_grad():
        epoch_start_time: float = time.perf_counter()

        # Validation
        step: int
        batch: Dict[str, Any]
        for step, batch in enumerate(val_dataloader):
            output: Any
            target: Tensor
            output, target = val_function(model=model, batch=batch, state=state)
            if criterion:
                loss: Tensor = criterion(output, target)
            else:
                loss: Tensor = output.loss

            # Metrics
            val_epoch_loss += loss.detach().float().cpu()
            val_step_perplexity.append(float(torch.exp(loss.detach().float().cpu())))
            val_step_loss.append(loss.detach().float().cpu().item())
            if target:
                pred: Tensor = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability

                if correct is None:
                    correct = 0
                correct += int(pred.eq(target.view_as(pred)).sum().item())

            # TQDM update
            pbar.update(1)
            pbar.set_description(
                f"Validation Epoch: {epoch + 1}/{epochs}, step {step+1}/{total_length} completed (loss: {val_step_loss[-1]:.4f})",
                refresh=True,
            )

        _val_epoch_loss: Tensor = get_average_distributed_loss(
            loss=val_epoch_loss, total_length=total_length, state=state
        )
        val_epoch_perplexity: Tensor = torch.exp(_val_epoch_loss)
        if correct is not None:
            assert val_dataloader.batch_size is not None

            val_acc: float = (100.0 * correct) / (
                total_length * val_dataloader.batch_size
            )

        epoch_total_time: float = time.perf_counter() - epoch_start_time

        if wandb_run:
            metrics: Mapping[str, Any] = {
                "train/Epoch": epoch + 1,
                "eval/Epoch loss": _val_epoch_loss,
                "eval/Epoch perplexity": val_epoch_perplexity,
                "train/Epoch time": epoch_total_time,
            }
            if correct is not None:
                metrics["eval/accuracy"] = val_acc

            wandb_run.log(
                metrics,
                commit=False,
            )

    return (
        _val_epoch_loss,
        val_epoch_perplexity,
        val_step_loss,
        val_step_perplexity,
        epoch_total_time,
        val_acc,
    )
