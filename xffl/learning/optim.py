"""Optimization utilities"""

import math
from logging import Logger, getLogger
from typing import Any, Callable, List, Mapping, Optional

import torch
from torch import GradScaler, nn
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from xffl.custom.config import XFFLConfig
from xffl.utils.utils import resolve_param

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""

# --------------------------------------------------------------------------- #
#                               Helper functions                              #
# --------------------------------------------------------------------------- #


def _is_optimizer_accumulation_step_time(
    gradient_accumulation: Optional[int], step: int, total_steps_per_epoch: int
) -> bool:
    """Returns true if the gradient accumulation is terminated and its time to step, False otherwise.

    :param gradient_accumulation: Gradient accumulation steps
    :type gradient_accumulation: Optional[int]
    :param step: Current training epoch step
    :type step: int
    :param total_steps_per_epoch: Number of training steps per epoch
    :type total_steps_per_epoch: int
    :return: True if the gradient accumulation is terminated and its time to step, False otherwise
    :rtype: bool
    """
    return (
        gradient_accumulation is None
        or (step + 1) % gradient_accumulation == 0
        or (step + 1) == total_steps_per_epoch
    )


# --------------------------------------------------------------------------- #
#                               Main methods                                  #
# --------------------------------------------------------------------------- #


def warmup_cosine_decay(
    optimizer: Optimizer | Mapping[nn.Parameter, Optimizer],
    total_steps_per_epoch: int,
    epochs: Optional[int] = None,
    gradient_accumulation: Optional[int] = None,
    lr_scheduler_params: Optional[Mapping[str, Any]] = None,
    config: Optional[XFFLConfig] = None,
) -> LRScheduler | Mapping[nn.Parameter, LRScheduler]:
    """Creates a Warmup + Cosine Decay learning rate scheduler.

    The scheduler operates on optimizer steps and is compatible with gradient
    accumulation. Training steps are internally converted into optimizer steps.

    Learning rate schedule:
        1) Linear warmup from 0 -> peak learning rate
        2) Cosine decay from peak learning rate -> final learning rate

    :param optimizer: Optimizer or mapping of parameters and optimizers
    :type optimizer: Optimizer|Mapping[nn.Parameter, Optimizer]
    :param total_steps_per_epoch: Number of training steps per epoch (before gradient accumulation)
    :type total_steps_per_epoch: int
    :param epochs: Number of epochs to train, defaults to None
    :type epochs: int, optional
    :param gradient_accumulation: Gradient accumulation steps, defaults to None
    :type gradient_accumulation: Optional[int], optional
    :param lr_scheduler_params: Learning rate parameters, defaults to None
    :type lr_scheduler_params: Optional[Mapping[str, Any]], optional
    :param config: xFFL training configuration
    :type config: XFFLConfig
    :return: Configured warmup + cosine decay scheduler
    :rtype: LRScheduler|Mapping[nn.Parameter, LRScheduler]
    """

    # Resolve parameters
    _epochs: Optional[int] = resolve_param(value=epochs, config=config, attr="epochs")
    if _epochs is None:
        _epochs = 1
    _gradient_accumulation: Optional[int] = resolve_param(
        value=gradient_accumulation, config=config, attr="gradient_accumulation"
    )
    if _gradient_accumulation is None:
        _gradient_accumulation = 1
    _lr_scheduler_params: Optional[Mapping[str, Any]] = resolve_param(
        value=lr_scheduler_params, config=config, attr="lr_scheduler_params"
    )
    if _lr_scheduler_params is None:
        _lr_scheduler_params = {}

    if total_steps_per_epoch <= 0:
        raise ValueError("total_steps_per_epoch must be > 0")
    if _gradient_accumulation <= 0:
        raise ValueError("gradient_accumulation must be > 0")

    class WarmupCosineScheduler(LRScheduler):
        """Linear warmup followed by cosine decay scheduler.

        Must be stepped after each optimizer.step()

        :param optimizer: Optimizer whose learning rate will be updated
        :type optimizer: Optimizer
        :param epochs: Number of training epochs
        :type epochs: int
        :param accum_steps: Gradient accumulation steps
        :type accum_steps: int
        :param peak_lr: Maximum learning rate
        :type peak_lr: float
        :param steps_per_epoch: Training steps per epoch (before accumulation)
        :type steps_per_epoch: int
        :param warmup_fraction: Fraction of total steps used for warmup, defaults to 0.01
        :type warmup_fraction: float, optional
        :param final_lr_ratio: Final LR as fraction of peak LR, defaults to 0.1
        :type final_lr_ratio: float, optional
        """

        def __init__(
            self,
            optimizer: Optimizer,
            epochs: int,
            accum_steps: int,
            peak_lr: float,
            steps_per_epoch: int,
            warmup_fraction: float = 0.01,
            final_lr_ratio: float = 0.01,
        ) -> None:

            self.peak_lr = peak_lr
            self.final_lr = peak_lr * final_lr_ratio

            # Convert training steps to optimizer steps
            effective_steps_per_epoch = steps_per_epoch // accum_steps
            self.total_steps = max(1, epochs * effective_steps_per_epoch)
            self.warmup_steps = max(1, int(self.total_steps * warmup_fraction))

            super().__init__(optimizer)

            logger.debug(
                "Initialized WarmupCosineScheduler: "
                f"total_steps={self.total_steps}, warmup_steps={self.warmup_steps}, "
                f"peak_lr={self.peak_lr:.2e}, final_lr={self.final_lr:.2e}"
            )

        def _lr_at_step(self, step: int) -> float:
            """Closed-form LR computation used by PyTorch internals."""
            if step >= self.total_steps:
                return self.final_lr

            if step < self.warmup_steps:
                return self.peak_lr * (step / self.warmup_steps)

            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            progress = min(max(progress, 0.0), 1.0)

            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.final_lr + (self.peak_lr - self.final_lr) * cosine_decay

        def get_lr(self) -> List[float]:
            """Compute learning rates for each param group."""
            step = self.last_epoch + 1  # PyTorch convention
            lr = self._lr_at_step(step)
            return [lr for _ in self.optimizer.param_groups]

        # Optional but recommended for PyTorch compatibility
        def _get_closed_form_lr(self) -> List[float]:
            """Closed-form LR for state_dict compatibility."""
            lr = self._lr_at_step(self.last_epoch)
            return [lr for _ in self.optimizer.param_groups]

    lr_scheduler: LRScheduler | Mapping[nn.Parameter, LRScheduler]
    if isinstance(optimizer, Optimizer):
        lr_scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            epochs=_epochs,
            accum_steps=_gradient_accumulation,
            peak_lr=optimizer.param_groups[0]["lr"],
            steps_per_epoch=total_steps_per_epoch,
            **_lr_scheduler_params,
        )
    else:
        lr_scheduler = {}
        p: nn.Parameter
        optim: Optimizer
        for p, optim in optimizer.items():
            lr_scheduler[p] = WarmupCosineScheduler(
                optimizer=optim,
                epochs=_epochs,
                accum_steps=_gradient_accumulation,
                peak_lr=optim.param_groups[0]["lr"],
                steps_per_epoch=total_steps_per_epoch,
                **_lr_scheduler_params,
            )

    return lr_scheduler


class XFFLOptimizer:
    """Creates the specified XFFL optimizer configured for FM pretraining.

    :param model: Model to train
    :type model: nn.Module | FullyShardedDataParallel
    :param optimizer: Optimizer class, defaults to None
    :type optimizer: Optional[Callable], None
    :param optimizer_params: Optimizer parameters, defaults to None
    :type optimizer_params: Optional[Mapping[str, Any]], optional
    :param gradient_clipping: Gradient clipping value, defaults to None
    :type gradient_clipping: Optional[float], optional
    :param gradient_accumulation: Gradient accumulation steps, defaults to None
    :type gradient_accumulation: Optional[int], optional
    :param interleaved_optim: Interleave optimizer and backward phase, defaults to None
    :type interleaved_optim: bool, optional
    :param lr_scheduler: Learning rate scheduler, defaults to None
    :type lr_scheduler: Optional[LRScheduler], optional
    :param total_steps_per_epoch: Number of training steps per epoch (before gradient accumulation), defaults to -1
    :type total_steps_per_epoch: int
    :param scaler: Gradient scaler, if necessary
    :type scaler: Optional[GradScaler]
    :param config: XFFL configuration, defaults to None
    :type config: Optional[XFFLConfig], optional
    :raises ValueError: If some configuration values are incompatible with their expected values
    """

    def __init__(
        self,
        model: nn.Module | FullyShardedDataParallel,
        optimizer: Optional[Callable] = None,
        optimizer_params: Optional[Mapping[str, Any]] = None,
        gradient_clipping: Optional[float] = None,
        gradient_accumulation: Optional[int] = None,
        interleaved_optim: Optional[bool] = None,
        lr_scheduler: Optional[Callable] = None,
        total_steps_per_epoch: int = -1,
        scaler: Optional[GradScaler] = None,
        config: Optional[XFFLConfig] = None,
    ) -> None:
        # Resolve parameters
        _optimizer: Optional[Callable] = resolve_param(
            value=optimizer, config=config, attr="optimizer"
        )
        if _optimizer is None:
            logger.critical("No optimizer provided - interrupting execution.")
            raise ValueError("No optimizer provided - interrupting execution.")
        _optimizer_params: Optional[Mapping[str, Any]] = resolve_param(
            value=optimizer_params, config=config, attr="optimizer_params"
        )
        _interleaved_optim: Optional[bool] = resolve_param(
            value=interleaved_optim, config=config, attr="interleaved_optim"
        )
        _gradient_clipping: Optional[float] = resolve_param(
            value=gradient_clipping, config=config, attr="gradient_clipping"
        )
        if _gradient_clipping is not None and _gradient_clipping <= 0.0:
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
        _lr_scheduler: Optional[Callable] = resolve_param(
            value=lr_scheduler, config=config, attr="lr_scheduler"
        )

        if _gradient_accumulation is not None and _gradient_accumulation <= 0:
            raise ValueError("gradient_accumulation must be > 0")

        self.model: nn.Module | FullyShardedDataParallel = model
        self.optimizer_class: Callable = _optimizer
        self.optimizer_params: Mapping[str, Any] = (
            {} if _optimizer_params is None else _optimizer_params
        )
        self.interleaved_optim: bool = bool(_interleaved_optim)
        self.gradient_clipping: Optional[float] = _gradient_clipping
        self.gradient_accumulation: Optional[int] = _gradient_accumulation
        self.lr_scheduler_class: Optional[Callable] = _lr_scheduler
        self.total_steps_per_epoch: int = total_steps_per_epoch
        self.scaler: Optional[GradScaler] = scaler

        self.optimizer_step: int = 0
        self._step: int = 0

        if self.interleaved_optim and self.gradient_clipping is not None:
            logger.warning(
                "Both gradient clipping and backward/optimization overlap are enabled. This changes the bahviour of gradient clipping from model-wise to parameter-wise, breking mathematical coherency with the non-overlapped execution."
            )

        self._create_optimizer()

        if self.lr_scheduler_class is not None:
            self.lr_scheduler: LRScheduler | Mapping[nn.Parameter, LRScheduler] = (
                self.lr_scheduler_class(
                    optimizer=self.optimizer,
                    total_steps_per_epoch=self.total_steps_per_epoch,
                    config=config,
                )
            )

    def _create_optimizer(self) -> None:
        """Creates the specified optimizer configured for LLM pretraining."""

        self.optimizer: Optimizer | Mapping[nn.Parameter, Optimizer]
        if self.interleaved_optim:
            logger.info("Enabling optimizer/backward interleaving")

            self.optimizer = {
                p: self.optimizer_class([p], **self.optimizer_params)
                for p in self.model.parameters()
            }
            self._register_interleaving()
        else:
            self.optimizer = self.optimizer_class(
                self.model.parameters(), **self.optimizer_params
            )
            self.clip_fn: Callable = self._get_clip_fn()

    def _register_interleaving(self) -> None:
        """Register optimizer as a post-gradient hook for every parameter."""

        class optimizer_hook:
            def __init__(
                self,
                optimizer: Optimizer,
                gradient_clipping_fn: Callable,
                gradient_accumulation: Optional[int] = None,
                total_steps_per_epoch: int = -1,
                scaler: Optional[GradScaler] = None,
            ) -> None:
                """Per-parameter optimizer hook class providing stateful functionalities and callable behaviour.

                :param optimizer: Parameter's optimizer
                :type optimizer: Optimizer
                :param gradient_clipping_fn: Gradient clipping function
                :type gradient_clipping_fn: Callable
                :param gradient_accumulation: Gradient accumulation steps, defaults to None
                :type gradient_accumulation: Optional[int], optional
                :param total_steps_per_epoch: Number of training steps per epoch (before gradient accumulation), defaults to -1
                :type total_steps_per_epoch: int
                :param scaler: Gradient scaler, if necessary
                :type scaler: Optional[GradScaler]
                """

                self.optimizer: Optimizer = optimizer
                self.gradient_clipping_fn: Callable = gradient_clipping_fn
                self.gradient_accumulation: Optional[int] = gradient_accumulation
                self.scaler: Optional[GradScaler] = scaler

                self.step = 0
                self.total_steps_per_epoch = total_steps_per_epoch

            def __call__(self, parameter: nn.Parameter) -> None:
                """Set a per-parameter optimizer phase for FSDP.

                :param parameter: Current model parameter
                :type parameter: nn.Parameter
                """
                if _is_optimizer_accumulation_step_time(
                    gradient_accumulation=self.gradient_accumulation,
                    step=self.step,
                    total_steps_per_epoch=self.total_steps_per_epoch,
                ):
                    self.gradient_clipping_fn(parameter)

                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                self.step += 1

        assert isinstance(self.optimizer, Mapping)
        for p in self.model.parameters():
            p.register_post_accumulate_grad_hook(
                optimizer_hook(
                    optimizer=self.optimizer[p],
                    gradient_clipping_fn=self._get_clip_fn(),
                    gradient_accumulation=self.gradient_accumulation,
                    total_steps_per_epoch=self.total_steps_per_epoch,
                    scaler=self.scaler,
                )
            )

    def _get_clip_fn(self) -> Callable:
        """Get the right gradient clip function.

        :return: Clipping function
        :rtype: Callable
        """
        if self.gradient_clipping is not None:
            if self.interleaved_optim:
                if isinstance(self.model, FullyShardedDataParallel):
                    return lambda p: p.clip_grad_norm_(max_norm=self.gradient_clipping)
                else:
                    return lambda p: torch.nn.utils.clip_grad_norm_(
                        parameters=p, max_norm=self.gradient_clipping  # type: ignore
                    )
            else:
                if isinstance(self.model, FullyShardedDataParallel):
                    return lambda p: p.clip_grad_norm_(max_norm=self.gradient_clipping)
                else:
                    return lambda p: torch.nn.utils.clip_grad_norm_(
                        parameters=p.parameters(), max_norm=self.gradient_clipping  # type: ignore
                    )
        else:
            return lambda _: None

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Reset the XFFL optimizers gradients.

        :param set_to_none: Set the gradients to None instead of zero, defaults to True
        :type set_to_none: bool, optional
        """
        if self.interleaved_optim:
            assert isinstance(self.optimizer, Mapping)
            for _, opt in self.optimizer.items():
                opt.zero_grad(set_to_none=set_to_none)
        else:
            assert isinstance(self.optimizer, Optimizer)
            self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(
        self,
        closure: Optional[Callable] = None,
        set_to_none: bool = True,
    ) -> None:
        """Perform a single XFFL optimization step.

        :param closure:  A closure that reevaluates the model and
        returns the loss, defaults to None
        :type closure: Optional[Callable], optional
        :param set_to_none: Set the gradients to None instead of zero, defaults to True
        :type set_to_none: bool, optional
        """
        if _is_optimizer_accumulation_step_time(
            gradient_accumulation=self.gradient_accumulation,
            step=self._step,
            total_steps_per_epoch=self.total_steps_per_epoch,
        ):
            if not self.interleaved_optim:
                assert isinstance(self.optimizer, Optimizer)
                self.clip_fn(self.model)

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step(closure=closure)
                self.optimizer.zero_grad(set_to_none=set_to_none)

            if self.lr_scheduler is not None:
                if self.interleaved_optim:
                    assert isinstance(self.lr_scheduler, Mapping)
                    for _, scheduler in self.lr_scheduler.items():
                        scheduler.step()
                else:
                    assert isinstance(self.lr_scheduler, LRScheduler)
                    self.lr_scheduler.step()

            self.optimizer_step += 1
        self._step += 1
