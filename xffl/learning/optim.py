import math
from logging import Logger, getLogger
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from xffl.custom.config import XFFLConfig

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def warmup_cosine_decay(
    optimizer: Optimizer,
    total_steps_per_epoch: int,
    config: XFFLConfig,
) -> LRScheduler:
    """Creates a Warmup + Cosine Decay learning rate scheduler.

    The scheduler operates on **optimizer steps** and is compatible with gradient
    accumulation. Training steps are internally converted into optimizer steps.

    Learning rate schedule:
        1) Linear warmup from 0 → peak learning rate
        2) Cosine decay from peak learning rate → final learning rate

    :param optimizer: Optimizer whose learning rate will be updated
    :type optimizer: Optimizer
    :param total_steps_per_epoch: Number of training steps per epoch (before gradient accumulation)
    :type total_steps_per_epoch: int
    :param config: xFFL training configuration
    :type config: XFFLConfig
    :return: Configured warmup + cosine decay scheduler
    :rtype: LRScheduler
    """

    if config.learning_rate is None:
        raise ValueError("learning_rate must be provided in XFFLConfig")

    epochs: int = config.epochs or 1
    accum_steps: int = config.gradient_accumulation or 1

    if total_steps_per_epoch <= 0:
        raise ValueError("total_steps_per_epoch must be > 0")
    if accum_steps <= 0:
        raise ValueError("gradient_accumulation must be > 0")

    class WarmupCosineScheduler(LRScheduler):
        """Linear warmup followed by cosine decay scheduler.

        Must be stepped **after each optimizer.step()**.

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
            last_epoch: int = -1,
        ) -> None:

            self.peak_lr = peak_lr
            self.final_lr = peak_lr * final_lr_ratio

            # Convert training steps → optimizer steps
            effective_steps_per_epoch = steps_per_epoch // accum_steps
            self.total_steps = max(1, epochs * effective_steps_per_epoch)
            self.warmup_steps = max(1, int(self.total_steps * warmup_fraction))

            super().__init__(optimizer, last_epoch)

            logger.info(
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

    return WarmupCosineScheduler(
        optimizer=optimizer,
        epochs=epochs,
        accum_steps=accum_steps,
        peak_lr=config.learning_rate,
        steps_per_epoch=total_steps_per_epoch,
    )
