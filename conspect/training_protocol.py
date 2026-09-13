"""Canonical P3 training components for formal experiments."""

from dataclasses import asdict, dataclass

import torch.nn as nn
import torch.optim as optim


@dataclass(frozen=True)
class TrainingProtocol:
    """Single source of truth for the final loss, optimizer, and scheduler."""

    smooth_l1_beta: float = 1.0
    learning_rate: float = 1e-3
    weight_decay: float = 0.05
    max_lr: float = 2e-3
    pct_start: float = 0.10

    def criterion(self) -> nn.SmoothL1Loss:
        return nn.SmoothL1Loss(beta=self.smooth_l1_beta)

    def optimizer(self, model: nn.Module) -> optim.AdamW:
        return optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def scheduler(
        self, optimizer: optim.Optimizer, epochs: int, steps_per_epoch: int
    ) -> optim.lr_scheduler.OneCycleLR:
        if epochs <= 0 or steps_per_epoch <= 0:
            raise ValueError("epochs and steps_per_epoch must be positive")
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=self.pct_start,
        )

    def as_dict(self) -> dict[str, float]:
        return asdict(self)
