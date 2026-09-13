import copy
import json
import math
from pathlib import Path
from typing import Optional
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from conspect.utils import AverageMeter, get_logger
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
    def __init__(self, epochs: int, train_loader: DataLoader, valid_loader: DataLoader,
                 criterion, optimizer, scaler, device, save_dir, model_name,
                 scheduler=None, early_stopping_patience: Optional[int] = None,
                 training_protocol: Optional[dict] = None,
                 experiment_metadata: Optional[dict] = None):
        self.epochs = epochs
        self.train_loader, self.valid_loader = train_loader, valid_loader
        self.criterion, self.optimizer = criterion, optimizer
        self.scaler = scaler
        self.device = device
        self.save_dir = save_dir
        self.model_name = model_name
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience
        self.training_protocol = training_protocol or {}
        self.experiment_metadata = experiment_metadata or {}
        timestamp = str(time.time())
        self.logger = get_logger(str(Path(self.save_dir).joinpath(f"{self.model_name}_{timestamp}_.log.txt")))
        self.best_loss = float("inf")
        self.best_epoch = None
        self.best_metrics = None
        self.no_improve_epochs = 0
        self.history = []
        self.lr_trajectory = []
        self.global_step = 0
        self.nonfinite_detected = False
        self.reload_consistency = None

    def _inverse_loss(self, out: torch.Tensor, y: torch.Tensor):
        y_true = self.scaler.inverse_transform(y.cpu().detach().numpy().reshape(-1, self.scaler.n_features_in_))
        y_pred = self.scaler.inverse_transform(out.cpu().detach().numpy().reshape(-1, self.scaler.n_features_in_))
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mask = y_true > 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else 0.0
        return rmse, mae, mape

    def fit(self, model: nn.Module) -> dict:
        self.logger.info("Trainer - epochs: %s, scheduler: %s, early_stopping_patience: %s",
                         self.epochs, type(self.scheduler).__name__ if self.scheduler else None,
                         self.early_stopping_patience)
        for epoch in range(self.epochs):
            model.train()
            train_loss = AverageMeter("train_loss")
            train_rmse = AverageMeter("train_rmse")
            epoch_lrs = []
            with tqdm(self.train_loader, dynamic_ncols=True) as pbar:
                pbar.set_description(f"[Epoch {epoch + 1}/{self.epochs}]")
                for tr_data in pbar:
                    tr_X = [d.to(self.device) for d in tr_data[:-1]]
                    tr_y = tr_data[-1].to(self.device)
                    self.optimizer.zero_grad()
                    out = model(*tr_X)
                    loss = self.criterion(out, tr_y)
                    if not torch.isfinite(out).all() or not torch.isfinite(loss):
                        self.nonfinite_detected = True
                        raise FloatingPointError(
                            f"NaN/Inf detected at epoch={epoch + 1}, step={self.global_step + 1}"
                        )
                    lr_before_step = float(self.optimizer.param_groups[0]["lr"])
                    loss.backward()
                    gradients = [p.grad for p in model.parameters() if p.grad is not None]
                    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                        self.nonfinite_detected = True
                        raise FloatingPointError(
                            f"missing or non-finite gradients at epoch={epoch + 1}, step={self.global_step + 1}"
                        )
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.global_step += 1
                    lr_after_step = float(self.optimizer.param_groups[0]["lr"])
                    epoch_lrs.append(lr_before_step)
                    self.lr_trajectory.append(
                        {
                            "global_step": self.global_step,
                            "epoch": epoch + 1,
                            "step_in_epoch": len(epoch_lrs),
                            "lr_used": lr_before_step,
                            "lr_after_scheduler": lr_after_step,
                        }
                    )
                    rmse, _, _ = self._inverse_loss(out, tr_y)
                    train_loss.update(loss.item(), tr_y.size(0))
                    train_rmse.update(rmse, tr_y.size(0))
                    pbar.set_postfix(loss=train_loss.avg, rmse=train_rmse.avg)
            metrics = self.evaluate(model, epoch)
            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": float(train_loss.avg),
                "train_rmse": float(train_rmse.avg),
                "validation_rmse": float(metrics["rmse"]),
                "validation_mae": float(metrics["mae"]),
                "validation_mape": float(metrics["mape"]),
                "is_best": bool(metrics["is_best"]),
                "lr_first": float(epoch_lrs[0]),
                "lr_min": float(min(epoch_lrs)),
                "lr_max": float(max(epoch_lrs)),
                "lr_last": float(epoch_lrs[-1]),
            }
            self.history.append(epoch_record)
            self.logger.info(
                "Epoch Summary - Epoch: %s, Train Loss: %.8f, Train RMSE: %.4f, "
                "Val RMSE: %.4f, Val MAE: %.4f, LR: [%.8g, %.8g]",
                epoch + 1,
                train_loss.avg,
                train_rmse.avg,
                metrics["rmse"],
                metrics["mae"],
                min(epoch_lrs),
                max(epoch_lrs),
            )
            self.no_improve_epochs = 0 if metrics["is_best"] else self.no_improve_epochs + 1
            if self.early_stopping_patience is not None and self.no_improve_epochs >= self.early_stopping_patience:
                self.logger.info("Early stopping at epoch %s after %s epochs without validation RMSE improvement.",
                                 epoch + 1, self.no_improve_epochs)
                break
        self._log_best_result()
        summary = self.summary()
        metrics_path = Path(self.save_dir).joinpath("metrics.json")
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        self.logger.info("Machine-readable metrics: %s", metrics_path)
        return summary

    def summary(self) -> dict:
        return {
            "best_epoch": self.best_epoch,
            "best_validation": self.best_metrics,
            "best_checkpoint": str(Path(self.save_dir).joinpath("best.pth").resolve()),
            "selection_metric": "validation_rmse",
            "test_evaluation_performed": False,
            "epochs_requested": self.epochs,
            "epochs_completed": len(self.history),
            "nonfinite_detected": self.nonfinite_detected,
            "history": self.history,
            "lr_trajectory": self.lr_trajectory,
            "reload_consistency": self.reload_consistency,
            "training_protocol": self.training_protocol,
            "experiment_metadata": self.experiment_metadata,
        }

    def _log_best_result(self) -> None:
        if self.best_metrics is None:
            self.logger.info("Best Validation - no validation result was recorded.")
            return
        self.logger.info("Best Validation - Epoch: {epoch}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}%".format(
            epoch=self.best_epoch, rmse=self.best_metrics["rmse"],
            mae=self.best_metrics["mae"], mape=self.best_metrics["mape"]))

    @torch.no_grad()
    def evaluate(self, model: nn.Module, epoch: Optional[int] = None) -> dict:
        model.eval()
        squared_error_sum = 0.0
        absolute_error_sum = 0.0
        element_count = 0
        absolute_percentage_error_sum = 0.0
        positive_count = 0
        reload_probe = None
        for va_data in tqdm(self.valid_loader):
            va_X = [d.to(self.device) for d in va_data[:-1]]
            va_y = va_data[-1].to(self.device)
            out = model(*va_X)
            if not torch.isfinite(out).all():
                self.nonfinite_detected = True
                raise FloatingPointError("NaN/Inf detected during validation")
            y_true = self.scaler.inverse_transform(
                va_y.cpu().numpy().reshape(-1, self.scaler.n_features_in_)
            )
            y_pred = self.scaler.inverse_transform(
                out.cpu().numpy().reshape(-1, self.scaler.n_features_in_)
            )
            difference = y_true - y_pred
            squared_error_sum += float(np.square(difference).sum(dtype=np.float64))
            absolute_error_sum += float(np.abs(difference).sum(dtype=np.float64))
            element_count += int(difference.size)
            positive_mask = y_true > 0
            if np.any(positive_mask):
                absolute_percentage_error_sum += float(
                    np.abs(difference[positive_mask] / y_true[positive_mask]).sum(
                        dtype=np.float64
                    )
                )
                positive_count += int(positive_mask.sum())
            if reload_probe is None:
                reload_probe = (
                    [tensor[: min(2, tensor.shape[0])].detach().clone() for tensor in va_X],
                    out[: min(2, out.shape[0])].detach().clone(),
                )
        if element_count == 0:
            raise RuntimeError("validation loader produced no elements")
        rmse = math.sqrt(squared_error_sum / element_count)
        mae = absolute_error_sum / element_count
        mape = (
            100.0 * absolute_percentage_error_sum / positive_count
            if positive_count
            else 0.0
        )
        self.logger.info(
            f"Final Validation - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}%"
        )
        is_best = False
        if epoch is not None and rmse <= self.best_loss:
            is_best = True
            self.best_loss = rmse
            self.best_epoch = epoch + 1
            self.best_metrics = {"rmse": rmse, "mae": mae, "mape": mape}
            checkpoint_path = Path(self.save_dir).joinpath("best.pth")
            checkpoint = {
                    "epoch": self.best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": (
                        self.scheduler.state_dict() if self.scheduler is not None else None
                    ),
                    "best_metrics": self.best_metrics,
                    "training_protocol": self.training_protocol,
                    "experiment_metadata": self.experiment_metadata,
            }
            torch.save(checkpoint, checkpoint_path)
        return {"rmse": rmse, "mae": mae, "mape": mape, "is_best": is_best}
