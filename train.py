import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

current_dir = Path(__file__).resolve().parent
sys.path.append(str(Path.joinpath(current_dir, "../")))

from conspect.trainer import Trainer
from conspect.dataset.makedataset import get_dataset
from conspect.model import ConSpect
from conspect.training_protocol import TrainingProtocol
from conspect.utils import Cfg
from torch.utils.data import DataLoader


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "-s", "--seed", default=0, type=int, help="seed for initializing training"
    )
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="device to use, e.g. cuda:0 or cpu. Defaults to cuda:0 when CUDA is available, otherwise cpu",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        type=str,
        help="directory for logs and checkpoints. Defaults to outputs/conspect/<timestamp>",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run one train batch and one valid batch shape/loss check, then exit",
    )
    parser.add_argument(
        "--epochs",
        default=None,
        type=int,
        help="override only the epoch count for a smoke run; all other protocol settings remain unchanged",
    )
    parser.add_argument(
        "--valid-batch-size",
        default=None,
        type=int,
        help="validation batch size; metrics are aggregated globally across batches",
    )
    parser.add_argument("--early-stopping-patience", default=30, type=int, help="disable with <=0")
    return parser.parse_args()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device_arg: str | None) -> str:
    if device_arg:
        return device_arg
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def resolve_save_dir(save_dir_arg: str | None) -> str:
    if save_dir_arg:
        return str(Path(save_dir_arg).expanduser().resolve())
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return str((current_dir / "outputs" / "conspect" / timestamp).resolve())


def print_environment(device: str) -> None:
    print("=== Environment ===")
    print(f"sys.executable: {sys.executable}")
    print(f"torch.__version__: {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    print(f"selected device: {device}")


def describe_batch(batch, device: str, model: torch.nn.Module, criterion) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = [d.to(device) for d in batch]
    inputs = tensors[:-1]
    target = tensors[-1]

    names = ["x_c", "x_p", "x_t", "ext"]
    print("=== Batch Check ===")
    for idx, tensor in enumerate(inputs):
        name = names[idx] if idx < len(names) else f"input_{idx}"
        print(f"{name}.shape: {tuple(tensor.shape)}")
    print(f"y.shape: {tuple(target.shape)}")

    model.eval()
    with torch.no_grad():
        output = model(*inputs)
        loss = criterion(output, target)

    print(f"output.shape: {tuple(output.shape)}")
    print(f"loss.item(): {loss.item():.8f}")
    return output, loss


def main():
    args = make_parser()
    set_seed(args.seed)

    args.device = resolve_device(args.device)
    save_dir = resolve_save_dir(args.save_dir)

    print_environment(args.device)

    config_file = args.config_file

    config = Cfg(config_file)
    dataset_params = config.get_params(type="dataset")
    model_params = config.get_params(type="model")
    learning_params = config.get_params(type="learning")
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive")
        learning_params["epochs"] = args.epochs
    train_dataset, valid_dataset, scaler, external_dim = get_dataset(
        data_files=dataset_params["data_files"],
        holiday_file=dataset_params["holiday_file"],
        meteorol_file=dataset_params["meteorol_file"],
        T=dataset_params["T"],
        len_closeness=dataset_params["len_closeness"],
        len_period=dataset_params["len_period"],
        len_trend=dataset_params["len_trend"],
        len_test=dataset_params["len_test"],
        use_meta=dataset_params["use_meta"],
        map_height=model_params["map_height"],
        map_width=model_params["map_width"],
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=learning_params["batch_size"],
        shuffle=False,
        drop_last=False,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.valid_batch_size or dataset_params["len_test"],
        shuffle=False,
        drop_last=False,
    )

    model = ConSpect(
        len_closeness=dataset_params["len_closeness"],
        len_period=dataset_params["len_period"],
        len_trend=dataset_params["len_trend"],
        external_dim=external_dim,
        nb_flow=model_params["nb_flow"],
        map_height=model_params["map_height"],
        map_width=model_params["map_width"],
        nb_residual_unit=model_params["nb_residual_unit"],
    )
    model.to(args.device)

    protocol = TrainingProtocol(
        smooth_l1_beta=learning_params["smooth_l1_beta"],
        learning_rate=learning_params["learning_rate"],
        weight_decay=learning_params["weight_decay"],
        max_lr=learning_params["max_lr"],
        pct_start=learning_params["pct_start"],
    )
    criterion = protocol.criterion()
    optimizer = protocol.optimizer(model)
    scheduler = protocol.scheduler(
        optimizer,
        epochs=learning_params["epochs"],
        steps_per_epoch=len(train_dataloader),
    )
    early_stopping_patience = args.early_stopping_patience
    if early_stopping_patience <= 0:
        early_stopping_patience = None

    print(
        "SEED: {}, DEVICE: {}, SAVE_DIR: {}, residual_units: {}, "
        "C/P/T: {}/{}/{}, protocol: {}, early_stop: {}".format(
            args.seed,
            args.device,
            save_dir,
            model_params["nb_residual_unit"],
            dataset_params["len_closeness"],
            dataset_params["len_period"],
            dataset_params["len_trend"],
            protocol.as_dict(),
            early_stopping_patience,
        )
    )

    train_batch = next(iter(train_dataloader))
    describe_batch(train_batch, args.device, model, criterion)

    if args.smoke_test:
        valid_batch = next(iter(valid_dataloader))
        print("=== Valid Batch Smoke Check ===")
        describe_batch(valid_batch, args.device, model, criterion)
        print("Smoke test completed. No training was run and no checkpoint was saved.")
        return

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        epochs=learning_params["epochs"],
        train_loader=train_dataloader,
        valid_loader=valid_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        device=args.device,
        save_dir=save_dir,
        model_name="ConSpect",
        scheduler=scheduler,
        early_stopping_patience=early_stopping_patience,
        training_protocol=protocol.as_dict(),
        experiment_metadata={
            "seed": args.seed,
            "config_file": str(Path(config_file).resolve()),
            "dataset": dataset_params,
            "model": model_params,
            "learning": learning_params,
        },
    )
    summary = trainer.fit(model)
    print(
        "TRAINING_COMPLETE best_epoch={} best_val_rmse={:.8f} "
        "best_val_mae={:.8f} checkpoint={}".format(
            summary["best_epoch"],
            summary["best_validation"]["rmse"],
            summary["best_validation"]["mae"],
            summary["best_checkpoint"],
        )
    )


if __name__ == "__main__":
    main()
