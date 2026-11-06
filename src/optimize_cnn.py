"""Week-4 sequential hyperparameter experiments, early stopping and selection."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from src.cnn_model import CargoCNN
from src.torch_data import CLASSES, make_loaders
from src.week4_utils import EarlyStopping, build_experiment_configs


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _save_history(history: list[dict], directory: Path, metadata: dict) -> None:
    if history:
        with (directory / "history.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
    _atomic_json(directory / "history.json", {**metadata, "history": history})


def _checkpoint(model: nn.Module, config: dict, epoch: int, valid_loss: float,
                valid_accuracy: float, image_size: int, seed: int) -> dict:
    return {
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "class_names": list(CLASSES),
        "architecture": {"name": "CargoCNN", "num_classes": 3, "input_channels": 3,
                         "image_size": [image_size, image_size], "convolutional_blocks": 4,
                         "filters": list(config["filters"]), "dropout": config["dropout"],
                         "fully_connected": [config["filters"][-1], 128, 3],
                         "pretrained_weights": False},
        "experiment": {**config, "seed": seed, "learning_rate": config["learning_rate"],
                       "effective_batch_size": config["batch_size"]},
        "epoch": epoch,
        "validation": {"loss": valid_loss, "accuracy": valid_accuracy},
    }


def _train_attempt(dataset_root: Path, attempt_dir: Path, config: dict, *, max_epochs: int,
                   patience: int, min_delta: float, seed: int, image_size: int,
                   progress_callback: Callable[[dict], None] | None,
                   experiment_index: int, experiment_total: int) -> dict:
    _seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = make_loaders(dataset_root, batch_size=config["batch_size"], workers=0, image_size=image_size)
    model = CargoCNN(filters=tuple(config["filters"]), dropout=config["dropout"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    stopper = EarlyStopping(patience, min_delta)
    history: list[dict] = []
    best_loss, best_accuracy, best_epoch = math.inf, 0.0, 0
    best_train_loss, best_train_accuracy = math.nan, math.nan
    best_checkpoint = None
    stop_reason = "max_epochs_reached"
    started = time.perf_counter()
    metadata = {"name": config["name"], "config": config, "max_epochs": max_epochs,
                "patience": patience, "min_delta": min_delta, "seed": seed,
                "device": str(device), "image_size": image_size}

    try:
        for epoch in range(1, max_epochs + 1):
            row = {"epoch": epoch}
            for phase in ("train", "valid"):
                training = phase == "train"
                model.train(training)
                sum_loss, correct, seen = 0.0, 0, 0
                context = torch.enable_grad if training else torch.no_grad
                with context():
                    for images, labels in loaders[phase]:
                        images = images.to(device, non_blocking=True)
                        labels = labels.to(device, dtype=torch.long, non_blocking=True)
                        if training:
                            optimizer.zero_grad(set_to_none=True)
                        logits = model(images)
                        loss = criterion(logits, labels)
                        if training:
                            loss.backward()
                            optimizer.step()
                        count = int(labels.size(0))
                        sum_loss += float(loss.detach().item()) * count
                        correct += int((logits.argmax(1) == labels).sum().item())
                        seen += count
                if not seen:
                    raise ValueError(f"Выборка {phase} не содержит изображений")
                row[f"{phase}_loss"] = sum_loss / seen
                row[f"{phase}_accuracy"] = correct / seen
                row[f"{phase}_images"] = seen

            history.append(row)
            if row["valid_loss"] < best_loss:
                best_loss = row["valid_loss"]
                best_accuracy = row["valid_accuracy"]
                best_epoch = epoch
                best_train_loss = row["train_loss"]
                best_train_accuracy = row["train_accuracy"]
                best_path = attempt_dir / "best_model.pth"
                temp_best_path = attempt_dir / "best_model.pth.tmp"
                best_checkpoint = _checkpoint(model, config, epoch, best_loss, best_accuracy, image_size, seed)
                torch.save(best_checkpoint, temp_best_path)
                os.replace(temp_best_path, best_path)

            improved, should_stop = stopper.update(row["valid_loss"])
            row["validation_loss_improved_by_min_delta"] = improved
            _save_history(history, attempt_dir, metadata)
            state = {**metadata, "status": "running", "best_epoch": best_epoch,
                     "best_validation_loss": best_loss, "best_validation_accuracy": best_accuracy,
                     "stop_reason": None, "epochs_completed": len(history)}
            _atomic_json(attempt_dir / "experiment.json", state)
            if progress_callback:
                progress_callback({"experiment_index": experiment_index, "experiment_total": experiment_total,
                                   "experiment": config["name"], **row, "best_epoch": best_epoch})
            if should_stop:
                stop_reason = "early_stopping_patience_exhausted"
                break

        result = {**metadata, "status": "completed", "epochs_completed": len(history),
                  "best_epoch": best_epoch, "best_train_loss": best_train_loss,
                  "best_train_accuracy": best_train_accuracy,
                  "best_validation_loss": best_loss, "best_validation_accuracy": best_accuracy,
                  "stop_reason": stop_reason, "elapsed_seconds": round(time.perf_counter() - started, 3),
                  "test_evaluation_performed": False, "best_model_path": str((attempt_dir / "best_model.pth").resolve())}
        best_checkpoint["training"] = {
            "loss": "CrossEntropyLoss", "optimizer": "AdamW",
            "max_epochs": max_epochs, "epochs_completed": len(history),
            "patience": patience, "min_delta": min_delta, "stop_reason": stop_reason,
            "seed": seed, "device": str(device), "elapsed_seconds": result["elapsed_seconds"],
        }
        temp_best_path = attempt_dir / "best_model.pth.tmp"
        torch.save(best_checkpoint, temp_best_path)
        os.replace(temp_best_path, attempt_dir / "best_model.pth")
        _atomic_json(attempt_dir / "experiment.json", result)
        _plot_history(history, attempt_dir / "training_curves.png", config["name"])
        return result
    except Exception as error:
        failure = {**metadata, "status": "failed", "epochs_completed": len(history),
                   "best_epoch": best_epoch or None, "best_validation_loss": best_loss if best_epoch else None,
                   "best_validation_accuracy": best_accuracy if best_epoch else None,
                   "best_train_loss": best_train_loss if best_epoch else None,
                   "best_train_accuracy": best_train_accuracy if best_epoch else None,
                   "stop_reason": "error",
                   "error_type": type(error).__name__, "error": str(error),
                   "elapsed_seconds": round(time.perf_counter() - started, 3)}
        _save_history(history, attempt_dir, metadata)
        _atomic_json(attempt_dir / "experiment.json", failure)
        if history:
            _plot_history(history, attempt_dir / "training_curves.png", config["name"] + " (partial)")
        raise
    finally:
        del model, optimizer, loaders
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _plot_history(history: list[dict], destination: Path, name: str) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=140)
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train")
    axes[0].plot(epochs, [row["valid_loss"] for row in history], marker="o", label="Validation")
    axes[0].set(title="Loss", xlabel="Эпоха", ylabel="CrossEntropyLoss")
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], marker="o", label="Train")
    axes[1].plot(epochs, [row["valid_accuracy"] for row in history], marker="o", label="Validation")
    axes[1].set(title="Accuracy", xlabel="Эпоха", ylabel="Доля правильных ответов", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=.25)
        axis.legend()
    fig.suptitle(f"Week 4 — {name}")
    fig.tight_layout()
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def _is_cuda_oom(error: Exception) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", ())
    return torch.cuda.is_available() and (
        (isinstance(oom_type, type) and isinstance(error, oom_type)) or "out of memory" in str(error).lower()
    )


def _write_summary(results: list[dict], session_dir: Path) -> None:
    _atomic_json(session_dir / "experiment_results.json", {"selection_metric": "lowest_best_validation_loss",
                                                           "results": results})
    columns = ["name", "status", "learning_rate", "requested_batch_size", "effective_batch_size",
               "batch_retries", "dropout", "filters", "max_epochs", "patience", "min_delta", "seed",
               "epochs_completed", "best_epoch", "best_train_loss", "best_train_accuracy",
               "best_validation_loss", "best_validation_accuracy", "stop_reason", "run_directory", "error"]
    with (session_dir / "experiment_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    successful = [r for r in results if r.get("status") == "completed"]
    if successful:
        ranked = sorted(successful, key=lambda r: (r["best_validation_loss"], -r["best_validation_accuracy"]))
        winner = ranked[0]
        _atomic_json(session_dir / "best_configuration.json", {
            "selection_metric": "lowest_best_validation_loss; tie-break: highest validation accuracy",
            "configuration": winner,
            "final_model_path": str((session_dir / "final_model.pth").resolve()),
        })
        shutil.copy2(winner["best_model_path"], session_dir / "final_model.pth")
        _plot_comparison(successful, session_dir / "comparison.png")
    else:
        _atomic_json(session_dir / "best_configuration.json", {"status": "no_successful_experiments"})


def _plot_comparison(results: list[dict], destination: Path) -> None:
    names = [r["name"] for r in results]
    x = list(range(len(names)))
    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(names) * 1.8), 4.5), dpi=150)
    axes[0].bar(x, [r["best_validation_loss"] for r in results], color="#4472C4")
    axes[0].set(title="Лучшая Validation Loss (меньше — лучше)", ylabel="Loss")
    axes[1].bar(x, [r["best_validation_accuracy"] for r in results], color="#70AD47")
    axes[1].set(title="Accuracy в эпоху лучшего Validation Loss", ylabel="Accuracy", ylim=(0, 1))
    for axis in axes:
        axis.set_xticks(x, names, rotation=30, ha="right")
        axis.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def run_experiments(dataset_root: str | Path, output_root: str | Path = "outputs/week4", *,
                    learning_rates=(0.001, 0.0005), batch_sizes=(8, 16),
                    dropouts=(0.0, 0.3), filter_sets=((32, 64, 128, 256), (16, 32, 64, 128)),
                    max_epochs: int = 20, patience: int = 4, min_delta: float = 0.001,
                    seed: int = 42, auto_reduce_batch: bool = True, image_size: int = 224,
                    progress_callback: Callable[[dict], None] | None = None,
                    status_callback: Callable[[dict], None] | None = None) -> dict:
    if max_epochs < 1 or patience < 1 or min_delta < 0:
        raise ValueError("max_epochs/patience должны быть положительными, min_delta — неотрицательным")
    configs = build_experiment_configs(learning_rates, batch_sizes, dropouts, filter_sets)
    data = Path(dataset_root)
    if not data.is_dir():
        raise FileNotFoundError(f"Каталог датасета не найден: {data}")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("run_%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    session_dir = root / "runs" / run_id
    session_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(root / "latest_run.json", {"run_id": run_id, "relative_path": str(Path("runs") / run_id)})
    results = []
    total = len(configs)

    for index, config in enumerate(configs, 1):
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", config["name"]).strip("_") or f"experiment_{index}"
        exp_dir = session_dir / f"{index:02d}_{slug}"
        exp_dir.mkdir(parents=True, exist_ok=False)
        requested_batch = int(config["batch_size"])
        _atomic_json(exp_dir / "configuration.json", {
            "config": config, "max_epochs": max_epochs, "patience": patience,
            "min_delta": min_delta, "seed": seed, "image_size": image_size,
            "auto_reduce_batch": auto_reduce_batch,
        })
        actual_batch = requested_batch
        retries = 0
        if status_callback:
            status_callback({"experiment_index": index, "experiment_total": total,
                             "experiment": config["name"], "status": "starting"})
        while True:
            attempt_dir = exp_dir / f"attempt_{retries + 1:02d}"
            attempt_dir.mkdir(parents=True, exist_ok=False)
            actual_config = {**config, "batch_size": actual_batch,
                             "requested_batch_size": requested_batch, "batch_retries": retries}
            try:
                result = _train_attempt(data, attempt_dir, actual_config, max_epochs=max_epochs,
                                        patience=patience, min_delta=min_delta, seed=seed,
                                        image_size=image_size, progress_callback=progress_callback,
                                        experiment_index=index, experiment_total=total)
                result.update({"requested_batch_size": requested_batch, "effective_batch_size": actual_batch,
                               "batch_retries": retries, "run_directory": str(exp_dir.resolve())})
                results.append(result)
                break
            except Exception as error:
                if _is_cuda_oom(error) and auto_reduce_batch and actual_batch > 1:
                    retries += 1
                    next_batch = max(1, actual_batch // 2)
                    if status_callback:
                        status_callback({"experiment_index": index, "experiment_total": total,
                                         "experiment": config["name"], "status": "cuda_oom_retry",
                                         "batch_size": next_batch, "error": str(error)})
                    actual_batch = next_batch
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                attempt_result = {}
                state_path = attempt_dir / "experiment.json"
                if state_path.is_file():
                    try:
                        attempt_result = json.loads(state_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        attempt_result = {}
                else:
                    attempt_result = {"status": "failed", "epochs_completed": 0,
                                      "error_type": type(error).__name__, "error": str(error)}
                    _atomic_json(state_path, attempt_result)
                failed = {"name": config["name"], "status": "failed", "learning_rate": config["learning_rate"],
                          "requested_batch_size": requested_batch, "effective_batch_size": actual_batch,
                          "batch_retries": retries, "dropout": config["dropout"], "filters": config["filters"],
                          "max_epochs": max_epochs, "patience": patience, "min_delta": min_delta, "seed": seed,
                          "epochs_completed": attempt_result.get("epochs_completed", 0),
                          "best_epoch": attempt_result.get("best_epoch"), "stop_reason": "error",
                          "best_train_loss": attempt_result.get("best_train_loss"),
                          "best_train_accuracy": attempt_result.get("best_train_accuracy"),
                          "best_validation_loss": attempt_result.get("best_validation_loss"),
                          "best_validation_accuracy": attempt_result.get("best_validation_accuracy"),
                          "run_directory": str(exp_dir.resolve()),
                          "error": f"{type(error).__name__}: {error}"}
                results.append(failed)
                break
        _write_summary(results, session_dir)
        if status_callback:
            status_callback({"experiment_index": index, "experiment_total": total,
                             "experiment": config["name"], "status": results[-1]["status"],
                             "result": results[-1]})

    summary = {"run_id": run_id, "run_directory": str(session_dir.resolve()),
               "experiments_requested": total,
               "experiments_completed": sum(r["status"] == "completed" for r in results),
               "experiments_failed": sum(r["status"] != "completed" for r in results),
               "selection_metric": "lowest_best_validation_loss; tie-break: highest validation accuracy",
               "results": results}
    _atomic_json(session_dir / "session_summary.json", summary)
    if status_callback:
        status_callback({"status": "finished", "result": summary})
    return summary


def run_smoke_test(project_root: str | Path = ".") -> dict:
    """Small synthetic technical run; never a quality result for cargo images."""
    from PIL import Image, ImageDraw
    base = Path(project_root)
    tmp = Path(tempfile.mkdtemp(prefix="cargo_week4_smoke_"))
    try:
        colors = {"barrel": (150, 70, 30), "box": (30, 90, 170), "sack": (65, 140, 65)}
        for split in ("train", "valid", "test"):
            for name, color in colors.items():
                folder = tmp / split / name
                folder.mkdir(parents=True, exist_ok=True)
                for i in range(2 if split == "train" else 1):
                    image = Image.new("RGB", (72, 72), color)
                    ImageDraw.Draw(image).rectangle((8 + i, 8, 60, 60), outline="white", width=2)
                    image.save(folder / f"sample_{i}.png")
        configs = build_experiment_configs((0.001,), (2,), (0.0, 0.25), ((4, 8, 16, 32),))
        result = run_experiments(tmp, base / "outputs" / "week4_smoke_test",
                                 learning_rates=(0.001,), batch_sizes=(2,), dropouts=(0.0, 0.25),
                                 filter_sets=((4, 8, 16, 32),), max_epochs=2, patience=1,
                                 min_delta=0.0, seed=42, image_size=64)
        assert len(configs) == 2
        assert result["experiments_completed"] == 2, result
        first = result["results"][0]
        session = Path(result["run_directory"])
        for artifact in ("experiment_results.csv", "experiment_results.json", "comparison.png",
                         "best_configuration.json", "final_model.pth"):
            assert (session / artifact).is_file(), artifact
        weights = Path(first["best_model_path"])
        assert weights.is_file()
        assert (weights.parent / "history.csv").is_file()
        assert (weights.parent / "history.json").is_file()
        assert (weights.parent / "training_curves.png").is_file()
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        model = CargoCNN(filters=tuple(checkpoint["architecture"]["filters"]),
                         dropout=checkpoint["architecture"]["dropout"])
        model.load_state_dict(checkpoint["model_state_dict"])
        torch.manual_seed(42)
        initial_model = CargoCNN(filters=(4, 8, 16, 32), dropout=0.0)
        assert not torch.equal(initial_model.features[0][0].weight,
                               model.features[0][0].weight), "Optimizer did not change the tested weights"
        with torch.no_grad():
            assert tuple(model(torch.zeros(1, 3, 64, 64)).shape) == (1, 3)
        es = EarlyStopping(patience=2, min_delta=0.0)
        assert es.update(1.0) == (True, False)
        assert es.update(1.1) == (False, False)
        assert es.update(1.2) == (False, True)
        note = {"type": "synthetic_smoke_test", "synthetic_data_only": True,
                "experiments_completed": result["experiments_completed"],
                "run_directory": result["run_directory"],
                "checks": ["config variants", "forward/backward/optimizer step", "early stopping logic",
                           "weights changed", "best checkpoint reload", "CSV/JSON/PNG artifacts"]}
        (session / "SMOKE_TEST_NOTE.json").write_text(
            json.dumps(note, ensure_ascii=False, indent=2), encoding="utf-8")
        return note
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Каталог object_crops с train/valid/test")
    parser.add_argument("--output", default="outputs/week4")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        print(json.dumps(run_smoke_test(Path(__file__).resolve().parents[1]), ensure_ascii=False, indent=2))
        return
    if not args.data:
        parser.error("укажите --data <object_crops> или --smoke-test")
    result = run_experiments(args.data, args.output, max_epochs=args.max_epochs,
                             patience=args.patience, min_delta=args.min_delta)
    print(json.dumps({k: result[k] for k in ("run_id", "experiments_completed", "experiments_failed", "run_directory")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
