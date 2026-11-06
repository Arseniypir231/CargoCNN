"""Initial week-3 CNN training and artifact generation (no tuning/early stop)."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from src.cnn_model import CargoCNN
from src.torch_data import CLASSES, make_loaders


def train_model(dataset_root: str | Path, output_dir: str | Path, epochs: int = 10,
                batch_size: int = 32, image_size: int = 224, workers: int = 0,
                progress_callback: Callable[[dict], None] | None = None,
                device: str | None = None, run_kind: str = "dataset") -> dict:
    """Train from random initialization, validating each epoch on validation.

    The test loader is intentionally not iterated in this initial training run.
    ``progress_callback`` receives one dictionary after each completed epoch.
    """
    if not 1 <= int(epochs) <= 500:
        raise ValueError("Число эпох должно быть от 1 до 500")
    if int(batch_size) < 1:
        raise ValueError("Размер пакета должен быть положительным")
    root, out = Path(dataset_root), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    chosen_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loaders = make_loaders(root, batch_size=int(batch_size), workers=int(workers), image_size=int(image_size))

    model = CargoCNN(num_classes=len(CLASSES)).to(chosen_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = []
    started = time.perf_counter()

    for epoch in range(1, int(epochs) + 1):
        row = {"epoch": epoch}
        for phase in ("train", "valid"):
            training = phase == "train"
            model.train(training)
            total_loss, total_correct, total_items = 0.0, 0, 0
            context = torch.enable_grad if training else torch.no_grad
            with context():
                for images, labels in loaders[phase]:
                    images = images.to(chosen_device, non_blocking=True)
                    labels = labels.to(chosen_device, dtype=torch.long, non_blocking=True)
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                    logits = model(images)
                    loss = criterion(logits, labels)
                    if training:
                        loss.backward()
                        optimizer.step()
                    size = int(labels.size(0))
                    total_loss += float(loss.detach().item()) * size
                    total_correct += int((logits.argmax(dim=1) == labels).sum().item())
                    total_items += size
            if not total_items:
                raise ValueError(f"Выборка {phase} не содержит изображений")
            row[f"{phase}_loss"] = total_loss / total_items
            row[f"{phase}_accuracy"] = total_correct / total_items
            row[f"{phase}_images"] = total_items
        history.append(row)
        if progress_callback:
            progress_callback(dict(row))

    result = {
        "model": "CargoCNN from scratch", "run_kind": run_kind,
        "dataset_root": str(root.resolve()), "class_names": list(CLASSES),
        "architecture": {
            "input_channels": 3, "input_size": [int(image_size), int(image_size)],
            "convolutional_blocks": 4,
            "block_operations": ["Conv2d", "BatchNorm2d", "ReLU", "Conv2d", "BatchNorm2d", "ReLU", "MaxPool2d"],
            "fully_connected": [256, 128, 3], "output_classes": 3,
            "pretrained_weights": False, "dropout": False,
        },
        "loss": "CrossEntropyLoss", "optimizer": {"name": "AdamW", "learning_rate": 0.001},
        "epochs_requested": int(epochs), "epochs_completed": len(history),
        "batch_size": int(batch_size), "device": str(chosen_device),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "test_evaluation_performed": False, "history": history,
    }
    _save_results(model, optimizer, result, out)
    return result


def _save_results(model: nn.Module, optimizer: torch.optim.Optimizer,
                  result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "class_names": list(CLASSES),
        "image_size": result["architecture"]["input_size"],
        "epoch": result["epochs_completed"], "architecture": result["architecture"],
        "run_kind": result["run_kind"],
    }
    target = output_dir / "cargo_cnn_week3_initial.pth"
    temp_path = target.with_suffix(".pth.tmp")
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, target)
    with (output_dir / "cnn_training_history.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(result["history"][0].keys()))
        writer.writeheader()
        writer.writerows(result["history"])
    (output_dir / "cnn_training_history.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_history(result["history"], output_dir / "cnn_training_curves.png")


def _plot_history(history: list[dict], destination: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), dpi=150)
    axes[0].plot(epochs, [r["train_loss"] for r in history], marker="o", label="Train")
    axes[0].plot(epochs, [r["valid_loss"] for r in history], marker="o", label="Validation")
    axes[0].set(title="Loss по эпохам", xlabel="Эпоха", ylabel="CrossEntropyLoss")
    axes[1].plot(epochs, [r["train_accuracy"] for r in history], marker="o", label="Train")
    axes[1].plot(epochs, [r["valid_accuracy"] for r in history], marker="o", label="Validation")
    axes[1].set(title="Accuracy по эпохам", xlabel="Эпоха", ylabel="Доля правильных ответов", ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle("Первоначальное обучение CargoCNN (фактические значения запуска)", weight="bold")
    fig.tight_layout()
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def run_smoke_test(project_root: str | Path = ".") -> dict:
    """Train one epoch on generated temporary images to test code and artifacts.

    The generated images are only a software smoke test; metrics must not be
    interpreted as accuracy on the cargo dataset.
    """
    project_root = Path(project_root)
    smoke_out = project_root / "outputs" / "week3_smoke_test"
    temp_root = Path(tempfile.mkdtemp(prefix="cargo_cnn_smoke_"))
    try:
        from PIL import Image, ImageDraw
        colors = {"barrel": (165, 85, 40), "box": (70, 120, 190), "sack": (90, 155, 90)}
        for split in ("train", "valid", "test"):
            for cls, color in colors.items():
                folder = temp_root / split / cls
                folder.mkdir(parents=True, exist_ok=True)
                for i in range(2 if split == "train" else 1):
                    img = Image.new("RGB", (72, 68), color)
                    ImageDraw.Draw(img).rectangle((5 + i, 5, 40, 50), outline=(245, 245, 245), width=2)
                    img.save(folder / f"{cls}_{i}.png")
        result = train_model(temp_root, smoke_out, epochs=1, batch_size=3,
                             image_size=64, device="cpu", run_kind="synthetic_smoke_test")
        result["warning"] = "Синтетические изображения: метрики smoke-теста не характеризуют качество на реальном датасете."
        (smoke_out / "smoke_test_note.txt").write_text(result["warning"] + "\n", encoding="utf-8")
        (smoke_out / "cnn_training_history.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Папка object_crops или classification_images с train/valid/test")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--smoke-test", action="store_true", help="Проверка на временных синтетических изображениях")
    args = parser.parse_args()
    if args.smoke_test:
        result = run_smoke_test(Path(__file__).resolve().parents[1])
        print(json.dumps({"run_kind": result["run_kind"], "epochs_completed": result["epochs_completed"],
                          "artifacts": "outputs/week3_smoke_test"}, ensure_ascii=False, indent=2))
        return
    if not args.data:
        parser.error("укажите --data <папка_с_train_valid_test> либо --smoke-test")
    result = train_model(args.data, args.output, epochs=args.epochs, batch_size=args.batch_size,
                         progress_callback=lambda row: print(json.dumps(row, ensure_ascii=False), flush=True))
    print(json.dumps({k: result[k] for k in ("epochs_completed", "elapsed_seconds", "test_evaluation_performed")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
