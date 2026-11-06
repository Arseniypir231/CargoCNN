
"""Подбор порога уверенности CNN на выборке Validation.

Модель не обучается. Test не используется для выбора порога.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluate_cnn import load_trained_model
from src.torch_data import CargoImageDataset, CLASSES


DATASET_ROOT = Path(
    r"D:\4kurs\7sem\ES\Dataset\F2"
    r"\Cargo_CNN_full_crops\prepared_cargo_cnn\object_crops"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

THRESHOLDS = (0.60, 0.70, 0.80, 0.85, 0.90, 0.95)
TARGET_ACCURACY = 0.98
BATCH_SIZE = 16


def collect_predictions():
    """Получить предсказания CNN на Validation."""

    validation_dir = DATASET_ROOT / "valid"

    if not validation_dir.is_dir():
        raise FileNotFoundError(
            f"Не найдена папка Validation: {validation_dir}"
        )

    model, device, checkpoint, image_size = load_trained_model()
    model.eval()

    dataset = CargoImageDataset(
        validation_dir,
        train=False,
        image_size=int(image_size),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    true_labels = []
    predicted_labels = []
    confidences = []

    print(f"Устройство: {device}")
    print(f"Изображений Validation: {len(dataset)}")
    print("Получаем предсказания без обучения...")

    processed = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)

            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)

            confidence, predictions = probabilities.max(dim=1)

            true_labels.extend(labels.numpy().tolist())
            predicted_labels.extend(predictions.cpu().numpy().tolist())
            confidences.extend(confidence.cpu().numpy().tolist())

            processed += len(labels)

            if processed % 500 < BATCH_SIZE or processed == len(dataset):
                print(f"Проверено: {processed}/{len(dataset)}")

    return (
        dataset,
        np.asarray(true_labels),
        np.asarray(predicted_labels),
        np.asarray(confidences),
        checkpoint,
    )


def calculate_thresholds(true_labels, predictions, confidences):
    """Сравнить пороги и число ошибок."""

    correct = true_labels == predictions
    total = len(correct)
    total_errors = int((~correct).sum())

    results = []

    for threshold in THRESHOLDS:
        accepted = confidences >= threshold
        manual = ~accepted

        accepted_count = int(accepted.sum())
        manual_count = int(manual.sum())

        accepted_errors = int((accepted & ~correct).sum())
        detected_errors = int((manual & ~correct).sum())

        accepted_accuracy = (
            float(correct[accepted].mean())
            if accepted_count > 0
            else None
        )

        detected_errors_percent = (
            100 * detected_errors / total_errors
            if total_errors > 0
            else 100.0
        )

        results.append({
            "threshold": threshold,
            "total_images": total,
            "accepted_images": accepted_count,
            "manual_review_images": manual_count,
            "manual_review_percent": 100 * manual_count / total,
            "accepted_accuracy": accepted_accuracy,
            "accepted_errors": accepted_errors,
            "detected_errors": detected_errors,
            "detected_errors_percent": detected_errors_percent,
        })

    return results, total_errors


def select_threshold(results):
    """Выбрать порог по заранее заданному правилу."""

    suitable = [
        row for row in results
        if row["accepted_accuracy"] is not None
        and row["accepted_accuracy"] >= TARGET_ACCURACY
    ]

    if suitable:
        # Среди порогов с точностью >= 98% выбираем тот,
        # который отправляет меньше всего фото на ручную проверку.
        selected = min(
            suitable,
            key=lambda row: (
                row["manual_review_images"],
                row["threshold"],
            ),
        )
        reason = (
            "Обеспечивает не менее 98% правильных ответов "
            "среди автоматически принятых изображений "
            "при минимальном числе ручных проверок."
        )
    else:
        # Если 98% недостижимы, честно фиксируем лучший вариант.
        selected = max(
            results,
            key=lambda row: (
                row["accepted_accuracy"]
                if row["accepted_accuracy"] is not None
                else -1,
                row["accepted_images"],
            ),
        )
        reason = (
            "Ни один проверенный порог не обеспечил целевые 98%. "
            "Выбран вариант с наибольшей точностью "
            "автоматически принятых изображений."
        )

    return selected, reason


def save_results(
    dataset, true_labels, predictions, confidences,
    results, total_errors, selected, reason, checkpoint
):
    """Сохранить CSV, JSON и график."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = (
        PROJECT_ROOT / "outputs" / "week5"
        / f"threshold_{timestamp}"
    )
    output.mkdir(parents=True, exist_ok=True)

    with (output / "threshold_comparison.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    with (output / "validation_predictions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "image", "true_class", "predicted_class",
            "confidence", "correct",
        ])

        for (path, _), true, pred, conf in zip(
            dataset.samples,
            true_labels,
            predictions,
            confidences,
        ):
            writer.writerow([
                str(path),
                CLASSES[int(true)],
                CLASSES[int(pred)],
                float(conf),
                bool(true == pred),
            ])

    summary = {
        "dataset": "Validation",
        "images": len(dataset),
        "total_errors": total_errors,
        "target_accepted_accuracy": TARGET_ACCURACY,
        "model_epoch": checkpoint.get("epoch"),
        "selected_threshold": selected["threshold"],
        "selection_reason": reason,
        "selected_metrics": selected,
        "all_results": results,
        "training_performed": False,
        "test_used_for_selection": False,
    }

    (output / "threshold_selection.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    x = [row["threshold"] * 100 for row in results]
    y_accuracy = [
        row["accepted_accuracy"] * 100
        if row["accepted_accuracy"] is not None else np.nan
        for row in results
    ]
    y_manual = [
        row["manual_review_percent"] for row in results
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(
        x, y_accuracy, marker="o",
        label="Точность автоматических ответов",
    )
    ax.plot(
        x, y_manual, marker="s",
        label="Доля ручных проверок",
    )
    ax.axhline(
        TARGET_ACCURACY * 100,
        linestyle="--",
        label="Целевая точность 98%",
    )
    ax.axvline(
        selected["threshold"] * 100,
        linestyle=":",
        label="Выбранный порог",
    )

    ax.set_xlabel("Порог уверенности, %")
    ax.set_ylabel("Значение, %")
    ax.set_title("Подбор порога уверенности CNN на Validation")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    fig.savefig(output / "threshold_comparison.png", dpi=160)
    plt.close(fig)

    return output


def main():
    dataset, true, pred, confidence, checkpoint = (
        collect_predictions()
    )

    results, total_errors = calculate_thresholds(
        true, pred, confidence
    )

    selected, reason = select_threshold(results)

    print("\nСРАВНЕНИЕ ПОРОГОВ НА VALIDATION")
    print(
        "Порог | Ручная проверка | Точность принятых "
        "| Выявлено ошибок"
    )

    for row in results:
        accuracy = row["accepted_accuracy"]
        accuracy_text = (
            f"{accuracy * 100:.2f}%"
            if accuracy is not None else "нет ответов"
        )

        print(
            f"{row['threshold']:.0%} | "
            f"{row['manual_review_images']} "
            f"({row['manual_review_percent']:.2f}%) | "
            f"{accuracy_text} | "
            f"{row['detected_errors']}/{total_errors}"
        )

    print(f"\nВЫБРАННЫЙ ПОРОГ: {selected['threshold']:.0%}")
    print(reason)

    output = save_results(
        dataset, true, pred, confidence,
        results, total_errors, selected, reason, checkpoint
    )

    print(f"\nРезультаты сохранены: {output}")
    print("Обучение не выполнялось. Test не использовался.")


if __name__ == "__main__":
    main()
