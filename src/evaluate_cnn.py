
"""Неделя 5. Проверка готовой CNN на тестовой выборке.

Модуль загружает модель четвёртой недели и проверяет её на Test.
Обучение модели не выполняется.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from src.cnn_model import CargoCNN
from src.torch_data import CargoImageDataset, CLASSES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEEK4_ROOT = PROJECT_ROOT / "outputs" / "week4"
WEEK5_ROOT = PROJECT_ROOT / "outputs" / "week5"

# Резервный порог, выбранный при сравнении на Validation.
FALLBACK_CONFIDENCE_THRESHOLD = 0.70


def load_selected_confidence_threshold(model_epoch=None):
    """Найти сохранённый выбор порога из Validation.

    Test не используется для выбора порога. При отсутствии файла
    применяется резервное значение из выполненного ранее анализа.
    """

    summaries = sorted(
        WEEK5_ROOT.glob("threshold_*/threshold_selection.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for summary_path in summaries:
        try:
            info = json.loads(summary_path.read_text(encoding="utf-8"))
            if info.get("dataset") != "Validation":
                continue
            if info.get("test_used_for_selection") is not False:
                continue
            if info.get("training_performed") is not False:
                continue
            if model_epoch is not None and info.get("model_epoch") != model_epoch:
                continue
            threshold = float(info["selected_threshold"])
            if 0.0 < threshold <= 1.0:
                return threshold, str(summary_path.resolve())
        except (OSError, ValueError, TypeError, KeyError):
            continue

    return FALLBACK_CONFIDENCE_THRESHOLD, None



def find_final_model():
    """Находит окончательную модель четвёртой недели."""

    pointer = WEEK4_ROOT / "latest_run.json"

    if pointer.is_file():
        data = json.loads(
            pointer.read_text(encoding="utf-8")
        )

        relative = Path(
            data["relative_path"].replace("\\", "/")
        )

        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                "В latest_run.json указан недопустимый путь."
            )

        model_path = (
            WEEK4_ROOT / relative / "final_model.pth"
        )

        if model_path.is_file():
            return model_path.resolve()

    # Резервный поиск, если указатель отсутствует.
    models = list(
        (WEEK4_ROOT / "runs").glob(
            "run_*/final_model.pth"
        )
    )

    if not models:
        raise FileNotFoundError(
            "Не найдена итоговая модель final_model.pth. "
            "Проверьте папку outputs/week4/runs."
        )

    return max(
        models,
        key=lambda path: path.stat().st_mtime
    ).resolve()


def load_trained_model(model_path=None):
    """Загружает архитектуру и обученные веса CNN."""

    if model_path is None:
        model_path = find_final_model()

    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Файл модели не найден: {model_path}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "Файл не содержит весов модели."
        )

    class_names = tuple(
        checkpoint.get("class_names", [])
    )

    if class_names != tuple(CLASSES):
        raise ValueError(
            "Порядок классов в модели не совпадает "
            "с порядком классов датасета."
        )

    architecture = checkpoint["architecture"]

    if architecture.get("name") != "CargoCNN":
        raise ValueError(
            "В файле указана другая архитектура CNN."
        )

    filters = tuple(architecture["filters"])
    dropout = float(architecture["dropout"])
    image_size = int(architecture["image_size"][0])

    model = CargoCNN(
        num_classes=3,
        filters=filters,
        dropout=dropout
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.to(device)
    model.eval()

    return model, device, checkpoint, image_size


def calculate_metrics(true_labels, predictions):
    """Вычисляет качество распознавания."""

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    precision = precision_score(
        true_labels,
        predictions,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predictions,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predictions,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0
    )

    report = classification_report(
        true_labels,
        predictions,
        labels=[0, 1, 2],
        target_names=list(CLASSES),
        output_dict=True,
        zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "classification_report": report,
    }


def save_confusion_matrix(matrix, output_path):
    """Создаёт график матрицы ошибок."""

    fig, ax = plt.subplots(
        figsize=(7, 6),
        dpi=150
    )

    image = ax.imshow(
        matrix,
        cmap="Blues"
    )

    ax.set_xticks(
        range(len(CLASSES)),
        CLASSES
    )

    ax.set_yticks(
        range(len(CLASSES)),
        CLASSES
    )

    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title("Матрица ошибок CNN на Test")

    for row in range(len(CLASSES)):
        for col in range(len(CLASSES)):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix[row, col] > matrix.max() / 2
                    else "black"
                )
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def compare_with_baseline(cnn_metrics, test_support, output_dir):
    """Сравнивает CNN с результатами HOG + SVM второй недели."""

    baseline_path = (
        PROJECT_ROOT / "outputs" / "hog_svm_metrics.json"
    )

    if not baseline_path.is_file():
        return {
            "status": "not_found",
            "message": "Результаты HOG + SVM не найдены."
        }

    baseline = json.loads(
        baseline_path.read_text(encoding="utf-8")
    )

    # Проверяем размер и состав тестовой выборки.
    if (
        baseline.get("test_images") != sum(test_support.values())
        or baseline.get("test_support") != test_support
    ):
        return {
            "status": "different_dataset",
            "message": (
                "Размер или состав тестовой выборки "
                "HOG + SVM отличается от CNN."
            )
        }

    metrics = [
        ("Accuracy", "accuracy"),
        ("Precision", "macro_precision"),
        ("Recall", "macro_recall"),
        ("Macro F1", "macro_f1"),
    ]

    cnn_values = [
        cnn_metrics[key]
        for _, key in metrics
    ]

    baseline_values = [
        baseline["test_" + key]
        for _, key in metrics
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(
        figsize=(9, 5),
        dpi=150
    )

    ax.bar(
        x - width / 2,
        cnn_values,
        width,
        label="CNN"
    )

    ax.bar(
        x + width / 2,
        baseline_values,
        width,
        label="HOG + SVM"
    )

    ax.set_xticks(
        x,
        [name for name, _ in metrics]
    )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Значение метрики")
    ax.set_title("Сравнение CNN и HOG + SVM на Test")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        output_dir / "cnn_vs_hog_svm.png",
        bbox_inches="tight"
    )
    plt.close(fig)

    return {
        "status": "completed",
        "cnn": {
            key: cnn_metrics[key]
            for _, key in metrics
        },
        "hog_svm": {
            key: baseline["test_" + key]
            for _, key in metrics
        }
    }


def evaluate_cnn(
    dataset_root,
    batch_size=16,
    model_path=None,
    progress_callback=None
):
    """Проверяет готовую модель на Test без обучения."""

    if batch_size < 1:
        raise ValueError(
            "Batch Size должен быть положительным."
        )

    dataset_root = Path(dataset_root)
    test_dir = dataset_root / "test"

    if not test_dir.is_dir():
        raise FileNotFoundError(
            f"Папка Test не найдена: {test_dir}"
        )

    model, device, checkpoint, image_size = (
        load_trained_model(model_path)
    )

    test_dataset = CargoImageDataset(
        test_dir,
        train=False,
        image_size=image_size
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    true_labels = []
    predictions = []
    confidences = []

    total_images = len(test_dataset)
    processed = 0

    model.eval()

    # В этом блоке нет обучения и обновления весов.
    with torch.inference_mode():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted = torch.max(
                probabilities,
                dim=1
            )

            true_labels.extend(labels.tolist())
            predictions.extend(
                predicted.cpu().tolist()
            )
            confidences.extend(
                confidence.cpu().tolist()
            )

            processed += len(labels)

            if progress_callback:
                progress_callback(
                    f"Проверено {processed} из {total_images}"
                )

    true_labels = np.asarray(true_labels)
    predictions = np.asarray(predictions)
    confidences = np.asarray(confidences)

    metrics = calculate_metrics(
        true_labels,
        predictions
    )

    matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=[0, 1, 2]
    )

    # Каждый запуск сохраняется в новую папку.
    run_name = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    output_dir = WEEK5_ROOT / (
        f"test_{run_name}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    save_confusion_matrix(
        matrix,
        output_dir / "cnn_confusion_matrix.png"
    )

    test_support = {
        name: int(np.sum(true_labels == index))
        for index, name in enumerate(CLASSES)
    }

    comparison = compare_with_baseline(
        metrics,
        test_support,
        output_dir
    )

    # Сохраняем отдельное предсказание для каждого фото.
    csv_path = output_dir / "predictions.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "image",
            "true_class",
            "predicted_class",
            "confidence",
            "correct"
        ])

        for index, (path, _) in enumerate(
            test_dataset.samples
        ):

            actual = int(true_labels[index])
            predicted = int(predictions[index])

            writer.writerow([
                str(path),
                CLASSES[actual],
                CLASSES[predicted],
                round(float(confidences[index]), 6),
                actual == predicted
            ])

    # Порог выбрали по Validation, а не по Test.
    threshold, threshold_source = load_selected_confidence_threshold(
        checkpoint.get("epoch")
    )
    accepted = confidences >= threshold
    correct = true_labels == predictions

    confidence_results = {
        "threshold": threshold,
        "threshold_source": threshold_source,
        "threshold_selection_dataset": "Validation",
        "accepted_images": int(accepted.sum()),
        "manual_review_images": int((~accepted).sum()),
        "accepted_fraction": float(accepted.mean()),
        "accepted_accuracy": (
            float(correct[accepted].mean())
            if accepted.any()
            else None
        ),
        "accepted_errors": int((accepted & ~correct).sum()),
        "manual_review_errors": int((~accepted & ~correct).sum()),
        "note": (
            f"Порог {threshold:.0%} "
            + (
                "загружен из результатов подбора на Validation. "
                if threshold_source
                else "используется как резервный результат ранее "
                     "выполненного подбора на Validation. "
            )
            + "Test не использовался для выбора порога. "
              "Уверенность softmax не гарантирует правильный ответ."
        ),
    }

    result = {
        "week": 5,
        "model": str(
            model_path or find_final_model()
        ),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "validation": checkpoint.get("validation"),
        "test_images": total_images,
        "test_support": test_support,
        "device": str(device),
        "metrics": metrics,
        "confusion_matrix": matrix.tolist(),
        "baseline_comparison": comparison,
        "confidence_analysis": confidence_results,
        "output_directory": str(output_dir.resolve()),
        "training_performed": False,
    }

    json_path = output_dir / "test_results.json"

    json_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Неделя 5. Тестирование готовой CNN."
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Путь к папке object_crops"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    args = parser.parse_args()

    result = evaluate_cnn(
        dataset_root=args.data,
        batch_size=args.batch_size,
        progress_callback=print
    )

    print("\nТестирование завершено.")
    print(
        "Accuracy:",
        round(result["metrics"]["accuracy"] * 100, 2),
        "%"
    )
    print(
        "Macro F1:",
        round(result["metrics"]["macro_f1"], 4)
    )
    print(
        "Результаты:",
        result["output_directory"]
    )


if __name__ == "__main__":
    main()