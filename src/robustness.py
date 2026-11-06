
"""Неделя 5. Проверка устойчивости CNN и отдельных фотографий.

Используется готовая модель четвёртой недели.
Повторное обучение не выполняется.
Исходные изображения не изменяются.
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

from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from src.evaluate_cnn import load_trained_model
from src.torch_data import (
    CargoImageDataset,
    CLASSES,
    _to_tensor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEEK5_ROOT = PROJECT_ROOT / "outputs" / "week5"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

EFFECTS = {
    "clean": "Без изменений",
    "blur": "Размытие",
    "dark": "Затемнение",
    "bright": "Осветление",
    "rotate": "Поворот",
}


class ModifiedCargoDataset(Dataset):
    """Создаёт изменённые изображения только в памяти."""

    def __init__(
        self,
        samples,
        image_size,
        effect="clean"
    ):
        self.samples = list(samples)
        self.image_size = int(image_size)
        self.effect = effect

        if effect not in EFFECTS:
            raise ValueError(
                f"Неизвестный вид изменения: {effect}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR
        )

        if self.effect == "blur":

            image = image.filter(
                ImageFilter.GaussianBlur(radius=1.5)
            )

        elif self.effect == "dark":

            image = ImageEnhance.Brightness(
                image
            ).enhance(0.65)

        elif self.effect == "bright":

            image = ImageEnhance.Brightness(
                image
            ).enhance(1.35)

        elif self.effect == "rotate":

            image = image.rotate(
                12,
                resample=Image.Resampling.BILINEAR,
                fillcolor=(120, 120, 120)
            )

        return _to_tensor(image), label


def select_test_samples(
    test_dataset,
    max_images=600,
    seed=42
):
    """Выбирает одни и те же изображения для всех проверок."""

    if max_images < 3:
        raise ValueError(
            "Для проверки требуется минимум 3 изображения."
        )

    rng = np.random.default_rng(seed)

    labels = np.asarray([
        label
        for _, label in test_dataset.samples
    ])

    selected = []

    max_images = min(
        max_images,
        len(test_dataset)
    )

    # По возможности берём одинаковое число
    # изображений каждого из трёх классов.
    for class_index in range(len(CLASSES)):

        available = np.flatnonzero(
            labels == class_index
        )

        amount = min(
            len(available),
            max_images // len(CLASSES)
        )

        chosen = rng.choice(
            available,
            size=amount,
            replace=False
        )

        selected.extend(chosen.tolist())

    # Если осталось место, добавляем другие изображения.
    remaining = max_images - len(selected)

    if remaining > 0:

        other_indices = np.setdiff1d(
            np.arange(len(test_dataset)),
            np.asarray(selected),
            assume_unique=False
        )

        additional = rng.choice(
            other_indices,
            size=min(remaining, len(other_indices)),
            replace=False
        )

        selected.extend(additional.tolist())

    selected.sort()

    return [
        test_dataset.samples[index]
        for index in selected
    ]


def predict_dataset(
    model,
    dataset,
    device,
    batch_size=16,
    progress_callback=None,
    description="Проверка"
):
    """Получает ответы готовой модели без обучения."""

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    true_labels = []
    predictions = []

    model.eval()

    with torch.inference_mode():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            predicted = outputs.argmax(
                dim=1
            )

            true_labels.extend(
                labels.tolist()
            )

            predictions.extend(
                predicted.cpu().tolist()
            )

            if progress_callback:

                progress_callback(
                    f"{description}: "
                    f"{len(true_labels)}/{len(dataset)}"
                )

    return (
        np.asarray(true_labels),
        np.asarray(predictions)
    )


def save_robustness_plot(rows, output_path):
    """Сохраняет график результатов проверки устойчивости."""

    descriptions = [
        row["description"]
        for row in rows
    ]

    accuracies = [
        row["accuracy"] * 100
        for row in rows
    ]

    fig, ax = plt.subplots(
        figsize=(10, 5),
        dpi=150
    )

    bars = ax.bar(
        descriptions,
        accuracies
    )

    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy, %")
    ax.set_title(
        "Устойчивость CNN к изменениям изображений"
    )

    ax.tick_params(
        axis="x",
        rotation=15
    )

    for bar, value in zip(bars, accuracies):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.1f}%",
            ha="center"
        )

    fig.tight_layout()

    fig.savefig(
        output_path,
        bbox_inches="tight"
    )

    plt.close(fig)


def check_robustness(
    model,
    device,
    test_dataset,
    image_size,
    output_dir,
    max_images=600,
    batch_size=16,
    progress_callback=None
):
    """Проверяет качество на обычных и изменённых изображениях."""

    selected_samples = select_test_samples(
        test_dataset,
        max_images=max_images,
        seed=42
    )

    # Сохраняем список фотографий, выбранных для проверки.
    manifest_path = output_dir / "robustness_samples.csv"

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "image",
            "class"
        ])

        for path, label in selected_samples:

            writer.writerow([
                str(path),
                CLASSES[label]
            ])

    results = []

    for effect, description in EFFECTS.items():

        dataset = ModifiedCargoDataset(
            selected_samples,
            image_size=image_size,
            effect=effect
        )

        true_labels, predictions = predict_dataset(
            model=model,
            dataset=dataset,
            device=device,
            batch_size=batch_size,
            progress_callback=progress_callback,
            description=description
        )

        accuracy = accuracy_score(
            true_labels,
            predictions
        )

        macro_f1 = f1_score(
            true_labels,
            predictions,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0
        )

        results.append({
            "effect": effect,
            "description": description,
            "images": len(dataset),
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
        })

    clean_accuracy = results[0]["accuracy"]

    for row in results:

        row["accuracy_change_pp"] = (
            row["accuracy"] - clean_accuracy
        ) * 100

    csv_path = output_dir / "robustness_results.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)

    save_robustness_plot(
        results,
        output_dir / "robustness_plot.png"
    )

    return {
        "images": len(selected_samples),
        "seed": 42,
        "results": results,
        "note": (
            "Все изменения проверены на одних и тех же "
            "изображениях Test. Исходные файлы не изменялись."
        )
    }


def check_independent_photos(
    model,
    device,
    image_size,
    photo_folder,
    output_dir,
    batch_size=16,
    progress_callback=None
):
    """Проверяет новые фотографии, не использованные при обучении."""

    if not photo_folder:

        return {
            "status": "not_provided",
            "message": (
                "Папка с независимыми фотографиями "
                "не была выбрана."
            )
        }

    photo_folder = Path(photo_folder)

    if not photo_folder.is_dir():

        raise FileNotFoundError(
            f"Папка фотографий не найдена: {photo_folder}"
        )

    image_paths = sorted(
        path
        for path in photo_folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:

        return {
            "status": "empty",
            "images": 0
        }

    predictions = []

    model.eval()

    with torch.inference_mode():

        for start in range(
            0,
            len(image_paths),
            batch_size
        ):

            paths = image_paths[
                start:start + batch_size
            ]

            tensors = []

            for path in paths:

                with Image.open(path) as source:

                    image = source.convert("RGB")

                image = image.resize(
                    (image_size, image_size),
                    Image.Resampling.BILINEAR
                )

                tensors.append(
                    _to_tensor(image)
                )

            batch = torch.stack(
                tensors
            ).to(device)

            outputs = model(batch)

            probabilities = torch.softmax(
                outputs,
                dim=1
            ).cpu().numpy()

            for path, probability in zip(
                paths,
                probabilities
            ):

                predicted_index = int(
                    np.argmax(probability)
                )

                predicted_class = CLASSES[
                    predicted_index
                ]

                # Метка известна, если фотографии
                # разложены по папкам barrel/box/sack.
                relative = path.relative_to(photo_folder)

                expected_class = ""

                if (
                    len(relative.parts) > 1
                    and relative.parts[0] in CLASSES
                ):
                    expected_class = relative.parts[0]

                correct = ""

                if expected_class:
                    correct = (
                        predicted_class == expected_class
                    )

                predictions.append({
                    "image": str(path),
                    "expected_class": expected_class,
                    "predicted_class": predicted_class,
                    "confidence": float(
                        np.max(probability)
                    ),
                    "correct": correct
                })

            if progress_callback:

                progress_callback(
                    "Отдельные фотографии: "
                    f"{min(start + len(paths), len(image_paths))}"
                    f"/{len(image_paths)}"
                )

    csv_path = (
        output_dir / "independent_predictions.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(predictions[0].keys())
        )

        writer.writeheader()
        writer.writerows(predictions)

    labeled = [
        row
        for row in predictions
        if row["expected_class"]
    ]

    result = {
        "status": "completed",
        "images": len(predictions),
        "labeled_images": len(labeled),
        "predictions_file": str(csv_path),
    }

    if labeled:

        result["accuracy_labeled"] = (
            sum(
                int(row["correct"])
                for row in labeled
            ) / len(labeled)
        )

    else:

        result["accuracy_labeled"] = None

    return result


def run_additional_checks(
    dataset_root,
    independent_folder=None,
    batch_size=16,
    robustness_images=600,
    model_path=None,
    output_dir=None,
    progress_callback=None
):
    """Запускает дополнительные проверки пятой недели."""

    if batch_size < 1:
        raise ValueError(
            "Batch Size должен быть положительным."
        )

    if robustness_images < 3:
        raise ValueError(
            "Укажите не менее трёх тестовых изображений."
        )

    model, device, checkpoint, image_size = (
        load_trained_model(model_path)
    )

    dataset_root = Path(dataset_root)

    test_dataset = CargoImageDataset(
        dataset_root / "test",
        train=False,
        image_size=image_size
    )

    if output_dir is None:

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        output_dir = (
            WEEK5_ROOT / f"additional_{timestamp}"
        )

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    robustness_result = check_robustness(
        model=model,
        device=device,
        test_dataset=test_dataset,
        image_size=image_size,
        output_dir=output_dir,
        max_images=robustness_images,
        batch_size=batch_size,
        progress_callback=progress_callback
    )

    independent_result = check_independent_photos(
        model=model,
        device=device,
        image_size=image_size,
        photo_folder=independent_folder,
        output_dir=output_dir,
        batch_size=batch_size,
        progress_callback=progress_callback
    )

    result = {
        "week": 5,
        "model_epoch": checkpoint.get("epoch"),
        "device": str(device),
        "training_performed": False,
        "robustness": robustness_result,
        "independent_photos": independent_result,
        "output_directory": str(
            output_dir.resolve()
        )
    }

    result_path = (
        output_dir / "additional_results.json"
    )

    result_path.write_text(
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
        description=(
            "Неделя 5. Проверка устойчивости "
            "и отдельных фотографий."
        )
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Путь к папке object_crops"
    )

    parser.add_argument(
        "--photos",
        default=None,
        help="Папка независимых фотографий"
    )

    parser.add_argument(
        "--images",
        type=int,
        default=600
    )

    args = parser.parse_args()

    result = run_additional_checks(
        dataset_root=args.data,
        independent_folder=args.photos,
        robustness_images=args.images,
        progress_callback=print
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        )
    )


if __name__ == "__main__":
    main()
