
"""Общий модуль распознавания грузов для веб-приложения."""

from __future__ import annotations

import torch
from PIL import Image

from src.evaluate_cnn import (
    load_selected_confidence_threshold,
    load_trained_model,
)
from src.expert_system import (
    CargoDetails,
    CARGO_LABELS,
    evaluate_cargo,
)
from src.torch_data import CLASSES, _to_tensor


class CargoInference:
    """Загрузка CNN и выполнение распознавания грузов."""

    def __init__(self):
        # Загружаем обученную модель один раз.
        (
            self.model,
            self.device,
            self.checkpoint,
            self.image_size,
        ) = load_trained_model()

        # Получаем порог, выбранный на Validation.
        self.threshold, self.threshold_path = (
            load_selected_confidence_threshold(
                self.checkpoint.get("epoch")
            )
        )

        self.classes = tuple(CLASSES)

        if len(self.classes) != 3:
            raise RuntimeError(
                "Ожидалось ровно три класса грузов."
            )

    def predict(
        self,
        image: Image.Image,
        details: CargoDetails | None = None,
        confirmed_class: str | None = None,
    ) -> dict:
        """Распознать изображение и сформировать рекомендации."""

        if not isinstance(image, Image.Image):
            raise TypeError(
                "В predict необходимо передать изображение PIL."
            )

        if details is None:
            details = CargoDetails()

        # Подготовка изображения по существующему алгоритму.
        image = image.convert("RGB")

        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR,
        )

        input_tensor = (
            _to_tensor(image)
            .unsqueeze(0)
            .to(self.device)
        )

        # Распознавание без обучения модели.
        with torch.inference_mode():
            logits = self.model(input_tensor)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )[0].detach().cpu()

        if probabilities.numel() != len(self.classes):
            raise RuntimeError(
                "Количество выходов CNN не совпадает "
                "с количеством классов."
            )

        predicted_index = int(
            torch.argmax(probabilities).item()
        )

        predicted_class = self.classes[predicted_index]

        confidence = float(
            probabilities[predicted_index].item()
        )

        # Применение существующих экспертных правил.
        expert_result = evaluate_cargo(
            predicted_class=predicted_class,
            confidence=confidence,
            details=details,
            threshold=self.threshold,
            confirmed_class=confirmed_class,
        )

        # Подготовка результата для передачи через API.
        return {
            "predicted_class": expert_result.predicted_class,
            "predicted_label": CARGO_LABELS[
                expert_result.predicted_class
            ],

            "probabilities": {
                cargo_class: float(
                    probabilities[index].item()
                )
                for index, cargo_class in enumerate(
                    self.classes
                )
            },

            "confidence": expert_result.confidence,
            "confidence_threshold": (
                expert_result.confidence_threshold
            ),

            "manual_review_required": (
                expert_result.manual_review_required
            ),

            "final_class": expert_result.final_class,
            "final_label": (
                CARGO_LABELS.get(expert_result.final_class)
                if expert_result.final_class is not None
                else None
            ),

            "volume_m3": expert_result.volume_m3,

            "recommendations": list(
                expert_result.recommendations
            ),

            "model_epoch": self.checkpoint.get("epoch"),
            "device": str(self.device),
        }
