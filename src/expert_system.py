"""Экспертные правила обработки грузов (неделя 6).

Модуль не обучает CNN и не определяет массу/размеры по фотографии:
класс и уверенность поступают от CNN, характеристики вводит пользователь.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Optional


CARGO_LABELS = {
    "barrel": "Бочка",
    "box": "Коробка",
    "sack": "Мешок",
}

BASE_RULES = {
    "barrel": (
        "Проверить целостность и герметичность бочки.",
        "Предусмотреть крепление, исключающее перекатывание при перевозке.",
    ),
    "box": (
        "Проверить целостность коробки и прочность упаковки.",
        "Учитывать возможность смятия коробки при размещении других грузов.",
    ),
    "sack": (
        "Проверить целостность мешка и отсутствие повреждений швов.",
        "Предусмотреть защиту упаковки от проколов и рассыпания содержимого.",
    ),
}

DEFAULT_CONFIDENCE_THRESHOLD = 0.70
HEAVY_CARGO_KG = 25.0  # Учебное правило, не норматив ручного подъёма.


@dataclass(frozen=True)
class CargoDetails:
    """Характеристики груза, вводимые пользователем (поля необязательны)."""

    mass_kg: Optional[float] = None
    length_cm: Optional[float] = None
    width_cm: Optional[float] = None
    height_cm: Optional[float] = None
    fragile: bool = False
    moisture_sensitive: bool = False
    keep_upright: bool = False


@dataclass(frozen=True)
class ExpertResult:
    predicted_class: str
    final_class: Optional[str]
    confidence: float
    confidence_threshold: float
    manual_review_required: bool
    volume_m3: Optional[float]
    recommendations: tuple[str, ...]


def _positive_number(value: object, field_name: str) -> Optional[float]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name}: требуется положительное число.")
    try:
        number = float(str(value).replace(",", "."))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name}: требуется положительное число.") from exc
    if not isfinite(number) or number <= 0:
        raise ValueError(f"{field_name}: значение должно быть больше нуля.")
    return number


def _probability(value: object, name: str, *, allow_zero: bool) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name}: требуется число от 0 до 1.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: требуется число от 0 до 1.") from exc
    if not isfinite(number) or number > 1 or (number < 0 if allow_zero else number <= 0):
        raise ValueError(f"{name}: значение вне допустимого диапазона.")
    return number


def _cargo_class(value: str) -> str:
    if not isinstance(value, str) or value.strip().lower() not in CARGO_LABELS:
        raise ValueError("Тип груза должен быть barrel, box или sack.")
    return value.strip().lower()


def evaluate_cargo(
    predicted_class: str,
    confidence: float,
    details: Optional[CargoDetails] = None,
    *,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    confirmed_class: Optional[str] = None,
) -> ExpertResult:
    """Применить экспертные правила к предсказанию CNN и характеристикам груза.

    Если уверенность ниже порога, рекомендации по типу не формируются,
    пока пользователь не подтвердит класс вручную (confirmed_class).
    """

    predicted = _cargo_class(predicted_class)
    confidence_value = _probability(confidence, "Уверенность CNN", allow_zero=True)
    threshold_value = _probability(threshold, "Порог уверенности", allow_zero=False)

    if details is None:
        details = CargoDetails()
    if not isinstance(details, CargoDetails):
        raise TypeError("Параметр details должен иметь тип CargoDetails.")

    mass = _positive_number(details.mass_kg, "Масса, кг")
    length = _positive_number(details.length_cm, "Длина, см")
    width = _positive_number(details.width_cm, "Ширина, см")
    height = _positive_number(details.height_cm, "Высота, см")

    for flag_name in ("fragile", "moisture_sensitive", "keep_upright"):
        if not isinstance(getattr(details, flag_name), bool):
            raise ValueError(f"{flag_name}: требуется значение True или False.")

    volume = None
    if length is not None and width is not None and height is not None:
        volume = length * width * height / 1_000_000

    verified = _cargo_class(confirmed_class) if confirmed_class is not None else None
    requires_review = confidence_value < threshold_value and verified is None

    if requires_review:
        return ExpertResult(
            predicted_class=predicted,
            final_class=None,
            confidence=confidence_value,
            confidence_threshold=threshold_value,
            manual_review_required=True,
            volume_m3=volume,
            recommendations=(
                "Уверенность CNN ниже выбранного порога. "
                "Необходимо вручную подтвердить тип груза перед выдачей рекомендаций.",
            ),
        )

    cargo_class = verified or predicted
    recommendations = list(BASE_RULES[cargo_class])

    if mass is not None:
        if mass >= HEAVY_CARGO_KG:
            recommendations.append(
                f"Масса {mass:g} кг: рассмотреть применение грузоподъёмного "
                "оборудования, рассчитанного на соответствующую нагрузку."
            )
        else:
            recommendations.append(
                f"Масса {mass:g} кг: учитывать её при выборе способа перемещения "
                "и соблюдать действующие требования охраны труда."
            )

    supplied_dimensions = [value is not None for value in (length, width, height)]
    if all(supplied_dimensions):
        recommendations.append(
            f"Габариты {length:g} × {width:g} × {height:g} см, "
            f"расчётный объём {volume:.3f} м³: "
            "проверить достаточность места для хранения и перемещения."
        )
    elif any(supplied_dimensions):
        recommendations.append(
            "Для оценки занимаемого объёма необходимо указать все три габарита."
        )

    if details.fragile:
        recommendations.append(
            "Хрупкий груз: исключить удары и предусмотреть защитную упаковку."
        )
    if details.moisture_sensitive:
        recommendations.append(
            "Груз чувствителен к влаге: обеспечить защиту от намокания."
        )
    if details.keep_upright:
        recommendations.append(
            "Сохранять вертикальное положение груза при перемещении и хранении."
        )

    return ExpertResult(
        predicted_class=predicted,
        final_class=cargo_class,
        confidence=confidence_value,
        confidence_threshold=threshold_value,
        manual_review_required=False,
        volume_m3=volume,
        recommendations=tuple(recommendations),
    )
