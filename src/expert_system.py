
"""Экспертные правила обработки грузов (неделя 6).

Модуль использует результаты CNN и характеристики, введённые пользователем.
Объём рассчитывается с учётом геометрии распознанного типа груза.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi
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
HEAVY_CARGO_KG = 25.0


@dataclass(frozen=True)
class CargoDetails:
    """Характеристики груза, вводимые пользователем."""

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


def _positive_number(
    value: object,
    field_name: str,
) -> Optional[float]:

    if value is None or (
        isinstance(value, str) and not value.strip()
    ):
        return None

    if isinstance(value, bool):
        raise ValueError(
            f"{field_name}: требуется положительное число."
        )

    try:
        number = float(str(value).replace(",", "."))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name}: требуется положительное число."
        ) from exc

    if not isfinite(number) or number <= 0:
        raise ValueError(
            f"{field_name}: значение должно быть больше нуля."
        )

    return number


def _probability(
    value: object,
    name: str,
    *,
    allow_zero: bool,
) -> float:

    if isinstance(value, bool):
        raise ValueError(
            f"{name}: требуется число от 0 до 1."
        )

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name}: требуется число от 0 до 1."
        ) from exc

    if (
        not isfinite(number)
        or number > 1
        or (number < 0 if allow_zero else number <= 0)
    ):
        raise ValueError(
            f"{name}: значение вне допустимого диапазона."
        )

    return number


def _cargo_class(value: str) -> str:

    if (
        not isinstance(value, str)
        or value.strip().lower() not in CARGO_LABELS
    ):
        raise ValueError(
            "Тип груза должен быть barrel, box или sack."
        )

    return value.strip().lower()


def _calculate_volume(
    cargo_class: str,
    length: float,
    width: float,
    height: float,
) -> float:
    """Расчёт объёма с учётом геометрии грузовой тары.

    Для бочки длина и ширина являются двумя замерами диаметра.
    Используется их среднее значение.

    Для коробки рассчитывается объём параллелепипеда.
    Для мешка определяется ориентировочный габаритный объём.
    Все размеры передаются в сантиметрах.
    """

    if cargo_class == "barrel":

        diameter = (length + width) / 2
        radius = diameter / 2

        volume_cm3 = pi * radius ** 2 * height

    else:

        volume_cm3 = length * width * height

    return volume_cm3 / 1_000_000


def evaluate_cargo(
    predicted_class: str,
    confidence: float,
    details: Optional[CargoDetails] = None,
    *,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    confirmed_class: Optional[str] = None,
) -> ExpertResult:
    """Применить экспертные правила к результату CNN.

    При низкой уверенности требуется ручное подтверждение.
    Геометрия объёма определяется по итоговому классу груза.
    """

    predicted = _cargo_class(predicted_class)

    confidence_value = _probability(
        confidence,
        "Уверенность CNN",
        allow_zero=True,
    )

    threshold_value = _probability(
        threshold,
        "Порог уверенности",
        allow_zero=False,
    )

    if details is None:
        details = CargoDetails()

    if not isinstance(details, CargoDetails):
        raise TypeError(
            "Параметр details должен иметь тип CargoDetails."
        )

    mass = _positive_number(
        details.mass_kg,
        "Масса, кг",
    )

    length = _positive_number(
        details.length_cm,
        "Длина, см",
    )

    width = _positive_number(
        details.width_cm,
        "Ширина, см",
    )

    height = _positive_number(
        details.height_cm,
        "Высота, см",
    )

    for flag_name in (
        "fragile",
        "moisture_sensitive",
        "keep_upright",
    ):
        if not isinstance(
            getattr(details, flag_name),
            bool,
        ):
            raise ValueError(
                f"{flag_name}: требуется значение True или False."
            )

    verified = (
        _cargo_class(confirmed_class)
        if confirmed_class is not None
        else None
    )

    requires_review = (
        confidence_value < threshold_value
        and verified is None
    )

    # При недостаточной уверенности ждём подтверждения класса.
    if requires_review:

        return ExpertResult(
            predicted_class=predicted,
            final_class=None,
            confidence=confidence_value,
            confidence_threshold=threshold_value,
            manual_review_required=True,
            volume_m3=None,
            recommendations=(
                "Уверенность CNN ниже выбранного порога. "
                "Необходимо вручную подтвердить тип груза "
                "перед выдачей рекомендаций.",
            ),
        )

    # Подтверждённый класс имеет приоритет над предсказанием CNN.
    cargo_class = verified or predicted

    # Объём рассчитывается только при наличии всех трёх размеров.
    volume = None

    dimensions_provided = all(
        value is not None
        for value in (length, width, height)
    )

    if dimensions_provided:
        volume = _calculate_volume(
            cargo_class,
            length,
            width,
            height,
        )

    recommendations = list(
        BASE_RULES[cargo_class]
    )

    # Правила обработки массы.
    if mass is not None:

        if mass >= HEAVY_CARGO_KG:

            recommendations.append(
                f"Масса {mass:g} кг: рассмотреть применение "
                "грузоподъёмного оборудования, рассчитанного "
                "на соответствующую нагрузку."
            )

        else:

            recommendations.append(
                f"Масса {mass:g} кг: учитывать её при выборе "
                "способа перемещения и соблюдать действующие "
                "требования охраны труда."
            )

    # Правила обработки габаритов и расчёта объёма.
    supplied_dimensions = [
        value is not None
        for value in (length, width, height)
    ]

    if all(supplied_dimensions):

        if cargo_class == "barrel":

            diameter = (length + width) / 2

            recommendations.append(
                f"Размеры бочки: {length:g} × {width:g} × "
                f"{height:g} см. Средний диаметр: "
                f"{diameter:g} см. Расчётный геометрический "
                f"объём цилиндра: {volume:.3f} м³. "
                "Проверить достаточность места для хранения "
                "и перемещения."
            )

        elif cargo_class == "box":

            recommendations.append(
                f"Габариты коробки: {length:g} × {width:g} × "
                f"{height:g} см. Габаритный объём: "
                f"{volume:.3f} м³. Проверить достаточность "
                "места для хранения и перемещения."
            )

        else:

            recommendations.append(
                f"Габариты мешка: {length:g} × {width:g} × "
                f"{height:g} см. Ориентировочный габаритный "
                f"объём (по внешнему параллелепипеду): "
                f"{volume:.3f} м³. Учитывать форму и степень "
                "наполнения мешка при размещении."
            )

    elif any(supplied_dimensions):

        recommendations.append(
            "Для оценки занимаемого объёма необходимо "
            "указать все три габарита."
        )

    # Дополнительные свойства груза.
    if details.fragile:

        recommendations.append(
            "Хрупкий груз: исключить удары и предусмотреть "
            "защитную упаковку."
        )

    if details.moisture_sensitive:

        recommendations.append(
            "Груз чувствителен к влаге: обеспечить защиту "
            "от намокания."
        )

    if details.keep_upright:

        recommendations.append(
            "Сохранять вертикальное положение груза "
            "при перемещении и хранении."
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
