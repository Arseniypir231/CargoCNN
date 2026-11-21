
"""FastAPI для распознавания и обработки грузов."""

from contextlib import asynccontextmanager
from io import BytesIO
from math import isfinite
from threading import Lock

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from PIL import Image, UnidentifiedImageError

from src.expert_system import CargoDetails
from src.inference import CargoInference
from src.torch_data import CLASSES


MAX_IMAGE_SIZE = 10 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG"}


def parse_optional_number(
    value: str | None,
    field_name: str,
) -> float | None:
    """Преобразовать необязательное числовое поле."""

    if value is None or not value.strip():
        return None

    try:
        number = float(value.strip().replace(",", "."))
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name}: требуется положительное число.",
        )

    if not isfinite(number) or number <= 0:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name}: значение должно быть больше нуля.",
        )

    return number


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка CNN при запуске сервера."""

    print("Загрузка модели CNN...")

    app.state.engine = CargoInference()
    app.state.inference_lock = Lock()

    print("Модель загружена.")
    print("Устройство:", app.state.engine.device)

    yield

    app.state.engine = None
    print("Работа API завершена.")


app = FastAPI(
    title="CargoCNN API",
    description=(
        "API классификации грузов и формирования "
        "рекомендаций экспертной системы."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "name": "CargoCNN API",
        "status": "running",
        "documentation": "/docs",
    }


@app.get("/health")
def health(request: Request):
    """Проверка состояния сервера и модели."""

    engine = getattr(request.app.state, "engine", None)

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Модель CNN не загружена.",
        )

    return {
        "status": "ok",
        "model_loaded": True,
        "model_epoch": engine.checkpoint.get("epoch"),
        "device": str(engine.device),
        "image_size": engine.image_size,
        "confidence_threshold": engine.threshold,
        "classes": list(CLASSES),
    }


@app.post("/predict")
def predict(
    request: Request,
    file: UploadFile = File(...),

    # Числовые характеристики принимаются как строки.
    # Это позволяет корректно обрабатывать пустые поля Swagger.
    mass_kg: str | None = Form(None),
    length_cm: str | None = Form(None),
    width_cm: str | None = Form(None),
    height_cm: str | None = Form(None),

    fragile: bool = Form(False),
    moisture_sensitive: bool = Form(False),
    keep_upright: bool = Form(False),

    confirmed_class: str | None = Form(None),
):
    """Распознавание фотографии и формирование рекомендаций."""

    engine = getattr(request.app.state, "engine", None)

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Модель CNN не загружена.",
        )

    # Чтение файла с ограничением размера.
    try:
        content = file.file.read(MAX_IMAGE_SIZE + 1)
    finally:
        file.file.close()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Загружен пустой файл.",
        )

    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Размер изображения не должен превышать 10 МБ.",
        )

    # Проверка формата и открытие изображения.
    try:
        with Image.open(BytesIO(content)) as source:

            image_format = source.format

            if image_format not in ALLOWED_FORMATS:
                raise HTTPException(
                    status_code=415,
                    detail="Поддерживаются только JPG и PNG.",
                )

            source.load()
            image = source.convert("RGB")

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Не удалось прочитать изображение.",
        )

    # Преобразование необязательных характеристик.
    details = CargoDetails(
        mass_kg=parse_optional_number(
            mass_kg, "Масса, кг"
        ),
        length_cm=parse_optional_number(
            length_cm, "Длина, см"
        ),
        width_cm=parse_optional_number(
            width_cm, "Ширина, см"
        ),
        height_cm=parse_optional_number(
            height_cm, "Высота, см"
        ),
        fragile=fragile,
        moisture_sensitive=moisture_sensitive,
        keep_upright=keep_upright,
    )

    # Пустое значение не считается подтверждением класса.
    if confirmed_class is not None:
        confirmed_class = confirmed_class.strip() or None

    # Выполнение распознавания и экспертной обработки.
    try:
        with request.app.state.inference_lock:
            result = engine.predict(
                image=image,
                details=details,
                confirmed_class=confirmed_class,
            )

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )

    return {
        "filename": file.filename,
        "image_format": image_format,
        **result,
    }
