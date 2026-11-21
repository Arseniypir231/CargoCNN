
"""Веб-интерфейс CargoCNN на Streamlit."""

import hashlib

import requests
import streamlit as st
from PIL import Image


API_URL = "http://127.0.0.1:8000"

# Отдельная сессия для обращения к локальному API.
# Отключаем системный прокси для запросов к FastAPI.
API_SESSION = requests.Session()
API_SESSION.trust_env = False


CLASS_LABELS = {
    "barrel": "Бочка",
    "box": "Коробка",
    "sack": "Мешок",
}


st.set_page_config(
    page_title="CargoCNN — Распознавание грузов",
    page_icon="📦",
    layout="wide",
)


def check_api():
    """Проверка доступности API."""

    try:
        response = API_SESSION.get(
            f"{API_URL}/health",
            timeout=5,
        )
        response.raise_for_status()
        return response.json()

    except requests.RequestException:
        return None

    except ValueError:
        return None


def send_prediction(image_bytes, filename, mime_type, data):
    """Отправка фотографии и характеристик груза в FastAPI."""

    files = {
        "file": (
            filename,
            image_bytes,
            mime_type,
        )
    }

    response = API_SESSION.post(
        f"{API_URL}/predict",
        files=files,
        data=data,
        timeout=120,
    )

    try:
        result = response.json()
    except ValueError:
        result = {"detail": response.text}

    if not response.ok:
        detail = result.get(
            "detail",
            "Неизвестная ошибка",
        )

        if isinstance(detail, list):
            detail = "; ".join(
                str(item) for item in detail
            )

        raise RuntimeError(
            f"Ошибка API ({response.status_code}): {detail}"
        )

    return result


def show_result(result):
    """Отображение результатов распознавания."""

    st.subheader("Результат распознавания")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Предсказанный класс",
        result["predicted_label"],
    )

    col2.metric(
        "Уверенность CNN",
        f"{result['confidence']:.2%}",
    )

    col3.metric(
        "Порог уверенности",
        f"{result['confidence_threshold']:.0%}",
    )

    st.markdown("#### Вероятности классов")

    probability_columns = st.columns(3)

    for column, cargo_class in zip(
        probability_columns,
        ("barrel", "box", "sack"),
    ):
        probability = result["probabilities"][cargo_class]

        with column:
            st.write(f"**{CLASS_LABELS[cargo_class]}**")
            st.progress(
                probability,
                text=f"{probability:.2%}",
            )

    if result["final_label"] is not None:
        st.success(
            f"Итоговый класс: {result['final_label']}"
        )

    if result["volume_m3"] is not None:
        st.write(
            f"**Расчётный объём:** "
            f"{result['volume_m3']:.3f} м³"
        )

    st.markdown("#### Рекомендации экспертной системы")

    for recommendation in result["recommendations"]:
        st.markdown(f"- {recommendation}")

    st.caption(
        f"Модель: эпоха {result['model_epoch']} · "
        f"Устройство: {result['device']}"
    )


st.title("📦 CargoCNN")
st.subheader("Система распознавания и обработки грузов")

st.write(
    "Загрузите фотографию бочки, коробки или мешка. "
    "Система определит тип груза и сформирует рекомендации "
    "с учётом его характеристик."
)


# Проверка состояния сервера.

health = check_api()

if health is None:
    st.error(
        "Сервер CargoCNN API недоступен. "
        "Проверьте, запущен ли FastAPI на порту 8000."
    )
    st.stop()

st.sidebar.success("Сервер API работает")

st.sidebar.write(
    f"Устройство: {health['device']}"
)

st.sidebar.write(
    f"Эпоха модели: {health['model_epoch']}"
)

st.sidebar.write(
    f"Порог: {health['confidence_threshold']:.0%}"
)


st.divider()

st.header("1. Загрузка фотографии")

uploaded_file = st.file_uploader(
    "Выберите фотографию груза",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()

    image_hash = hashlib.sha256(
        image_bytes
    ).hexdigest()

    if st.session_state.get("image_hash") != image_hash:
        st.session_state["image_hash"] = image_hash
        st.session_state.pop("last_result", None)
        st.session_state.pop("last_request", None)
        st.session_state.pop("last_image", None)

    try:
        with Image.open(uploaded_file) as image:
            st.image(
                image,
                caption=uploaded_file.name,
                use_container_width=True,
            )

    except Exception:
        st.error("Не удалось открыть изображение.")
        st.stop()

    st.divider()

    st.header("2. Характеристики груза")

    with st.form("cargo_form"):

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Масса и габариты**")

            mass_kg = st.text_input(
                "Масса, кг",
                placeholder="Например, 25",
            )

            length_cm = st.text_input(
                "Длина, см",
                placeholder="Например, 60",
            )

            width_cm = st.text_input(
                "Ширина, см",
                placeholder="Например, 40",
            )

            height_cm = st.text_input(
                "Высота, см",
                placeholder="Например, 35",
            )

        with col2:
            st.markdown("**Дополнительные свойства**")

            fragile = st.checkbox(
                "Хрупкий груз"
            )

            moisture_sensitive = st.checkbox(
                "Чувствителен к влаге"
            )

            keep_upright = st.checkbox(
                "Требуется сохранять вертикальное положение"
            )

        submitted = st.form_submit_button(
            "Распознать груз",
            type="primary",
            use_container_width=True,
        )

    if submitted:

        request_data = {
            "mass_kg": mass_kg.strip(),
            "length_cm": length_cm.strip(),
            "width_cm": width_cm.strip(),
            "height_cm": height_cm.strip(),
            "fragile": str(fragile).lower(),
            "moisture_sensitive": str(
                moisture_sensitive
            ).lower(),
            "keep_upright": str(
                keep_upright
            ).lower(),
            "confirmed_class": "",
        }

        try:
            with st.spinner(
                "Выполняется распознавание..."
            ):
                result = send_prediction(
                    image_bytes,
                    uploaded_file.name,
                    uploaded_file.type or "image/jpeg",
                    request_data,
                )

            st.session_state["last_result"] = result
            st.session_state["last_request"] = request_data
            st.session_state["last_image"] = image_bytes
            st.session_state["last_filename"] = uploaded_file.name
            st.session_state["last_mime_type"] = (
                uploaded_file.type or "image/jpeg"
            )

        except requests.RequestException as error:
            st.error(
                f"Ошибка соединения с API: {error}"
            )

        except RuntimeError as error:
            st.error(str(error))


# Обработка результата и ручное подтверждение.

result = st.session_state.get("last_result")

if result is not None:

    st.divider()

    if result["manual_review_required"]:

        st.warning(
            "Уверенность CNN ниже установленного порога. "
            "Необходимо вручную подтвердить тип груза."
        )

        st.markdown("#### Подтверждение типа груза")

        with st.form("confirmation_form"):

            confirmed_class = st.selectbox(
                "Выберите фактический тип груза",
                options=list(CLASS_LABELS.keys()),
                format_func=lambda value: CLASS_LABELS[value],
            )

            confirm_submitted = st.form_submit_button(
                "Подтвердить класс и сформировать рекомендации",
                type="primary",
                use_container_width=True,
            )

        if confirm_submitted:

            confirmed_data = dict(
                st.session_state["last_request"]
            )

            confirmed_data["confirmed_class"] = (
                confirmed_class
            )

            try:
                with st.spinner(
                    "Формирование рекомендаций..."
                ):
                    confirmed_result = send_prediction(
                        st.session_state["last_image"],
                        st.session_state["last_filename"],
                        st.session_state["last_mime_type"],
                        confirmed_data,
                    )

                st.session_state["last_result"] = (
                    confirmed_result
                )

                st.session_state["last_request"] = (
                    confirmed_data
                )

                st.rerun()

            except requests.RequestException as error:
                st.error(
                    f"Ошибка соединения с API: {error}"
                )

            except RuntimeError as error:
                st.error(str(error))

    else:
        show_result(result)
