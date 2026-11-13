

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from src.expert_system import (
    CARGO_LABELS,
    CargoDetails,
    evaluate_cargo,
)


class Week6Tab(ttk.Frame):
    """Вкладка распознавания и применения экспертных правил."""

    def __init__(self, master):
        super().__init__(master, padding=14)

        self.photo_path = tk.StringVar()
        self.mass_kg = tk.StringVar()
        self.length_cm = tk.StringVar()
        self.width_cm = tk.StringVar()
        self.height_cm = tk.StringVar()

        self.fragile = tk.BooleanVar(value=False)
        self.moisture_sensitive = tk.BooleanVar(value=False)
        self.keep_upright = tk.BooleanVar(value=False)

        self.confirmed_class = tk.StringVar(value="")

        self.prediction_text = tk.StringVar(
            value="Выберите изображение и нажмите «Распознать CNN»."
        )

        self.status_text = tk.StringVar(
            value="Экспертная система не изменяет веса нейронной сети."
        )

        self._predicted_class = None
        self._confidence = None
        self._threshold = 0.70
        self._preview = None
        self._working = False

        self._build()

    def _build(self):
        ttk.Label(
            self,
            text="Экспертная система обработки грузов",
            font=("Segoe UI", 15, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            self,
            text=(
                "CNN определяет тип груза по фотографии. Массу, габариты и "
                "свойства пользователь вводит вручную."
            ),
            wraplength=1050,
        ).pack(anchor="w", pady=(4, 12))

        photo_box = ttk.LabelFrame(
            self,
            text="1. Распознавание груза",
            padding=10,
        )
        photo_box.pack(fill="x")

        photo_row = ttk.Frame(photo_box)
        photo_row.pack(fill="x")

        ttk.Entry(
            photo_row,
            textvariable=self.photo_path,
            state="readonly",
        ).pack(
            side="left",
            fill="x",
            expand=True,
        )

        ttk.Button(
            photo_row,
            text="Выбрать фотографию…",
            command=self._select_photo,
        ).pack(
            side="left",
            padx=(8, 0),
        )

        self.recognize_button = ttk.Button(
            photo_row,
            text="Распознать CNN",
            command=self._recognize,
        )
        self.recognize_button.pack(
            side="left",
            padx=(8, 0),
        )

        info_row = ttk.Frame(photo_box)
        info_row.pack(
            fill="x",
            pady=(10, 0),
        )

        self.preview_label = ttk.Label(
            info_row,
            text="Предпросмотр\nизображения",
            anchor="center",
            width=30,
        )
        self.preview_label.pack(
            side="left",
            padx=(0, 16),
        )

        prediction_box = ttk.Frame(info_row)
        prediction_box.pack(
            side="left",
            fill="both",
            expand=True,
        )

        ttk.Label(
            prediction_box,
            textvariable=self.prediction_text,
            font=("Segoe UI", 11, "bold"),
            wraplength=800,
        ).pack(anchor="w")

        ttk.Label(
            prediction_box,
            text=(
                "При уверенности ниже 70% результат нужно проверить вручную. "
                "Фактический порог загружается из результатов пятой недели."
            ),
            foreground="#555555",
            wraplength=800,
        ).pack(
            anchor="w",
            pady=(6, 0),
        )

        confirm_row = ttk.Frame(prediction_box)
        confirm_row.pack(
            fill="x",
            pady=(10, 0),
        )

        ttk.Label(
            confirm_row,
            text="Ручное подтверждение типа:",
        ).pack(side="left")

        self.confirm_box = ttk.Combobox(
            confirm_row,
            textvariable=self.confirmed_class,
            values=(
                "",
                "barrel",
                "box",
                "sack",
            ),
            state="readonly",
            width=15,
        )
        self.confirm_box.pack(
            side="left",
            padx=(8, 0),
        )

        ttk.Label(
            confirm_row,
            text="Оставьте пустым, если подтверждение не требуется.",
            foreground="#666666",
        ).pack(
            side="left",
            padx=(12, 0),
        )

        details_box = ttk.LabelFrame(
            self,
            text="2. Характеристики груза (вводятся вручную)",
            padding=10,
        )
        details_box.pack(
            fill="x",
            pady=(12, 0),
        )

        main_fields = ttk.Frame(details_box)
        main_fields.pack(
            anchor="w",
            fill="x",
        )

        self._numeric_field(
            main_fields,
            "Масса, кг:",
            self.mass_kg,
            12,
        )

        self._numeric_field(
            main_fields,
            "Длина, см:",
            self.length_cm,
            12,
        )

        self._numeric_field(
            main_fields,
            "Ширина, см:",
            self.width_cm,
            12,
        )

        self._numeric_field(
            main_fields,
            "Высота, см:",
            self.height_cm,
            12,
        )

        properties_row = ttk.Frame(details_box)
        properties_row.pack(
            fill="x",
            pady=(12, 2),
        )

        ttk.Checkbutton(
            properties_row,
            text="Хрупкий груз",
            variable=self.fragile,
        ).pack(
            side="left",
            padx=(0, 24),
        )

        ttk.Checkbutton(
            properties_row,
            text="Боится влаги",
            variable=self.moisture_sensitive,
        ).pack(
            side="left",
            padx=(0, 24),
        )

        ttk.Checkbutton(
            properties_row,
            text="Требуется вертикальное положение",
            variable=self.keep_upright,
        ).pack(side="left")

        ttk.Label(
            details_box,
            text=(
                "Массу и габариты можно не указывать. Если введены все три "
                "размера, программа вычислит объём. Правило о массе от 25 кг "
                "используется только в учебных целях."
            ),
            foreground="#666666",
            wraplength=1050,
        ).pack(
            anchor="w",
            pady=(9, 0),
        )

        result_box = ttk.LabelFrame(
            self,
            text="3. Результат экспертной системы",
            padding=10,
        )
        result_box.pack(
            fill="both",
            expand=True,
            pady=(12, 0),
        )

        action_row = ttk.Frame(result_box)
        action_row.pack(fill="x")

        self.recommend_button = ttk.Button(
            action_row,
            text="Сформировать рекомендации",
            command=self._make_recommendations,
        )
        self.recommend_button.pack(side="left")

        ttk.Button(
            action_row,
            text="Очистить результат",
            command=lambda: self._set_recommendations(""),
        ).pack(
            side="left",
            padx=(8, 0),
        )

        ttk.Label(
            result_box,
            textvariable=self.status_text,
            foreground="#174a7e",
        ).pack(
            anchor="w",
            pady=(8, 6),
        )

        text_frame = ttk.Frame(result_box)
        text_frame.pack(
            fill="both",
            expand=True,
        )

        self.result_text = tk.Text(
            text_frame,
            height=9,
            wrap="word",
            font=("Segoe UI", 10),
            state="disabled",
        )
        self.result_text.pack(
            side="left",
            fill="both",
            expand=True,
        )

        scroll = ttk.Scrollbar(
            text_frame,
            orient="vertical",
            command=self.result_text.yview,
        )
        scroll.pack(
            side="right",
            fill="y",
        )

        self.result_text.configure(
            yscrollcommand=scroll.set,
        )

    @staticmethod
    def _numeric_field(parent, label, variable, width):
        group = ttk.Frame(parent)
        group.pack(
            side="left",
            padx=(0, 14),
        )

        ttk.Label(
            group,
            text=label,
        ).pack(anchor="w")

        ttk.Entry(
            group,
            textvariable=variable,
            width=width,
        ).pack(
            anchor="w",
            pady=(3, 0),
        )

    def _set_recommendations(self, content):
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")

        if content:
            self.result_text.insert("1.0", content)

        self.result_text.configure(state="disabled")

    def _select_photo(self):
        filename = filedialog.askopenfilename(
            parent=self,
            title="Выберите фотографию груза",
            filetypes=[
                ("Изображения", "*.jpg *.jpeg *.png"),
                ("Все файлы", "*.*"),
            ],
        )

        if not filename:
            return

        self.photo_path.set(filename)
        self._predicted_class = None
        self._confidence = None
        self._threshold = 0.70

        self.confirmed_class.set("")

        self.prediction_text.set(
            "Фотография выбрана. Нажмите «Распознать CNN»."
        )

        self.status_text.set(
            "Ожидание распознавания."
        )

        self._set_recommendations("")

        try:
            from PIL import Image, ImageTk

            with Image.open(filename) as source:
                picture = source.convert("RGB")

            picture.thumbnail((240, 150))

            self._preview = ImageTk.PhotoImage(picture)

            self.preview_label.configure(
                image=self._preview,
                text="",
            )

        except Exception as exc:
            self._preview = None

            self.preview_label.configure(
                image="",
                text="Не удалось показать предпросмотр.",
            )

            messagebox.showerror(
                "Ошибка изображения",
                str(exc),
                parent=self,
            )

    def _recognize(self):
        if self._working:
            return

        photo = self.photo_path.get().strip()

        if not photo or not Path(photo).is_file():
            messagebox.showwarning(
                "Фотография не выбрана",
                "Сначала выберите существующий файл изображения.",
                parent=self,
            )
            return

        self._working = True

        self.recognize_button.state(["disabled"])
        self.recommend_button.state(["disabled"])

        self._predicted_class = None
        self._confidence = None

        self.confirmed_class.set("")

        self.prediction_text.set(
            "Идёт распознавание фотографии…"
        )

        self.status_text.set(
            "Загружается сохранённая CNN. Обучение не выполняется."
        )

        self._set_recommendations("")

        threading.Thread(
            target=self._recognize_worker,
            args=(photo,),
            daemon=True,
        ).start()

    def _recognize_worker(self, photo):
        try:
            from PIL import Image
            import torch

            from src.evaluate_cnn import (
                load_selected_confidence_threshold,
                load_trained_model,
            )

            from src.torch_data import (
                CLASSES,
                _to_tensor,
            )

            model, device, checkpoint, image_size = (
                load_trained_model()
            )

            with Image.open(photo) as source:
                image = source.convert("RGB")

            image = image.resize(
                (image_size, image_size),
                Image.Resampling.BILINEAR,
            )

            input_tensor = (
                _to_tensor(image)
                .unsqueeze(0)
                .to(device)
            )

            with torch.no_grad():
                prediction = model(input_tensor)

                probabilities = torch.softmax(
                    prediction,
                    dim=1,
                )[0]

                index = int(
                    probabilities.argmax().item()
                )

                confidence = float(
                    probabilities[index].item()
                )

            cargo_class = CLASSES[index]

            threshold, threshold_path = (
                load_selected_confidence_threshold(
                    checkpoint.get("epoch")
                )
            )

            self.after(
                0,
                lambda: self._recognition_done(
                    cargo_class,
                    confidence,
                    threshold,
                    threshold_path,
                ),
            )

        except Exception as exc:
            error = str(exc)

            self.after(
                0,
                lambda message=error: (
                    self._recognition_error(message)
                ),
            )

    def _recognition_done(
        self,
        cargo_class,
        confidence,
        threshold,
        threshold_path,
    ):
        self._working = False

        self.recognize_button.state(["!disabled"])
        self.recommend_button.state(["!disabled"])

        self._predicted_class = cargo_class
        self._confidence = confidence
        self._threshold = threshold

        name = CARGO_LABELS[cargo_class]

        self.prediction_text.set(
            f"CNN: {name} ({cargo_class}). "
            f"Уверенность: {confidence:.2%}. "
            f"Порог: {threshold:.0%}."
        )

        if confidence < threshold:
            self.status_text.set(
                "Уверенность ниже порога. Перед выдачей рекомендаций "
                "подтвердите тип груза в выпадающем списке."
            )
        else:
            self.status_text.set(
                "Распознавание завершено. "
                "Можно сформировать рекомендации."
            )

        if threshold_path is None:
            self.status_text.set(
                self.status_text.get()
                + " Файл порога не найден: "
                "используется резервное значение 70%."
            )

    def _recognition_error(self, message):
        self._working = False

        self.recognize_button.state(["!disabled"])
        self.recommend_button.state(["!disabled"])

        self._predicted_class = None
        self._confidence = None

        self.prediction_text.set(
            "Не удалось распознать фотографию."
        )

        self.status_text.set(
            "Ошибка загрузки или распознавания CNN."
        )

        messagebox.showerror(
            "Ошибка распознавания",
            message,
            parent=self,
        )

    def _make_recommendations(self):
        if self._working:
            return

        if (
            self._predicted_class is None
            or self._confidence is None
        ):
            messagebox.showwarning(
                "Результат CNN отсутствует",
                "Сначала выберите и распознайте фотографию груза.",
                parent=self,
            )
            return

        try:
            details = CargoDetails(
                mass_kg=(
                    self.mass_kg.get().strip() or None
                ),
                length_cm=(
                    self.length_cm.get().strip() or None
                ),
                width_cm=(
                    self.width_cm.get().strip() or None
                ),
                height_cm=(
                    self.height_cm.get().strip() or None
                ),
                fragile=self.fragile.get(),
                moisture_sensitive=self.moisture_sensitive.get(),
                keep_upright=self.keep_upright.get(),
            )

            result = evaluate_cargo(
                self._predicted_class,
                self._confidence,
                details,
                threshold=self._threshold,
                confirmed_class=(
                    self.confirmed_class.get() or None
                ),
            )

        except (ValueError, TypeError) as exc:
            # При ошибке удаляем прежние рекомендации,
            # чтобы пользователь не принял их за актуальные.
            self._set_recommendations("")

            self.status_text.set(
                "Ошибка ввода: рекомендации не сформированы. "
                "Исправьте характеристики груза."
            )

            messagebox.showerror(
                "Проверьте характеристики груза",
                str(exc),
                parent=self,
            )
            return

        lines = [
            f"Предсказание CNN: "
            f"{CARGO_LABELS[result.predicted_class]}",
            f"Уверенность: {result.confidence:.2%}",
            f"Порог: {result.confidence_threshold:.0%}",
        ]

        if result.manual_review_required:
            lines.extend(
                [
                    "Статус: требуется ручная проверка типа груза.",
                    "",
                    *result.recommendations,
                ]
            )

            self.status_text.set(
                "Рекомендации по типу груза "
                "не выдаются без подтверждения."
            )

        else:
            lines.append(
                f"Используемый тип: "
                f"{CARGO_LABELS[result.final_class]}"
            )

            if result.volume_m3 is not None:
                lines.append(
                    f"Объём: {result.volume_m3:.3f} м³"
                )

            lines.extend(
                [
                    "",
                    "Рекомендации:",
                    *(
                        f"{number}. {recommendation}"
                        for number, recommendation in enumerate(
                            result.recommendations,
                            start=1,
                        )
                    ),
                ]
            )

            self.status_text.set(
                "Экспертные правила применены."
            )

        self._set_recommendations(
            "\n".join(lines)
        )
