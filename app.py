"""Графический интерфейс Cargo CNN: подготовка, обучение и тестирование (недели 2–5)."""
from pathlib import Path
import json
import os
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from src.inspect_dataset import inspect
from src.hog_svm import run


class CargoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cargo CNN — недели 2–5")
        self.geometry("1180x850")
        self.minsize(980, 680)
        self.data_dir = tk.StringVar(value="data/prepared_cargo_cnn/object_crops")
        self.status = tk.StringVar(value="Выберите папку object_crops и проверьте данные.")
        self.epochs = tk.StringVar(value="10")
        self.batch_size = tk.StringVar(value="32")
        self.w4_lrs = tk.StringVar(value="0.001, 0.0005")
        self.w4_batches = tk.StringVar(value="8, 16")
        self.w4_dropouts = tk.StringVar(value="0.0, 0.3")
        self.w4_filters = tk.StringVar(value="32,64,128,256;16,32,64,128")
        self.w4_max_epochs = tk.StringVar(value="20")
        self.w4_patience = tk.StringVar(value="4")
        self.w4_min_delta = tk.StringVar(value="0.001")
        self.w4_seed = tk.StringVar(value="42")
        self.w4_auto_batch = tk.BooleanVar(value=True)
        self.w5_batch_size = tk.StringVar(value="16")
        self.w5_robustness_images = tk.StringVar(value="600")
        self.w5_independent_dir = tk.StringVar(value="")
        # Независимый путь датасета для пятой недели.
        self.w5_dataset_dir = tk.StringVar(value=self.data_dir.get())
        self.w5_status = tk.StringVar(value="Тестирование ещё не запускалось.")
        self.w5_last_dir = None
        self.w5_start_btn = None
        self._training_busy = False
        self.week4_last_dir = None
        self.cnn_train_btn = None
        self._build()

    def _build(self):
        self.notebook = ttk.Notebook(self); self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        root = ttk.Frame(self.notebook, padding=16); self.notebook.add(root, text="Подготовка, baseline и CNN")
        ttk.Label(root, text="Распознавание типа груза", font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ttk.Label(root, text="Недели 2–3: проверка данных, HOG + SVM и первоначальная CNN", font=("Segoe UI", 10)).pack(anchor="w", pady=(0,12))
        row=ttk.Frame(root); row.pack(fill="x")
        ttk.Entry(row, textvariable=self.data_dir).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Выбрать папку…", command=self._choose).pack(side="left", padx=(8,0))
        buttons=ttk.Frame(root); buttons.pack(fill="x", pady=10)
        ttk.Button(buttons, text="Проверить датасет и показать предобработку", command=self._inspect).pack(side="left")
        self.train_btn=ttk.Button(buttons, text="Обучить HOG + SVM и оценить на test", command=self._train)
        self.train_btn.pack(side="left", padx=8)
        cnn=ttk.LabelFrame(root, text="Первоначальное обучение CNN (неделя 3)", padding=8); cnn.pack(fill="x", pady=(0,8))
        ttk.Label(cnn, text="Эпохи:").pack(side="left")
        ttk.Entry(cnn, textvariable=self.epochs, width=6).pack(side="left", padx=(4,14))
        ttk.Label(cnn, text="Размер пакета:").pack(side="left")
        ttk.Entry(cnn, textvariable=self.batch_size, width=6).pack(side="left", padx=4)
        self.cnn_train_btn=ttk.Button(cnn, text="Обучить CNN", command=self._train_cnn)
        self.cnn_train_btn.pack(side="left", padx=8)
        ttk.Button(cnn, text="Открыть графики CNN", command=self._open_cnn_plot).pack(side="left", padx=4)
        ttk.Label(root, textvariable=self.status, foreground="#174a7e").pack(anchor="w", pady=6)
        self.table=ttk.Treeview(root, columns=("class","train","valid","test","total"), show="headings", height=4)
        for col,title in [("class","Класс"),("train","Train"),("valid","Valid"),("test","Test"),("total","Всего")]:
            self.table.heading(col,text=title); self.table.column(col,anchor="center",width=130)
        self.table.pack(fill="x", pady=6)
        self.log=tk.Text(root, height=18, wrap="word", font=("Consolas",10)); self.log.pack(fill="both", expand=True)
        ttk.Label(root, text="CNN обучается с нуля; на каждой эпохе отображаются train и validation.", foreground="#666").pack(anchor="w", pady=(8,0))
        self._build_week4_tab()
        self._build_week5_tab()

    def _build_week4_tab(self):
        tab=ttk.Frame(self.notebook,padding=14); self.notebook.add(tab,text="Неделя 4 — оптимизация CNN")
        ttk.Label(tab,text="Однофакторные эксперименты на одних Train/Validation",font=("Segoe UI",14,"bold")).pack(anchor="w")
        ttk.Label(tab,text="Первое значение каждого списка — базовая конфигурация; следующие значения меняются по одному параметру.").pack(anchor="w",pady=(2,10))
        params=ttk.LabelFrame(tab,text="Конфигурации и остановка",padding=8); params.pack(fill="x")
        row1=ttk.Frame(params); row1.pack(fill="x",pady=3)
        self._entry_field(row1,"Learning Rate:",self.w4_lrs,18)
        self._entry_field(row1,"Batch Size:",self.w4_batches,14)
        self._entry_field(row1,"Dropout:",self.w4_dropouts,14)
        self._entry_field(row1,"Макс. эпох:",self.w4_max_epochs,7)
        row2=ttk.Frame(params); row2.pack(fill="x",pady=3)
        self._entry_field(row2,"Фильтры (профили через ;):",self.w4_filters,37)
        self._entry_field(row2,"Patience:",self.w4_patience,7)
        self._entry_field(row2,"Min Delta:",self.w4_min_delta,9)
        self._entry_field(row2,"Seed:",self.w4_seed,7)
        ttk.Checkbutton(params,text="Автоматически уменьшать Batch Size при CUDA OOM (перезапуск конфигурации с тем же seed)",
                        variable=self.w4_auto_batch).pack(anchor="w",pady=(5,2))
        buttons=ttk.Frame(tab); buttons.pack(fill="x",pady=8)
        self.w4_start_btn=ttk.Button(buttons,text="Запустить последовательные эксперименты",command=self._run_week4)
        self.w4_start_btn.pack(side="left")
        ttk.Button(buttons,text="Открыть таблицу",command=lambda:self._open_week4_artifact("experiment_results.csv")).pack(side="left",padx=6)
        ttk.Button(buttons,text="Открыть график сравнения",command=lambda:self._open_week4_artifact("comparison.png")).pack(side="left",padx=4)
        ttk.Button(buttons,text="Открыть папку результатов",command=self._open_week4_folder).pack(side="left",padx=4)
        self.w4_status=tk.StringVar(value="Эксперименты ещё не запускались.")
        ttk.Label(tab,textvariable=self.w4_status,foreground="#174a7e",wraplength=1100).pack(anchor="w",pady=4)
        self.w4_best=tk.StringVar(value="Лучшая конфигурация появится после завершения сравнения.")
        ttk.Label(tab,textvariable=self.w4_best,foreground="#245a32",wraplength=1100).pack(anchor="w",pady=4)
        self.w4_table=ttk.Treeview(tab,columns=("name","status","epoch","loss","accuracy","stop"),show="headings",height=5)
        headings=[("name","Эксперимент"),("status","Статус"),("epoch","Лучшая эпоха"),("loss","Best Val Loss"),("accuracy","Val Accuracy"),("stop","Причина остановки")]
        for col,title in headings:
            self.w4_table.heading(col,text=title); self.w4_table.column(col,anchor="center",width=150)
        self.w4_table.column("name",width=175); self.w4_table.column("stop",width=230)
        self.w4_table.pack(fill="x",pady=5)
        self.w4_log=tk.Text(tab,height=14,wrap="word",font=("Consolas",9)); self.w4_log.pack(fill="both",expand=True)

    def _build_week5_tab(self):
        """New testing controls. Saved checkpoints are never retrained."""
        tab = ttk.Frame(self.notebook, padding=14)
        self.notebook.add(tab, text="Неделя 5 — тестирование CNN")
        ttk.Label(
            tab, text="Тестирование готовой модели четвёртой недели",
            font=("Segoe UI", 14, "bold")
        ).pack(anchor="w")
        ttk.Label(
            tab,
            text="Загружается final_model.pth. Обучение и изменение весов не выполняются.",
            foreground="#245a32"
        ).pack(anchor="w", pady=(3, 10))

        data_box = ttk.LabelFrame(tab, text="Изображения", padding=8)
        data_box.pack(fill="x")
        row = ttk.Frame(data_box)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text="Папка object_crops:").pack(side="left")
        ttk.Entry(row, textvariable=self.w5_dataset_dir).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(row, text="Выбрать датасет…", command=self._choose_week5_dataset).pack(side="left")

        photos = ttk.Frame(data_box)
        photos.pack(fill="x", pady=3)
        ttk.Label(photos, text="Отдельные фотографии (необязательно):").pack(side="left")
        ttk.Entry(photos, textvariable=self.w5_independent_dir).pack(
            side="left", fill="x", expand=True, padx=8
        )
        ttk.Button(photos, text="Выбрать фото…", command=self._choose_week5_photos).pack(side="left")
        ttk.Button(photos, text="Очистить", command=lambda: self.w5_independent_dir.set("")).pack(
            side="left", padx=(6, 0)
        )

        params = ttk.LabelFrame(tab, text="Настройки проверки", padding=8)
        params.pack(fill="x", pady=(8, 0))
        pr = ttk.Frame(params)
        pr.pack(fill="x")
        self._entry_field(pr, "Размер пакета:", self.w5_batch_size, 7)
        self._entry_field(pr, "Изображений для проверки устойчивости:", self.w5_robustness_images, 8)
        ttk.Label(
            params,
            text="Полная оценка выполняется на всей папке Test; проверка устойчивости — на фиксированной части Test.",
            foreground="#555"
        ).pack(anchor="w", pady=(7, 0))

        controls = ttk.Frame(tab)
        controls.pack(fill="x", pady=10)
        self.w5_start_btn = ttk.Button(
            controls, text="Запустить тестирование недели 5", command=self._run_week5
        )
        self.w5_start_btn.pack(side="left")
        ttk.Button(controls, text="Папка результатов", command=self._open_week5_folder).pack(
            side="left", padx=6
        )
        ttk.Button(
            controls, text="Матрица ошибок",
            command=lambda: self._open_week5_file("cnn_confusion_matrix.png")
        ).pack(side="left", padx=3)
        ttk.Button(
            controls, text="Сравнение с SVM",
            command=lambda: self._open_week5_file("cnn_vs_hog_svm.png")
        ).pack(side="left", padx=3)
        ttk.Button(
            controls, text="Устойчивость",
            command=lambda: self._open_week5_file("robustness_plot.png")
        ).pack(side="left", padx=3)
        ttk.Button(
            controls, text="График порога",
            command=self._open_threshold_plot
        ).pack(side="left", padx=3)

        ttk.Label(
            tab, textvariable=self.w5_status, foreground="#174a7e", wraplength=1100
        ).pack(anchor="w", pady=4)
        self.w5_summary = tk.StringVar(value="Результаты Accuracy и Macro F1 появятся после тестирования.")
        ttk.Label(
            tab, textvariable=self.w5_summary, foreground="#245a32", wraplength=1100
        ).pack(anchor="w", pady=4)
        self.w5_log = tk.Text(tab, height=19, wrap="word", font=("Consolas", 10))
        self.w5_log.pack(fill="both", expand=True)

    def _choose_week5_dataset(self):
        """Выбор только исходного датасета (папка с test)."""
        folder = filedialog.askdirectory(
            parent=self, title="Датасет: выберите object_crops, внутри которой есть test"
        )
        if not folder:
            return
        root = Path(folder)
        if (root / "test").is_dir():
            self.w5_dataset_dir.set(str(root))
            self.w5_status.set("Выбран датасет: " + str(root))
        elif all((root / name).is_dir() for name in ("barrel", "box", "sack")):
            # Даже если пользователь нажал верхнюю кнопку, набор из трёх
            # классов нельзя случайно поставить на место исходного датасета.
            self.w5_independent_dir.set(str(root))
            self.w5_status.set("Выбраны независимые фотографии. Путь к датасету сохранён.")
            messagebox.showinfo(
                "Выбраны независимые фотографии",
                "Эта папка содержит barrel, box и sack, но не содержит test.\n"
                "Я указал её в нижнем поле «Отдельные фотографии».\n"
                "Путь к исходному датасету не изменён.",
                parent=self
            )
        else:
            messagebox.showerror(
                "Неверная папка датасета",
                "Для исходного датасета выберите object_crops, в которой есть папка test.\n"
                "Для новых фотографий используйте нижнюю кнопку «Выбрать фото…».",
                parent=self
            )

    def _choose_week5_photos(self):
        """Выбор независимых фотографий без изменения пути датасета."""
        folder = filedialog.askdirectory(
            parent=self, title="Новые фотографии: папка с barrel, box и sack"
        )
        if not folder:
            return
        root = Path(folder)
        if (root / "test").is_dir() and (root / "train").is_dir():
            self.w5_dataset_dir.set(str(root))
            self.w5_status.set("Выбран датасет. Путь к отдельным фотографиям сохранён.")
            messagebox.showinfo(
                "Выбран исходный датасет",
                "Выбрана папка с train и test, поэтому я указал её в верхнем поле.\n"
                "Для независимой проверки выберите общую папку с barrel, box и sack.",
                parent=self
            )
        else:
            self.w5_independent_dir.set(str(root))
            self.w5_status.set("Выбраны отдельные фотографии: " + str(root))

    @staticmethod
    def _entry_field(parent,label,variable,width):
        group=ttk.Frame(parent); group.pack(side="left",padx=(0,10))
        ttk.Label(group,text=label).pack(side="left")
        ttk.Entry(group,textvariable=variable,width=width).pack(side="left",padx=(4,0))

    def _choose(self):
        p=filedialog.askdirectory(title="Выберите object_crops")
        if p: self.data_dir.set(p)

    def _inspect(self):
        try:
            result=inspect(self.data_dir.get(),"outputs")
            for item in self.table.get_children(): self.table.delete(item)
            for c in ("barrel","box","sack"):
                self.table.insert("","end",values=(c,result["split_counts"]["train"][c],result["split_counts"]["valid"][c],result["split_counts"]["test"][c],result["totals_by_class"][c]))
            self.log.insert("end",json.dumps(result,ensure_ascii=False,indent=2)+"\n\nСхема предобработки сохранена: outputs/dataset_pipeline.png\n")
            self.status.set("Проверка завершена; изображения проверены на открытие.")
        except Exception as e: messagebox.showerror("Ошибка проверки",str(e))

    def _train(self):
        self._set_training_busy(True); self.status.set("Извлекаю HOG-признаки и обучаю baseline; повторный запуск будет быстрее из кэша.")
        dataset_root=self.data_dir.get()
        def worker():
            try:
                result=run(dataset_root,"outputs")
                self.after(0,lambda: self._done(result))
            except Exception as e:
                message=str(e)
                self.after(0,lambda msg=message: self._error(msg))
        threading.Thread(target=worker,daemon=True).start()

    def _train_cnn(self):
        try:
            epochs=int(self.epochs.get()); batch_size=int(self.batch_size.get())
            if not 1 <= epochs <= 500 or batch_size < 1:
                raise ValueError("Введите 1–500 эпох и положительный размер пакета.")
        except ValueError as e:
            messagebox.showerror("Проверьте параметры",str(e)); return
        self._set_training_busy(True)
        self.status.set("CNN обучается в отдельном потоке; интерфейс остаётся доступным.")
        self.log.insert("end", "\nЗапуск первоначального обучения CNN с нуля…\n")
        dataset_root=self.data_dir.get()
        def worker():
            try:
                from src.train_cnn import train_model
                result=train_model(dataset_root,"outputs",epochs=epochs,batch_size=batch_size,
                                   progress_callback=lambda row:self.after(0,lambda r=row:self._cnn_epoch(r)))
                self.after(0,lambda res=result:self._cnn_done(res))
            except Exception as e:
                message=str(e)
                self.after(0,lambda msg=message:self._cnn_error(msg))
        threading.Thread(target=worker,daemon=True).start()

    def _cnn_epoch(self,row):
        text=(f"Эпоха {row['epoch']}: train loss={row['train_loss']:.4f}, accuracy={row['train_accuracy']:.3f}; "
              f"valid loss={row['valid_loss']:.4f}, accuracy={row['valid_accuracy']:.3f}\n")
        self.log.insert("end",text); self.log.see("end")
        self.status.set(f"CNN: завершена эпоха {row['epoch']}; train/validation рассчитаны.")

    def _cnn_done(self,result):
        self._set_training_busy(False)
        self.status.set("Первоначальное обучение CNN завершено; модель и история сохранены.")
        self.log.insert("end", "CNN завершена. Модель: outputs/cargo_cnn_week3_initial.pth; "
                        "история: outputs/cnn_training_history.csv / .json; график: outputs/cnn_training_curves.png\n")
        self.log.see("end")
        messagebox.showinfo("Обучение завершено", "Первоначальная CNN обучена.\nМодель, история и графики сохранены в папке outputs.")

    def _cnn_error(self,message):
        self._set_training_busy(False)
        self.status.set("При обучении CNN произошла ошибка.")
        messagebox.showerror("Ошибка обучения CNN",message)

    def _open_cnn_plot(self):
        path=Path("outputs/cnn_training_curves.png").resolve()
        if not path.exists():
            messagebox.showinfo("График пока не создан","Сначала выполните обучение CNN."); return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))
            elif sys.platform == "darwin":
                subprocess.Popen(["open",str(path)])
            else:
                subprocess.Popen(["xdg-open",str(path)])
        except Exception as e:
            messagebox.showerror("Не удалось открыть график",str(e))

    def _done(self,result):
        self._set_training_busy(False); self.status.set("Baseline обучен. Итоговая оценка выполнена на test.")
        self.log.insert("end",json.dumps({k:result[k] for k in ("selected_C_by_validation_macro_f1","test_accuracy","test_macro_precision","test_macro_recall","test_macro_f1","elapsed_seconds","test_support")},ensure_ascii=False,indent=2)+"\n\nМатрица ошибок: outputs/confusion_matrix.png\nМодель: outputs/hog_svm_model.joblib\n")

    def _error(self,e):
        self._set_training_busy(False); self.status.set("При запуске произошла ошибка."); messagebox.showerror("Ошибка обучения",str(e))

    def _set_training_busy(self,busy):
        self._training_busy=busy
        state="disabled" if busy else "!disabled"
        for button in (self.train_btn,self.cnn_train_btn,self.w4_start_btn,self.w5_start_btn):
            if button is not None: button.state([state])

    def _run_week5(self):
        if self._training_busy:
            return
        try:
            batch_size = int(self.w5_batch_size.get())
            robustness_images = int(self.w5_robustness_images.get())
            if batch_size < 1 or robustness_images < 3:
                raise ValueError("Размер пакета должен быть больше 0; для устойчивости нужно минимум 3 изображения.")
            dataset_root = Path(self.w5_dataset_dir.get().strip())
            if not (dataset_root / "test").is_dir():
                raise FileNotFoundError("Выберите папку object_crops, в которой находится папка test.")
            from src.evaluate_cnn import find_final_model
            model_path = find_final_model()
            independent_folder = self.w5_independent_dir.get().strip() or None
            if independent_folder and not Path(independent_folder).is_dir():
                raise FileNotFoundError("Не найдена папка отдельных фотографий.")
        except Exception as exc:
            messagebox.showerror("Неделя 5: проверьте настройки", str(exc))
            return

        self._set_training_busy(True)
        self.w5_log.delete("1.0", "end")
        self.w5_summary.set("Идёт тестирование. CNN не обучается.")
        self.w5_status.set("Проверка модели на Test…")
        self.w5_log.insert(
            "end", f"Датасет: {dataset_root}\nМодель: {model_path}\n"
            f"Размер пакета: {batch_size}; для устойчивости: {robustness_images}\n"
            f"Новые фото: {independent_folder or 'не выбраны'}\n\n"
        )

        def progress(message):
            self.after(0, lambda value=str(message): self._week5_progress(value))

        def worker():
            try:
                from src.evaluate_cnn import evaluate_cnn
                from src.robustness import run_additional_checks

                result = evaluate_cnn(
                    dataset_root=str(dataset_root),
                    batch_size=batch_size,
                    model_path=model_path,
                    progress_callback=progress
                )
                progress("Полная проверка Test завершена. Проверяю устойчивость…")
                extra = run_additional_checks(
                    dataset_root=str(dataset_root),
                    independent_folder=independent_folder,
                    batch_size=batch_size,
                    robustness_images=robustness_images,
                    model_path=model_path,
                    output_dir=result["output_directory"],
                    progress_callback=progress
                )
                self.after(0, lambda a=result, b=extra: self._week5_done(a, b))
            except Exception as exc:
                message = str(exc)
                self.after(0, lambda error=message: self._week5_error(error))

        threading.Thread(target=worker, daemon=True).start()

    def _week5_progress(self, message):
        self.w5_status.set(message)
        # Keep the log readable; progress is refreshed in the status label.
        if message.startswith(("Проверено ", "Без изменений", "Размытие", "Затемнение",
                               "Осветление", "Поворот", "Отдельные фотографии")):
            return
        self.w5_log.insert("end", message + "\n")
        self.w5_log.see("end")

    def _week5_done(self, result, extra):
        self._set_training_busy(False)
        self.w5_last_dir = Path(result["output_directory"])
        m = result["metrics"]
        self.w5_status.set("Неделя 5: проверка завершена без обучения.")
        self.w5_summary.set(
            f"Test: {result['test_images']} изображений; "
            f"Accuracy: {m['accuracy']*100:.2f}%; "
            f"Macro Precision: {m['macro_precision']:.4f}; "
            f"Macro Recall: {m['macro_recall']:.4f}; "
            f"Macro F1: {m['macro_f1']:.4f}."
        )
        self.w5_log.insert(
            "end",
            f"\nИТОГИ TEST\n"
            f"Accuracy: {m['accuracy']:.4f}\n"
            f"Macro Precision: {m['macro_precision']:.4f}\n"
            f"Macro Recall: {m['macro_recall']:.4f}\n"
            f"Macro F1: {m['macro_f1']:.4f}\n"
        )
        comparison = result.get("baseline_comparison", {})
        if comparison.get("status") == "completed":
            self.w5_log.insert("end", "Сравнение CNN с HOG + SVM сохранено.\n")
        else:
            self.w5_log.insert("end", f"Сравнение с HOG + SVM: {comparison.get('message', 'нет данных')}\n")
        conf = result.get("confidence_analysis", {})
        threshold = float(conf.get("threshold", 0.70))
        accuracy = conf.get("accepted_accuracy")
        accuracy_text = (
            f"{accuracy * 100:.2f}%"
            if accuracy is not None
            else "нет принятых ответов"
        )
        self.w5_log.insert(
            "end",
            f"Порог {threshold:.0%} (подобран на Validation): "
            f"на ручную проверку {conf.get('manual_review_images', '?')} "
            f"изображений, выявлено ошибок "
            f"{conf.get('manual_review_errors', '?')}; "
            f"точность автоматических ответов: {accuracy_text}.\n"
        )
        robust = extra.get("robustness", {})
        self.w5_log.insert(
            "end", f"Проверка устойчивости: {robust.get('images', 0)} изображений; результаты сохранены.\n"
        )
        independent = extra.get("independent_photos", {})
        if independent.get("status") == "completed":
            self.w5_log.insert(
                "end", f"Отдельные фотографии: {independent.get('images', 0)}.\n"
            )
            labeled = independent.get("labeled_images", 0)
            accuracy = independent.get("accuracy_labeled")
            if accuracy is not None:
                self.w5_log.insert(
                    "end",
                    f"Из них с известным классом: {labeled}; "
                    f"правильно распознано: {round(accuracy * labeled)}/{labeled}; "
                    f"Accuracy: {accuracy * 100:.2f}%.\n"
                )
        else:
            self.w5_log.insert("end", "Отдельные фотографии не проверялись (папка не выбрана).\n")
        self.w5_log.insert("end", f"\nРезультаты сохранены: {self.w5_last_dir}\n")
        self.w5_log.see("end")
        messagebox.showinfo("Неделя 5", "Тестирование завершено. Результаты сохранены в outputs/week5.")

    def _week5_error(self, message):
        self._set_training_busy(False)
        self.w5_status.set("Ошибка при проверке недели 5.")
        self.w5_log.insert("end", "\nОШИБКА: " + message + "\n")
        self.w5_log.see("end")
        messagebox.showerror("Ошибка недели 5", message)

    def _open_threshold_plot(self):
        """Показать график последнего подбора порога на Validation."""
        root = Path(__file__).resolve().parent / "outputs" / "week5"
        candidates = sorted(
            root.glob("threshold_*/threshold_comparison.png"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            messagebox.showinfo(
                "Подбор порога",
                "График пока не найден. Сначала запустите "
                "python -m src.threshold_analysis"
            )
            return
        self._open_path(candidates[0])

    def _open_week5_folder(self):
        if self.w5_last_dir is None or not self.w5_last_dir.is_dir():
            messagebox.showinfo("Неделя 5", "Сначала завершите тестирование.")
            return
        self._open_path(self.w5_last_dir)

    def _open_week5_file(self, filename):
        if self.w5_last_dir is None:
            messagebox.showinfo("Неделя 5", "Сначала завершите тестирование.")
            return
        path = self.w5_last_dir / filename
        if not path.is_file():
            messagebox.showinfo("Файл не найден", "Такой результат не создан: " + filename)
            return
        self._open_path(path)

    @staticmethod
    def _parse_numbers(value,cast):
        result=[cast(item.strip()) for item in value.split(",") if item.strip()]
        if not result: raise ValueError("Заполните список значений через запятую.")
        return tuple(result)

    def _parse_filter_profiles(self):
        profiles=[]
        for profile in self.w4_filters.get().split(";"):
            values=tuple(int(value.strip()) for value in profile.split(",") if value.strip())
            if values: profiles.append(values)
        if not profiles: raise ValueError("Укажите профили фильтров через ;, например 32,64,128,256")
        return tuple(profiles)

    def _run_week4(self):
        if self._training_busy: return
        try:
            lrs=self._parse_numbers(self.w4_lrs.get(),float)
            batches=self._parse_numbers(self.w4_batches.get(),int)
            dropouts=self._parse_numbers(self.w4_dropouts.get(),float)
            filters=self._parse_filter_profiles()
            max_epochs=int(self.w4_max_epochs.get()); patience=int(self.w4_patience.get())
            min_delta=float(self.w4_min_delta.get()); seed=int(self.w4_seed.get())
            from src.optimize_cnn import build_experiment_configs
            configs=build_experiment_configs(lrs,batches,dropouts,filters)
            if not 1 <= max_epochs <= 500 or patience < 1 or min_delta < 0:
                raise ValueError("Задайте max epochs 1–500, patience ≥ 1 и min delta ≥ 0.")
        except Exception as e:
            messagebox.showerror("Проверьте параметры недели 4",str(e)); return

        dataset_root=self.data_dir.get()
        auto_reduce_batch=self.w4_auto_batch.get()
        self._set_training_busy(True)
        for item in self.w4_table.get_children(): self.w4_table.delete(item)
        self.w4_log.delete("1.0","end")
        self.w4_best.set("Подбор конфигурации выполняется; окончательный выбор появится после завершения всех запусков.")
        self.w4_status.set(f"Запускается {len(configs)} последовательных экспериментов на одном Train/Validation.")
        self.w4_log.insert("end",f"Датасет: {dataset_root}\nSeed: {seed}; max epochs: {max_epochs}; patience: {patience}; min_delta: {min_delta}\n")

        def progress(event):
            self.after(0,lambda e=dict(event):self._week4_progress(e))
        def status(event):
            self.after(0,lambda e=dict(event):self._week4_event(e))
        def worker():
            try:
                from src.optimize_cnn import run_experiments
                result=run_experiments(dataset_root,"outputs/week4",learning_rates=lrs,batch_sizes=batches,
                                       dropouts=dropouts,filter_sets=filters,max_epochs=max_epochs,
                                       patience=patience,min_delta=min_delta,seed=seed,
                                       auto_reduce_batch=auto_reduce_batch,
                                       progress_callback=progress,status_callback=status)
                self.after(0,lambda r=result:self._week4_done(r))
            except Exception as e:
                message=str(e)
                self.after(0,lambda m=message:self._week4_error(m))
        threading.Thread(target=worker,daemon=True).start()

    def _week4_progress(self,event):
        line=(f"{event['experiment_index']}/{event['experiment_total']} {event['experiment']} | "
              f"epoch {event['epoch']}: train loss {event['train_loss']:.4f}, acc {event['train_accuracy']:.3f}; "
              f"val loss {event['valid_loss']:.4f}, acc {event['valid_accuracy']:.3f}; best epoch {event['best_epoch']}\n")
        self.w4_log.insert("end",line); self.w4_log.see("end")
        self.w4_status.set(f"Эксперимент {event['experiment_index']}/{event['experiment_total']}: {event['experiment']}, эпоха {event['epoch']}")

    def _week4_event(self,event):
        state=event.get("status"); name=event.get("experiment","")
        if state=="starting":
            self.w4_log.insert("end",f"\nСтарт {event['experiment_index']}/{event['experiment_total']}: {name}\n")
        elif state=="cuda_oom_retry":
            self.w4_log.insert("end",f"CUDA OOM: {name}; повтор с batch size={event['batch_size']}\n")
        elif state in {"completed","failed"} and "result" in event:
            r=event["result"]
            values=(r.get("name",name),state,r.get("best_epoch") or "—",
                    f"{r['best_validation_loss']:.4f}" if r.get("best_validation_loss") is not None else "—",
                    f"{r['best_validation_accuracy']:.3f}" if r.get("best_validation_accuracy") is not None else "—",
                    r.get("stop_reason",r.get("error","—")))
            self.w4_table.insert("","end",values=values)
            if state=="failed": self.w4_log.insert("end",f"ОШИБКА {name}: {r.get('error','неизвестная ошибка')}\n")
            else: self.w4_log.insert("end",f"Завершён {name}: epochs={r['epochs_completed']}, best epoch={r['best_epoch']}, stop={r['stop_reason']}\n")
        elif state=="finished":
            result=event["result"]
            self.week4_last_dir=Path(result["run_directory"])
            self.w4_status.set(f"Завершено: успешно {result['experiments_completed']}, с ошибкой {result['experiments_failed']}. Результаты: {self.week4_last_dir}")
        self.w4_log.see("end")

    def _week4_done(self,result):
        self._set_training_busy(False)
        self.week4_last_dir=Path(result["run_directory"])
        successful=[r for r in result["results"] if r.get("status")=="completed"]
        if successful:
            best=min(successful,key=lambda r:(r["best_validation_loss"],-r["best_validation_accuracy"]))
            self.w4_best.set(f"Лучшая конфигурация по минимальной best Validation Loss (при равенстве — Accuracy): {best['name']}; "
                             f"LR={best['learning_rate']}, batch={best['effective_batch_size']}, dropout={best['dropout']}, "
                             f"filters={best['filters']}; эпоха {best['best_epoch']}; Val Loss={best['best_validation_loss']:.4f}; "
                             f"Val Accuracy={best['best_validation_accuracy']:.3f}. Модель: {self.week4_last_dir / 'final_model.pth'}")
        else:
            self.w4_best.set("Не удалось завершить ни одного эксперимента; журнал ошибок сохранён в каталоге запуска.")
        self.w4_status.set(f"Оптимизация завершена: {result['experiments_completed']} успешно, {result['experiments_failed']} ошибок. Каталог: {self.week4_last_dir}")
        messagebox.showinfo("Неделя 4 завершена",self.w4_status.get())

    def _week4_error(self,message):
        self._set_training_busy(False)
        self.w4_status.set("Оптимизация прервана до завершения серии.")
        self.w4_log.insert("end",f"Критическая ошибка: {message}\n")
        messagebox.showerror("Ошибка оптимизации",message)

    def _latest_week4_dir(self):
        if self.week4_last_dir and self.week4_last_dir.is_dir(): return self.week4_last_dir
        pointer=Path("outputs/week4/latest_run.json")
        if not pointer.is_file(): return None
        try: return Path("outputs/week4")/json.loads(pointer.read_text(encoding="utf-8"))["relative_path"]
        except Exception: return None

    def _open_week4_artifact(self,filename):
        folder=self._latest_week4_dir()
        path=folder/filename if folder else None
        if not path or not path.is_file():
            messagebox.showinfo("Файл пока не создан","Сначала запустите сравнение конфигураций."); return
        self._open_path(path)

    def _open_week4_folder(self):
        folder=self._latest_week4_dir()
        if not folder or not folder.is_dir():
            messagebox.showinfo("Каталог пока не создан","Сначала запустите сравнение конфигураций."); return
        self._open_path(folder)

    @staticmethod
    def _open_path(path):
        path=Path(path).resolve()
        try:
            if sys.platform.startswith("win"): os.startfile(str(path))
            elif sys.platform=="darwin": subprocess.Popen(["open",str(path)])
            else: subprocess.Popen(["xdg-open",str(path)])
        except Exception as e: messagebox.showerror("Не удалось открыть файл",str(e))


if __name__ == "__main__":
    CargoApp().mainloop()
