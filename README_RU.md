# Cargo CNN — недели 2–4

Проект классифицирует изображения трёх типов груза: `barrel` (бочка), `box` (коробка) и `sack` (мешок). Неделя 2 включает проверку данных и HOG + Linear SVM baseline. Неделя 3 — самостоятельно реализованную CargoCNN на PyTorch. Неделя 4 — последовательные эксперименты, Early Stopping и выбор конфигурации по Validation. Предобученные веса не используются.

## Состав

- `app.py` — GUI с вкладками: проверка данных, HOG + SVM, первоначальная CNN и оптимизация CNN.
- `src/inspect_dataset.py` — проверка файлов, подсчёт по классам и визуализация preprocessing.
- `src/torch_data.py` — PyTorch Dataset/DataLoader: RGB, 224×224, нормализация; аугментация включается только в train.
- `src/hog_svm.py` — HOG + LinearSVC и метрики baseline.
- `src/cnn_model.py` — CNN из четырёх свёрточных блоков; week-3 конфигурация остаётся значением по умолчанию, параметры week 4 позволяют менять фильтры и Dropout.
- `src/train_cnn.py` — первоначальный цикл train/validation недели 3.
- `src/optimize_cnn.py` — гиперпараметрические эксперименты, Early Stopping, checkpoint лучших эпох, сводные CSV/JSON и графики.
- `outputs/`, `outputs_backup_week3_2epochs/` — сохранённые результаты и веса предыдущих недель из переданного архива.
- Новые результаты оптимизации записываются отдельно в `outputs/week4/runs/`; веса и результаты недель 2–3 не перезаписываются.

Датасет в ZIP не включён. Выберите локальный каталог `object_crops` со структурой:

```text
object_crops/
  train/{barrel,box,sack}/
  valid/{barrel,box,sack}/
  test/{barrel,box,sack}/
```

Week 4 использует только `train` и `valid`. `test` в подборе параметров не участвует.

## Запуск в VS Code на Windows

1. Распакуйте ZIP и откройте папку проекта через **File → Open Folder…**.
2. В VS Code откройте **Terminal → New Terminal**. Если у вас есть существующее окружение `.venv` с нужным PyTorch/CUDA, выберите его через **Ctrl+Shift+P → Python: Select Interpreter** и не переустанавливайте PyTorch. Сам каталог `.venv` в ZIP не нужен.
3. Если окружения нет, создайте проектное и установите зависимости:

   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   pip install -r requirements-torch.txt
   ```

   Не устанавливайте зависимости в глобальный Python. Для CUDA-сборки PyTorch выберите команду под Windows, версию Python и CUDA через [официальный селектор PyTorch](https://pytorch.org/get-started/locally/). Перед обучением проверьте выбранный интерпретатор:

   ```powershell
   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
   ```

4. Запустите GUI из корня проекта:

   ```powershell
   python app.py
   ```

5. На первой вкладке нажмите **Выбрать папку…** и укажите ваш `...\prepared_cargo_cnn\object_crops`. Та же выбранная папка применяется и в Week 4. При необходимости сначала выполните проверку датасета.

## Week 4: настройка и запуск экспериментов

Откройте вкладку **Неделя 4 — оптимизация CNN**. В форме можно изменить Learning Rate, Batch Size, Dropout, профили фильтров, максимальное число эпох, Patience, Min Delta и Seed. Learning Rate, Batch Size и Dropout вводятся списками через запятую; каждый профиль фильтров содержит четыре числа, а профили разделяются `;`.

Первое значение каждого поля образует базовую конфигурацию; остальные варианты меняют по одному параметру относительно неё. Значения по умолчанию создают пять сравнений:

| Конфигурация | Learning Rate | Batch Size | Dropout | Фильтры |
|---|---:|---:|---:|---|
| Baseline | 0.001 | 8 | 0.0 | 32, 64, 128, 256 |
| LR | 0.0005 | 8 | 0.0 | 32, 64, 128, 256 |
| Batch | 0.001 | 16 | 0.0 | 32, 64, 128, 256 |
| Dropout | 0.001 | 8 | 0.3 | 32, 64, 128, 256 |
| Filters | 0.001 | 8 | 0.0 | 16, 32, 64, 128 |

Запуски выполняются последовательно, по одному; интерфейс остаётся отзывчивым. Если CUDA сообщает об OOM и отмечен флажок уменьшения batch, неудачная попытка сохраняется, Batch Size уменьшается вдвое, а конфигурация перезапускается с тем же Seed. В отчёте фиксируются исходный и фактический batch и число повторов.

Early Stopping следит за Validation Loss. Лучшая эпоха сохраняется по минимальной Validation Loss; остановка происходит после заданного количества эпох без улучшения более чем на Min Delta. Конфигурация-победитель выбирается по наименьшей лучшей Validation Loss; при равенстве — по более высокой Validation Accuracy. Test не используется для выбора.

Вкладка показывает текущую конфигурацию и эпоху, промежуточные Loss/Accuracy, ошибки и итоговый выбор. Кнопки открывают сводную таблицу, график сравнения и папку результатов.

### Файлы результатов Week 4

Каждый запуск получает уникальную папку `outputs/week4/runs/run_.../`. В ней создаются:

- `experiment_results.csv` и `experiment_results.json` — конфигурации, эпохи, лучшие метрики, причина остановки, OOM-повторы и ошибки;
- `comparison.png` — сравнение лучших Validation Loss и Accuracy;
- папка каждого эксперимента с `history.csv`, `history.json`, `experiment.json`, `training_curves.png` и `attempt_XX/best_model.pth`;
- `best_configuration.json` и `final_model.pth` — выбранные конфигурация и checkpoint с архитектурой, параметрами и состоянием весов.

Путь к последнему запуску находится в `outputs/week4/latest_run.json`. Успешные запуски сохраняются, даже если следующая конфигурация завершается ошибкой.

CLI доступен как альтернатива GUI:

```powershell
python -m src.optimize_cnn --data "D:\4kurs\7sem\ES\Dataset\F2\Cargo_CNN_full_crops\prepared_cargo_cnn\object_crops" --max-epochs 20 --patience 4 --min-delta 0.001
```

Проверка программного цикла на временных синтетических изображениях:

```powershell
python -m src.optimize_cnn --smoke-test
```

Это технический тест кода, не эксперимент с реальным датасетом и не показатель качества модели.

## Сохранённые фактические результаты Week 3

Основная история `outputs/cnn_training_history.json` содержит 10 фактических эпох на 26 412 изображениях Train и 5 319 Validation, LR 0.001 и Batch Size 8. Лучший Validation Accuracy — 0.9581 на эпохе 9; на эпохе 10 Train Accuracy — 0.9512, Validation Accuracy — 0.9466. В архиве также сохранены исходный checkpoint, графики, отдельный синтетический smoke-test и резервная папка запуска на 2 эпохи. Эти результаты не являются результатами Week 4.
