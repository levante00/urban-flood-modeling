# Моделирование уровня воды (Urban Flood Modeling)

> Автор: **Асатрян Левон Ваганович**

Этот репозиторий $-$ оформленная в индустриальном виде задача Kaggle
**[UrbanFloodBench: Flood Modelling](https://www.kaggle.com/competitions/urban-flood-modelling/)**

Проект решает задачу прогноза уровня воды в городской дренажной сети (UrbanFloodBench / Kaggle Flood
Modelling). Репозиторий оформлен как воспроизводимый MLOps-пайплайн: данные через DVC, конфиги через
Hydra, обучение/инференс через PyTorch Lightning, трекинг через MLflow.

## 1) Смысловое содержание проекта

### 1.1 Цель

Предсказать следующий уровень воды (`water_level`) для узлов городской гидросети по временным рядам:

- динамика по 1D-узлам (каналы/трубопроводы);
- динамика по 2D-узлам (поверхностные зоны);
- топология графа (рёбра 1D и 2D) для учета соседей;
- тип узла (1D/2D) как дополнительный сигнал.

### 1.2 Подход

Модель в [urban_flood_modeling/model/model.py](urban_flood_modeling/model/model.py):

- LSTM по последовательности признаков длины `seq_len`;
- эмбеддинг типа узла (`node_type`);
- MLP-голова для предсказания остатка.

На этапе предсказания остаток добавляется к последнему наблюдаемому уровню воды и сглаживается
коэффициентами из postprocessing-конфига.

### 1.3 Фичи и препроцессинг

Для каждого узла формируются признаки на окне длины `seq_len`:

1. `water_level`
2. `rainfall` (если столбца нет — заполняется нулями)
3. среднее `water_level` в окне
4. стандартное отклонение `water_level` в окне
5. средний уровень воды соседних узлов по графу

Стандартизация колонок выполняется в
[urban_flood_modeling/utils/preprocessing.py](urban_flood_modeling/utils/preprocessing.py):

- `node_idx`/`node_id` → `node_id`
- time-поля → `timestep`
- water level-поля → `water_level`
- rain-поля → `rainfall`

### 1.4 Используемые данные (датасеты)

Проект использует табличные CSV-данные из [data](data), которые делятся на 4 группы:

1. **Динамика узлов (основные временные ряды)**
   - train: [data/1d_nodes_dynamic_all.csv](data/1d_nodes_dynamic_all.csv),
     [data/2d_nodes_dynamic_all.csv](data/2d_nodes_dynamic_all.csv)
   - test: [data/test_1d_nodes_dynamic_all.csv](data/test_1d_nodes_dynamic_all.csv),
     [data/test_2d_nodes_dynamic_all.csv](data/test_2d_nodes_dynamic_all.csv)
   - содержат `timestep`, `node_idx/node_id`, `water_level` (+ опционально `rainfall` и другие
     служебные поля)

2. **Топология графа (связи между узлами)**
   - [data/1d_edge_index.csv](data/1d_edge_index.csv),
     [data/2d_edge_index.csv](data/2d_edge_index.csv),
     [data/1d2d_connections.csv](data/1d2d_connections.csv)

3. **Статические и узловые-признаки**
   - [data/1d_nodes_static.csv](data/1d_nodes_static.csv),
     [data/2d_nodes_static.csv](data/2d_nodes_static.csv)
   - [data/1d_edges_static.csv](data/1d_edges_static.csv),
     [data/2d_edges_static.csv](data/2d_edges_static.csv)
   - [data/1d_edges_dynamic_all.csv](data/1d_edges_dynamic_all.csv),
     [data/2d_edges_dynamic_all.csv](data/2d_edges_dynamic_all.csv)

4. **Вспомогательные файлы соревнования**
   - [data/timesteps.csv](data/timesteps.csv), [data/test_timesteps.csv](data/test_timesteps.csv)
   - [data/sample_submission.csv](data/sample_submission.csv)

Размер данных:

- общий размер каталога [data](data): **~1.4 GB**
- крупнейшие файлы:
  - [data/sample_submission.csv](data/sample_submission.csv) — **~1.1 GB**, **50,910,193** строк
    (включая заголовок)
  - [data/test_2d_edges_dynamic_all.csv](data/test_2d_edges_dynamic_all.csv) — **~37 MB**,
    **3,531,076** строк
  - [data/test_2d_nodes_dynamic_all.csv](data/test_2d_nodes_dynamic_all.csv) — **~39 MB**,
    **1,653,621** строк
  - [data/2d_edges_dynamic_all.csv](data/2d_edges_dynamic_all.csv) — **~22 MB**, **745,891** строк
  - [data/2d_nodes_dynamic_all.csv](data/2d_nodes_dynamic_all.csv) — **~18 MB**, **349,305** строк

### 1.5 Что делает пайплайн

- Подтягивает данные через DVC (в `prepare_data`).
- Собирает датасеты для train/predict.
- Обучает модель и сохраняет checkpoint.
- Запускает инференс на тестовых данных.
- Логирует метрики в MLflow + сохраняет локальные артефакты графиков/метрик.

### 1.6 Метрики

На этапе обучения считаются и логируются:

- `train_loss` = MSE (Mean Squared Error)
- `train_rmse` = $\sqrt{\text{MSE}}$
- `train_mae` = MAE (Mean Absolute Error)

**Основная метрика проекта — `train_loss` (MSE)**, так как именно она используется как функция
потерь и минимизируется оптимизатором.

`train_rmse` и `train_mae` используются как дополнительные интерпретируемые метрики качества.

На этапе инференса (когда нет ground truth) сохраняются агрегаты распределения предсказаний:

- `pred_count`, `pred_mean`, `pred_std`, `pred_min`, `pred_max`

Это диагностические метрики, а не целевая метрика качества модели.

---

## 2) Техническая инструкция

## Setup

### 2.1 Требования

- Linux (проект тестировался на Linux)
- Python version >= 3.12
- `uv` для установки зависимостей
- Git + доступ в интернет для `dvc pull`

### 2.2 Клонирование репозитория

```bash
git clone git@github.com:levante00/urban-flood-modeling.git
cd urban-flood-modeling
```

### 2.3 Установка окружения

В корне проекта:

```bash
pip install uv
uv venv
uv sync --dev
source .venv/bin/activate
uv run pre-commit install
uv run pre-commit run --all-files
```

Зависимости и их версии фиксируются в:

- [pyproject.toml](pyproject.toml)
- [uv.lock](uv.lock)

### 2.4 Загрузка данных

Данные хранятся через DVC и подтягиваются автоматически в train/infer. Удаленно хранятся в amazon s3
хранилище. При необходимости можно скачать вручную заранее:

```bash
uv run dvc pull
```

### 2.5 MLflow (опционально, но рекомендуется)

По умолчанию конфиги train/infer ожидают MLflow на `127.0.0.1:8081`.

```bash
uv run mlflow server \
  --host 127.0.0.1 \
  --port 8081
```

Если MLflow не нужен, отключите логгер override-параметром:

```bash
uv run python main.py train logging.enabled=false
```

## Train

Обучение запускается через единый CLI ([main.py](main.py)).

### 3.1 Базовый запуск

```bash
uv run python main.py train
```

### 3.2 Запуск с override-параметрами Hydra

Примеры:

```bash
uv run python main.py train training.epochs=20 data.batch_size=512
uv run python main.py train training.trainer.accelerator=gpu training.trainer.devices=1
uv run python main.py train logging.enabled=false
```

### 3.3 Что происходит по шагам

1. Подтягиваются DVC-данные.
2. Выполняется препроцессинг и сбор train-датасета.
3. Обучается `FloodLightningModule`.
4. Сохраняется checkpoint: `artifacts/flood_model.ckpt`.
5. Сохраняются локальные train-метрики и графики:
   - `plots/mlflow/train/<run_id>/metrics.csv`
   - `plots/mlflow/train/<run_id>/train_loss.png`
   - `plots/mlflow/train/<run_id>/train_rmse.png`
   - `plots/mlflow/train/<run_id>/train_mae.png`

## Production preparation

Ниже — практический чек-лист подготовки модели к развёртыванию.

### 4.1 Зафиксировать артефакты поставки

Минимальный набор:

1. checkpoint: `artifacts/flood_model.ckpt`
2. конфиги инференса:
   - [configs/infer.yaml](configs/infer.yaml)
   - [configs/data/default.yaml](configs/data/default.yaml)
   - [configs/preprocessing/default.yaml](configs/preprocessing/default.yaml)
   - [configs/postprocessing/default.yaml](configs/postprocessing/default.yaml)
3. код инференса:
   - [urban_flood_modeling/infer.py](urban_flood_modeling/infer.py)
   - [urban_flood_modeling/modules](urban_flood_modeling/modules)
   - [urban_flood_modeling/model](urban_flood_modeling/model)
4. lock-файл окружения: [uv.lock](uv.lock)

### 4.2 Минимизация зависимостей для inference

Код инференса отделён от train-процедуры (см.
[urban_flood_modeling/infer.py](urban_flood_modeling/infer.py)), что упрощает сборку отдельного
runtime-образа. Для production-сценария рекомендуется отдельный env/образ с минимальным набором
пакетов (без dev-инструментов).

## Infer

Инференс также запускается через единый CLI.

### 5.1 Формат входных данных

Минимально необходимые CSV-файлы задаются в [configs/data/default.yaml](configs/data/default.yaml):

- динамика train/test для 1D/2D (`*_nodes_dynamic_all.csv`)
- графовые связи (`1d_edge_index.csv`, `2d_edge_index.csv`)

#### Ожидаемые колонки (динамика узлов)

- обязательные: `timestep`, `node_idx` (или `node_id`), `water_level`
- опциональные: `rainfall` (если нет — подставляется 0)

Примеры реальных файлов:

- [data/test_1d_nodes_dynamic_all.csv](data/test_1d_nodes_dynamic_all.csv)
- [data/test_2d_nodes_dynamic_all.csv](data/test_2d_nodes_dynamic_all.csv)

### 5.2 Базовый запуск инференса

```bash
uv run python main.py infer
```

### 5.3 Полезные override-примеры

```bash
uv run python main.py infer inference.pred_node_type=2
uv run python main.py infer paths.checkpoint_path=artifacts/flood_model.ckpt
uv run python main.py infer paths.predictions_dir=artifacts paths.predictions_csv_name=my_predictions.csv
uv run python main.py infer logging.enabled=false
```

### 5.4 Что сохраняется после инференса

- в консоль: количество обработанных батчей
- файл с предсказаниями в CSV (формат: `node_id,prediction`)
  - по умолчанию: `artifacts/predictions.csv`
  - путь задается через [configs/infer.yaml](configs/infer.yaml) (`paths.predictions_dir`,
    `paths.predictions_csv_name`)
- локально: агрегированные метрики предсказаний
  - `plots/mlflow/infer/<run_id>/final_metrics.csv`

---

## Быстрый старт (TL;DR)

```bash
uv sync --dev
source .venv/bin/activate
uv run dvc pull
uv run python main.py train
uv run python main.py infer
```
