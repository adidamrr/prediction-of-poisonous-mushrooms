# Mushroom Safety Screener

Pet project по мотивам Kaggle-конкурса [Binary Prediction of Poisonous Mushrooms](https://www.kaggle.com/competitions/playground-series-s4e8/overview).

Идея проекта: не просто обучить модель, а собрать end-to-end решение:
- обучение модели на табличных данных;
- inference через API;
- UI для ручного ввода признаков;
- feedback loop через Postgres;
- дообучение модели на пользовательском feedback.

## Результат на Kaggle
- Public Score: `0.96134`
- Private Score: `0.96142`

## Что используется
- `Python`
- `CatBoost`
- `pandas`, `numpy`, `scikit-learn`
- `FastAPI`
- `Streamlit`
- `PostgreSQL`
- `Docker Compose`

## Структура проекта
- [src/eda_train.ipynb](/Users/grigory/Code/pet-projects/3/src/eda_train.ipynb) — EDA и эксперименты
- [train.py](/Users/grigory/Code/pet-projects/3/train.py) — базовое обучение модели
- [predict.py](/Users/grigory/Code/pet-projects/3/predict.py) — preprocessing и inference
- [app.py](/Users/grigory/Code/pet-projects/3/app.py) — FastAPI backend
- [streamlit_app.py](/Users/grigory/Code/pet-projects/3/streamlit_app.py) — интерфейс
- [db.py](/Users/grigory/Code/pet-projects/3/db.py) — подключение к Postgres и ORM-модели
- [feedback_dataset.py](/Users/grigory/Code/pet-projects/3/feedback_dataset.py) — выгрузка feedback-данных из БД
- [finetune.py](/Users/grigory/Code/pet-projects/3/finetune.py) — дообучение текущей модели на feedback
- [clear_feedback.py](/Users/grigory/Code/pet-projects/3/clear_feedback.py) — очистка feedback-таблиц
- [docker-compose.yml](/Users/grigory/Code/pet-projects/3/docker-compose.yml) — запуск сервисов в Docker

## Данные
Папка `data/` не включена в репозиторий из-за размера файлов.

Для запуска [train.py](/Users/grigory/Code/pet-projects/3/train.py) нужно отдельно скачать данные конкурса с Kaggle: [playground-series-s4e8/data](https://www.kaggle.com/competitions/playground-series-s4e8/data)

После скачивания файлы нужно положить в папку `data/`.

## Что делает сервис
Пользователь заполняет признаки гриба в UI.  
Frontend отправляет их в backend.  
Backend:
- загружает модель;
- считает prediction;
- сохраняет prediction в Postgres;
- возвращает результат в UI.

Если случай пограничный, интерфейс показывает feedback-кнопки:
- `Скорее верно`
- `Скорее неверно`

Feedback сохраняется в БД и потом используется для дообучения модели.

## Запуск через Docker
Это основной способ запуска проекта.

### 1. Обучить базовую модель
Для начала необходимо обучить модель:

```bash
docker compose run --rm backend python train.py
```

После этого в папке `model/` появятся:
- `model.cbm`
- `meta.json`

### 2. Поднять сервис
```bash
docker compose up --build
```

После запуска:
- frontend: [http://localhost:8501](http://localhost:8501)
- backend docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Остановить сервис
```bash
docker compose down
```

## Локальный запуск без Docker
Если хочется запускать вручную, нужен `.env` в корне проекта.

Файл `.env` добавлен в `.gitignore`, поэтому его нужно создать вручную.

```env
DATABASE_URL=postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/mushroom_feedback
```

Потом:

```bash
uv run uvicorn app:app --reload
uv run streamlit run streamlit_app.py
```

Для запуска через Docker `.env` не нужен: настройки базы уже заданы в `docker-compose.yml`.

## Дообучение модели
После накопления feedback можно дообучить текущую модель:

```bash
docker compose run --rm backend python finetune.py
```

Что делает `finetune.py`:
- забирает feedback из Postgres;
- восстанавливает таргет по `correct/incorrect`;
- загружает текущую модель `model/model.cbm`;
- дообучает её малым числом итераций;
- сохраняет новую модель в `model/finetuned_model.cbm`;
- очищает таблицы `predictions` и `prediction_feedback`.

## Полезные команды
### Обучение базовой модели
```bash
docker compose run --rm backend python train.py
```

### Дообучение по feedback
```bash
docker compose run --rm backend python finetune.py
```

### Ручная очистка feedback-таблиц
```bash
docker compose run --rm backend python clear_feedback.py
```

### Запуск всего проекта
```bash
docker compose up --build
```

## Что хранится в базе
Таблица `predictions`:
- входные признаки пользователя;
- score модели;
- итоговый класс;
- threshold;
- флаг пограничного случая.

Таблица `prediction_feedback`:
- ссылка на prediction;
- отзыв пользователя `correct` / `incorrect`.
