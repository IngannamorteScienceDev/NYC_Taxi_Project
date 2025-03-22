"""
Главный скрипт для проекта прогнозирования спроса на такси.
Он объединяет все этапы:
  1. Загрузка данных и первичный анализ (EDA)
  2. Предобработка данных
  3. Агрегация данных по дням
  4. Разделение данных на train/test для оценки модели
  5. Обучение модели Prophet и прогнозирование
  6. Оценка прогноза на тестовой выборке
  7. Обучение модели на полном датасете и прогнозирование будущего
"""

import pandas as pd

# Импортируем функции из модулей в папке src
from src.eda import run_eda, load_all_data  # load_all_data объединяет файлы за 2024 и январь 2025
from src.preprocessing import preprocess_data
from src.aggregation import aggregate_daily
from src.modeling import (
    train_prophet_model,
    make_forecast,
    plot_forecast,
    plot_components
)
from src.evaluation import (
    evaluate_forecast,
    plot_actual_vs_forecast,
    train_test_split_prophet
)


def main():
    # 1. Загрузка данных: объединяем все файлы за 2024 год и файл за январь 2025
    data_dir = "data"
    df_raw = load_all_data(data_dir)
    print("Исходные данные загружены. Всего строк:", len(df_raw))

    # Применяем дополнительную фильтрацию, чтобы оставить только данные от 2024-01-01 до 2025-02-01
    df_raw = df_raw[(df_raw['tpep_pickup_datetime'] >= pd.Timestamp('2024-01-01')) &
                    (df_raw['tpep_pickup_datetime'] < pd.Timestamp('2025-02-01'))]
    print("После фильтрации по дате, строк:", len(df_raw))

    # 2. Выполнение EDA (опционально, графики будут отображены)
    print("Выполняется первичный анализ (EDA)...")
    run_eda(df_raw)

    # 3. Предобработка данных
    print("Выполняется предобработка данных...")
    df_clean = preprocess_data(df_raw)

    # 4. Агрегация данных по дням
    print("Агрегация данных по дням...")
    df_daily = aggregate_daily(df_clean)
    print("Агрегация по дням (первые 5 строк):")
    print(df_daily.head())

    # Приводим столбец ds к типу datetime для агрегированных данных
    df_daily['ds'] = pd.to_datetime(df_daily['ds'], errors='coerce')

    # 5. Разделение данных на тренировочную и тестовую выборки (например, последние 30 дней для теста)
    test_size = 30  # число дней для тестовой выборки
    train_df, test_df = train_test_split_prophet(df_daily, test_size=test_size)
    print(f"Данные разделены: тренировка ({len(train_df)} точек), тест ({len(test_df)} точек).")

    # 6. Обучение модели Prophet на тренировочных данных
    print("Обучение модели на тренировочных данных...")
    model = train_prophet_model(train_df)

    # 7. Прогноз на период тестовой выборки (30 дней)
    forecast_test = make_forecast(model, periods=test_size)
    forecast_for_test = forecast_test.tail(test_size).copy()  # делаем копию, чтобы избежать предупреждений

    # Приводим столбец ds к типу datetime для прогнозных данных и тестового DataFrame
    forecast_for_test['ds'] = pd.to_datetime(forecast_for_test['ds'], errors='coerce')
    test_df['ds'] = pd.to_datetime(test_df['ds'], errors='coerce')

    # 8. Оценка прогноза: вычисляем метрики и строим график сравнения
    metrics = evaluate_forecast(test_df['y'], forecast_for_test['yhat'])
    print("Метрики прогноза на тестовой выборке:")
    print(metrics)

    print("Построение графика сравнения фактических и прогнозных значений (тест)...")
    plot_actual_vs_forecast(test_df, forecast_for_test)

    # 9. Обучение модели на полном датасете и прогнозирование будущего (например, на следующие 30 дней)
    print("Обучение модели на полном датасете и прогнозирование будущего...")
    model_full = train_prophet_model(df_daily)
    forecast_future = make_forecast(model_full, periods=30)

    print("Построение графика прогноза на будущее...")
    plot_forecast(model_full, forecast_future)
    plot_components(model_full, forecast_future)


if __name__ == "__main__":
    main()
