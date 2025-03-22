"""
Модуль для оценки результатов прогноза.
Содержит функции для расчёта метрик и визуализации сравнения фактических и прогнозных значений.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_forecast(actual: pd.Series, predicted: pd.Series) -> dict:
    """
    Вычисляет метрики качества прогноза: MAE, MSE, RMSE и MAPE.

    :param actual: Фактические значения (Series или список).
    :param predicted: Прогнозные значения (Series или список).
    :return: Словарь с метриками {'MAE': ..., 'MSE': ..., 'RMSE': ..., 'MAPE': ...}.
    """
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mask_nonzero = (actual != 0)
    mape = (
        np.mean(np.abs((actual[mask_nonzero] - predicted[mask_nonzero]) / actual[mask_nonzero])) * 100
        if mask_nonzero.any() else np.nan
    )
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }


def plot_actual_vs_forecast(
    df_actual: pd.DataFrame,
    df_forecast: pd.DataFrame,
    start_date=None,
    end_date=None
) -> None:
    """
    Строит график фактических значений и прогнозных значений на одном поле.

    :param df_actual: DataFrame с фактическими данными, должен содержать колонки:
                      - ds (дата/время)
                      - y (фактические значения)
    :param df_forecast: DataFrame с прогнозом Prophet, должен содержать колонки:
                        - ds (дата/время)
                        - yhat (прогноз)
    :param start_date: Начальная дата (строка или datetime), если нужно обрезать диапазон.
    :param end_date: Конечная дата (строка или datetime), если нужно обрезать диапазон.
    """
    # Создаём копии, чтобы избежать SettingWithCopyWarning
    df_actual = df_actual.copy()
    df_forecast = df_forecast.copy()

    # Приводим столбцы 'ds' к типу datetime
    df_actual['ds'] = pd.to_datetime(df_actual['ds'], errors='coerce')
    df_forecast['ds'] = pd.to_datetime(df_forecast['ds'], errors='coerce')

    merged_df = pd.merge(df_actual, df_forecast[['ds', 'yhat']], on='ds', how='inner')

    if start_date:
        merged_df = merged_df[merged_df['ds'] >= pd.to_datetime(start_date)]
    if end_date:
        merged_df = merged_df[merged_df['ds'] <= pd.to_datetime(end_date)]

    plt.figure(figsize=(10, 5))
    plt.plot(merged_df['ds'], merged_df['y'], label='Фактические', marker='o')
    plt.plot(merged_df['ds'], merged_df['yhat'], label='Прогноз', marker='x')
    plt.title("Сравнение фактических и прогнозных значений")
    plt.xlabel("Дата")
    plt.ylabel("Количество поездок")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def train_test_split_prophet(df: pd.DataFrame, test_size: int) -> (pd.DataFrame, pd.DataFrame):
    """
    Разделяет временной ряд на тренировочную и тестовую выборки по "последним" точкам.
    Подходит для Prophet-формата (ds, y).

    :param df: Исходный DataFrame с колонками ds, y (упорядочен по ds).
    :param test_size: Количество последних точек, которые выделяются под тест.
    :return: (train_df, test_df)
    """
    df = df.sort_values(by='ds').reset_index(drop=True)
    train_df = df.iloc[:-test_size]
    test_df = df.iloc[-test_size:]
    return train_df, test_df
