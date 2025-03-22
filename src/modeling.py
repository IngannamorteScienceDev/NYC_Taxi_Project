"""
Модуль для построения модели прогнозирования с использованием Prophet на реальных данных такси.
Улучшения:
 - Точная настройка гиперпараметров модели (changepoint_prior_scale, seasonality_prior_scale, n_changepoints, seasonality_mode).
 - Опциональная кросс-валидация модели (для оценки стабильности прогноза).
 - Визуализация прогноза и его компонентов.
"""

import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# Импортируем функции агрегации, включая load_all_data
from src.aggregation import load_all_data, aggregate_daily

def train_prophet_model(df: pd.DataFrame) -> Prophet:
    """
    Обучает модель Prophet на агрегированных данных с тонкой настройкой гиперпараметров.

    :param df: DataFrame с агрегированными данными (колонки 'ds' и 'y').
    :return: Обученная модель Prophet.
    """
    model = Prophet(
        growth='linear',
        seasonality_mode='additive',
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        n_changepoints=25
    )
    model.fit(df)
    return model

def make_forecast(model: Prophet, periods: int) -> pd.DataFrame:
    """
    Создает DataFrame с будущими датами и вычисляет прогноз с использованием обученной модели.

    :param model: Обученная модель Prophet.
    :param periods: Количество будущих периодов (дней) для прогноза.
    :return: DataFrame с прогнозом, содержащий колонки 'yhat', 'yhat_lower', 'yhat_upper' и др.
    """
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

def plot_forecast(model: Prophet, forecast: pd.DataFrame) -> None:
    """
    Визуализирует прогноз, построенный моделью Prophet.

    :param model: Обученная модель Prophet.
    :param forecast: DataFrame с прогнозом.
    """
    fig = model.plot(forecast)
    plt.title("Прогноз спроса на такси")
    plt.xlabel("Дата")
    plt.ylabel("Количество поездок")
    plt.show()

def plot_components(model: Prophet, forecast: pd.DataFrame) -> None:
    """
    Визуализирует компоненты прогноза (тренд, недельная и ежедневная сезонности).

    :param model: Обученная модель Prophet.
    :param forecast: DataFrame с прогнозом.
    """
    fig = model.plot_components(forecast)
    plt.show()

def run_cross_validation(model: Prophet, initial: str, period: str, horizon: str) -> pd.DataFrame:
    """
    Выполняет кросс-валидацию модели Prophet и возвращает DataFrame с метриками производительности.

    :param model: Обученная модель Prophet.
    :param initial: Начальный период обучения (например, '30 days').
    :param period: Период между пересчетами (например, '7 days').
    :param horizon: Горизонт прогноза (например, '7 days').
    :return: DataFrame с метриками кросс-валидации.
    """
    df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon)
    df_metrics = performance_metrics(df_cv)
    return df_metrics

if __name__ == "__main__":
    # Загрузим реальные данные такси, объединяя файлы за 2024 год и январь 2025
    data_dir = "../data"
    df_raw = load_all_data(data_dir)
    print("Считано", len(df_raw), "строк из всех файлов.")

    # Агрегируем данные по дням (с использованием функции aggregate_daily)
    df_daily = aggregate_daily(df_raw)
    print("Агрегация по дням:")
    print(df_daily.head())

    # Обучаем модель Prophet на агрегированных данных
    model = train_prophet_model(df_daily)

    # Получаем прогноз на 30 дней вперед
    forecast = make_forecast(model, periods=30)
    print("Прогноз (последние 5 строк):")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # Визуализируем прогноз и его компоненты
    plot_forecast(model, forecast)
    plot_components(model, forecast)

    # Опционально: выполнить кросс-валидацию
    # cv_metrics = run_cross_validation(model, initial='30 days', period='7 days', horizon='7 days')
    # print("Метрики кросс-валидации:")
    # print(cv_metrics.head())
