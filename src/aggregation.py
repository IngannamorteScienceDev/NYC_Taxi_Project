"""
Модуль для агрегации данных.
Содержит функции для группировки данных по выбранному временно́му интервалу.
"""

import pandas as pd

def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует данные по дням, подсчитывая количество поездок для каждого дня.
    Использует столбец 'tpep_pickup_datetime'.

    :param df: DataFrame с данными такси.
    :return: DataFrame с агрегированными данными, с колонками:
             - ds: дата (без времени)
             - y: количество поездок в этот день
    """
    # Создаем копию DataFrame, чтобы не изменять оригинал
    df = df.copy()

    # Проверяем, что столбец 'tpep_pickup_datetime' имеет тип datetime, иначе приводим его
    if not pd.api.types.is_datetime64_any_dtype(df['tpep_pickup_datetime']):
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')

    # Создаем новый столбец с датой (без времени)
    df['pickup_date'] = df['tpep_pickup_datetime'].dt.date

    # Группируем по дате и считаем количество записей (поездок)
    df_daily = df.groupby('pickup_date').size().reset_index(name='trip_count')

    # Переименовываем столбцы для совместимости с Prophet (ds и y)
    df_daily.rename(columns={'pickup_date': 'ds', 'trip_count': 'y'}, inplace=True)

    # Сортируем по дате
    df_daily = df_daily.sort_values(by='ds').reset_index(drop=True)

    return df_daily

def aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует данные по часам, подсчитывая количество поездок для каждого часа.
    Использует столбец 'tpep_pickup_datetime'.

    :param df: DataFrame с данными такси.
    :return: DataFrame с агрегированными данными, с колонками:
             - ds: дата и время, округленные до часа
             - y: количество поездок в этот час
    """
    # Создаем копию DataFrame
    df = df.copy()

    # Проверяем, что 'tpep_pickup_datetime' имеет тип datetime, иначе приводим его
    if not pd.api.types.is_datetime64_any_dtype(df['tpep_pickup_datetime']):
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')

    # Округляем время до ближайшего часа (используем 'h' вместо 'H' согласно новому стандарту)
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.floor('h')

    # Группируем по округленному часу и считаем количество поездок
    df_hourly = df.groupby('pickup_hour').size().reset_index(name='trip_count')

    # Переименовываем столбцы для совместимости с Prophet
    df_hourly.rename(columns={'pickup_hour': 'ds', 'trip_count': 'y'}, inplace=True)

    # Сортируем по времени
    df_hourly = df_hourly.sort_values(by='ds').reset_index(drop=True)

    return df_hourly

if __name__ == "__main__":
    # Пример использования функций агрегации
    data_path = "../data/yellow_tripdata_2025-01.parquet"
    df = pd.read_parquet(data_path)

    # Агрегация по дням
    df_daily = aggregate_daily(df)
    print("Агрегация по дням:")
    print(df_daily.head())

    # Агрегация по часам
    df_hourly = aggregate_hourly(df)
    print("\nАгрегация по часам:")
    print(df_hourly.head())
