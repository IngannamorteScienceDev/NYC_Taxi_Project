"""
Модуль для агрегации данных.
Содержит функции для группировки данных по выбранному временно́му интервалу.
"""

import pandas as pd
import glob
import os


def load_all_data(data_dir: str) -> pd.DataFrame:
    """
    Считывает все Parquet-файлы за 2024 год и файл за январь 2025,
    объединяет их в один DataFrame и оставляет только записи с датами от 2024-01-01 до 2025-02-01.

    :param data_dir: Путь к директории, где лежат файлы Parquet.
    :return: Объединённый DataFrame со всеми данными в нужном диапазоне.
    """
    # Получаем файлы за 2024 год
    files_2024 = sorted(glob.glob(os.path.join(data_dir, "yellow_tripdata_2024-*.parquet")))
    # Файл за январь 2025
    file_jan_2025 = os.path.join(data_dir, "yellow_tripdata_2025-01.parquet")

    df_list = []
    for file in files_2024:
        df_temp = pd.read_parquet(file)
        df_list.append(df_temp)

    df_list.append(pd.read_parquet(file_jan_2025))
    df_all = pd.concat(df_list, ignore_index=True)

    # Приводим столбец к datetime, если это необходимо
    if not pd.api.types.is_datetime64_any_dtype(df_all['tpep_pickup_datetime']):
        df_all['tpep_pickup_datetime'] = pd.to_datetime(df_all['tpep_pickup_datetime'], errors='coerce')

    # Фильтруем записи: оставляем даты от 2024-01-01 до 2025-02-01 (включительно начало и исключая конец)
    df_all = df_all[(df_all['tpep_pickup_datetime'] >= '2024-01-01') &
                    (df_all['tpep_pickup_datetime'] < '2025-02-01')]

    return df_all


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует данные по дням, подсчитывая количество поездок для каждого дня.
    Использует столбец 'tpep_pickup_datetime'.

    :param df: DataFrame с данными такси.
    :return: DataFrame с агрегированными данными, с колонками:
             - ds: дата (без времени)
             - y: количество поездок в этот день
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['tpep_pickup_datetime']):
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df['pickup_date'] = df['tpep_pickup_datetime'].dt.date
    df_daily = df.groupby('pickup_date').size().reset_index(name='trip_count')
    df_daily.rename(columns={'pickup_date': 'ds', 'trip_count': 'y'}, inplace=True)
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
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['tpep_pickup_datetime']):
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.floor('h')
    df_hourly = df.groupby('pickup_hour').size().reset_index(name='trip_count')
    df_hourly.rename(columns={'pickup_hour': 'ds', 'trip_count': 'y'}, inplace=True)
    df_hourly = df_hourly.sort_values(by='ds').reset_index(drop=True)
    return df_hourly


if __name__ == "__main__":
    data_dir = "../data"
    df = load_all_data(data_dir)
    print(f"Считано {len(df)} строк из всех файлов после фильтрации.")

    df_daily = aggregate_daily(df)
    print("Агрегация по дням:")
    print(df_daily.head())

    df_hourly = aggregate_hourly(df)
    print("\nАгрегация по часам:")
    print(df_hourly.head())
