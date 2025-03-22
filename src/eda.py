"""
Модуль для первичного анализа данных (EDA).
Запускается как самостоятельный скрипт:
    python eda.py
"""

import pandas as pd
import matplotlib.pyplot as plt


def load_data(path: str) -> pd.DataFrame:
    """
    Загружает данные из Parquet-файла по указанному пути.

    :param path: Путь к Parquet-файлу.
    :return: DataFrame с загруженными данными.
    """
    df = pd.read_parquet(path)
    return df


def run_eda(df: pd.DataFrame) -> None:
    """
    Выполняет первичный анализ (EDA):
    - Выводит первые строки
    - Общую информацию (df.info())
    - Статистическую сводку (df.describe())
    - Количество пропусков
    - Пример визуализации (гистограмма дат или другого признака)

    :param df: DataFrame с данными.
    """
    print("=== Первые 5 строк ===")
    print(df.head(), "\n")

    print("=== Информация о DataFrame ===")
    print(df.info(), "\n")

    print("=== Статистическая сводка (describe) ===")
    print(df.describe(), "\n")

    print("=== Пропущенные значения ===")
    print(df.isna().sum(), "\n")

    # Если в данных есть колонка с датой и временем (tpep_pickup_datetime), построим гистограмму
    if 'tpep_pickup_datetime' in df.columns:
        # Убедимся, что колонка в формате datetime
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')

        plt.figure(figsize=(10, 5))
        df['tpep_pickup_datetime'].hist(bins=50, color='skyblue', edgecolor='black')
        plt.title("Распределение дат (tpep_pickup_datetime)")
        plt.xlabel("Дата и время")
        plt.ylabel("Число записей")
        plt.tight_layout()
        plt.show()

    # (Опционально) Построим гистограмму распределения расстояний, если есть trip_distance
    if 'trip_distance' in df.columns:
        plt.figure(figsize=(10, 5))
        df['trip_distance'].hist(bins=50, color='salmon', edgecolor='black')
        plt.title("Распределение дистанции поездок (trip_distance)")
        plt.xlabel("Расстояние, мили")
        plt.ylabel("Число записей")
        plt.tight_layout()
        plt.show()


def main():
    """
    Точка входа: загружает данные и запускает EDA.
    """
    # Укажите путь к вашему файлу Parquet
    # Если eda.py лежит в папке src, а данные в папке data, используйте "../data/..."
    data_path = "../data/yellow_tripdata_2025-01.parquet"

    df = load_data(data_path)
    run_eda(df)


if __name__ == "__main__":
    main()
