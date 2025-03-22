"""
Модуль для первичного анализа данных (EDA).
Запускается как самостоятельный скрипт:
    python eda.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


def load_all_data(data_dir: str) -> pd.DataFrame:
    """
    Считывает все Parquet-файлы за 2024 год + январь 2025,
    объединяет их в один DataFrame.

    :param data_dir: Путь к директории, где хранятся файлы Parquet.
    :return: Единый DataFrame со всеми данными.
    """
    # Собираем все файлы за 2024 год
    all_files_2024 = sorted(glob.glob(os.path.join(data_dir, "yellow_tripdata_2024-*.parquet")))

    # Файл за январь 2025
    file_jan_2025 = os.path.join(data_dir, "yellow_tripdata_2025-01.parquet")

    df_list = []
    # Читаем файлы за 2024 год
    for file_path in all_files_2024:
        df_temp = pd.read_parquet(file_path)
        df_list.append(df_temp)

    # Читаем январь 2025
    df_jan_2025 = pd.read_parquet(file_jan_2025)
    df_list.append(df_jan_2025)

    # Объединяем все в один DataFrame
    df_all = pd.concat(df_list, ignore_index=True)
    return df_all


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
    Точка входа: загружает данные за 2024 год + январь 2025 и запускает EDA.
    """
    data_dir = "../data"  # Папка, где лежат файлы Parquet
    df = load_all_data(data_dir)
    print(f"Считано {len(df)} строк из всех файлов 2024 года и января 2025.")
    run_eda(df)


if __name__ == "__main__":
    main()
