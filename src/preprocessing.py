"""
Модуль для предобработки данных: заполнение пропусков и корректировка выбросов.
"""

import pandas as pd
import numpy as np

def fill_missing_numeric(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """
    Заполняет пропуски в числовых столбцах медианными значениями.

    :param df: Исходный DataFrame.
    :param numeric_cols: Список имен столбцов с числовыми данными.
    :return: DataFrame с заполненными пропусками.
    """
    for col in numeric_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    return df

def fill_missing_categorical(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    """
    Заполняет пропуски в категориальных столбцах наиболее часто встречающимся значением (модой).

    :param df: Исходный DataFrame.
    :param categorical_cols: Список имен категориальных столбцов.
    :return: DataFrame с заполненными пропусками.
    """
    for col in categorical_cols:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
    return df

def cap_outliers(df: pd.DataFrame, col: str, lower_percentile: float = 0.01, upper_percentile: float = 0.99) -> pd.DataFrame:
    """
    Корректирует выбросы в числовом столбце, обрезая значения до заданных процентилей.

    :param df: Исходный DataFrame.
    :param col: Имя столбца, в котором необходимо скорректировать выбросы.
    :param lower_percentile: Нижний процентиль (по умолчанию 1-й).
    :param upper_percentile: Верхний процентиль (по умолчанию 99-й).
    :return: DataFrame с обрезанными выбросами в указанном столбце.
    """
    lower_val = df[col].quantile(lower_percentile)
    upper_val = df[col].quantile(upper_percentile)
    df[col] = df[col].clip(lower=lower_val, upper=upper_val)
    return df

def remove_negative_values(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Для столбцов, где не должно быть отрицательных значений (например, дистанция или суммы),
    заменяет отрицательные значения на NaN.

    :param df: Исходный DataFrame.
    :param cols: Список имен столбцов, где отрицательные значения считаются ошибочными.
    :return: DataFrame с замененными отрицательными значениями.
    """
    for col in cols:
        df.loc[df[col] < 0, col] = np.nan
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Выполняет комплексную предобработку данных:
    1. Заменяет отрицательные значения в определённых столбцах (например, дистанция и суммы) на NaN.
    2. Заполняет пропуски в числовых столбцах медианными значениями.
    3. Заполняет пропуски в категориальных столбцах модой.
    4. Корректирует выбросы для выбранных столбцов.

    :param df: Исходный DataFrame.
    :return: Предобработанный DataFrame.
    """
    # 1. Заменяем отрицательные значения в столбцах, где их быть не должно
    df = remove_negative_values(df, cols=['trip_distance', 'fare_amount', 'total_amount'])

    # 2. Заполнение пропусков в числовых столбцах
    numeric_cols = [
        'passenger_count',
        'congestion_surcharge',
        'Airport_fee',
        'RatecodeID',
        'fare_amount',
        'total_amount'
    ]
    df = fill_missing_numeric(df, numeric_cols)

    # 3. Заполнение пропусков в категориальных столбцах
    categorical_cols = ['store_and_fwd_flag']
    df = fill_missing_categorical(df, categorical_cols)

    # 4. Корректировка выбросов для ключевых числовых столбцов
    for col in ['trip_distance', 'fare_amount', 'total_amount']:
        df = cap_outliers(df, col, lower_percentile=0.01, upper_percentile=0.99)

    return df

if __name__ == "__main__":
    # Пример использования: загрузка данных и их предобработка.
    # Убедитесь, что путь к файлу корректный.
    data_path = "../data/yellow_tripdata_2025-01.parquet"
    df = pd.read_parquet(data_path)

    print("До предобработки:")
    print(df.isna().sum())

    df_processed = preprocess_data(df)

    print("\nПосле предобработки:")
    print(df_processed.isna().sum())
