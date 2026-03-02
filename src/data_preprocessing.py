"""
Data loading and preprocessing functions.
"""

import pandas as pd
import numpy as np
from src.feature_extraction import is_it_position


def load_and_rename_columns(filepath: str) -> pd.DataFrame:
    """
    Load CSV data and rename columns to English for easier handling.

    Args:
        filepath: Path to the CSV file

    Returns:
        DataFrame with renamed columns
    """
    df = pd.read_csv(filepath, low_memory=False)

    # Rename columns to English
    df = df.rename(columns={
        'Unnamed: 0': 'id',
        'Пол, возраст': 'demographics',
        'ЗП': 'salary',
        'Ищет работу на должность:': 'position',
        'Город': 'city',
        'Занятость': 'employment_type',
        'График': 'schedule',
        'Опыт (двойное нажатие для полной версии)': 'experience',
        'Последенее/нынешнее место работы': 'last_workplace',
        'Последеняя/нынешняя должность': 'last_position',
        'Образование и ВУЗ': 'education',
        'Обновление резюме': 'resume_update',
        'Авто': 'car'
    })

    return df


def filter_it_resumes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter only IT-related resumes based on position keywords.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame containing only IT-related resumes
    """
    it_mask = df['position'].apply(is_it_position) | df['last_position'].apply(is_it_position)
    it_df = df[it_mask].copy()

    print(f"Total resumes: {len(df)}")
    print(f"IT resumes: {len(it_df)}")
    print(f"IT resumes percentage: {len(it_df) / len(df) * 100:.2f}%")

    return it_df


def prepare_features_dataframe(it_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a DataFrame with selected features for modeling.

    Args:
        it_df: Filtered IT resumes DataFrame

    Returns:
        DataFrame with selected features
    """
    features_df = it_df[[
        'level', 'age', 'salary_numeric', 'experience_months',
        'city_clean', 'gender', 'education', 'employment_type'
    ]].copy()

    return features_df


def handle_missing_values(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the feature DataFrame.

    Args:
        features_df: Feature DataFrame with potential missing values

    Returns:
        DataFrame with handled missing values
    """
    df = features_df.copy()

    # Fill missing values with appropriate statistics
    df['age'] = df['age'].fillna(df['age'].median())
    df['salary_numeric'] = df['salary_numeric'].fillna(df['salary_numeric'].median())
    df['experience_months'] = df['experience_months'].fillna(0)

    return df