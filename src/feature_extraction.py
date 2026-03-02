"""
Feature extraction functions for parsing raw text data.
"""

import re
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from config.config import (
    IT_KEYWORDS, SENIOR_KEYWORDS, MIDDLE_KEYWORDS, JUNIOR_KEYWORDS,
    JUNIOR_EXPERIENCE_MAX, MIDDLE_EXPERIENCE_MAX, MIN_SALARY, MAX_SALARY,
    TOP_CITIES_COUNT
)


def extract_age(demo: str) -> int | None:
    """
    Extract age from demographic string.

    Args:
        demo: String with demographic information (e.g., "Мужчина, 42 года, родился 6 октября 1976")

    Returns:
        Age in years or None if extraction fails
    """
    if not isinstance(demo, str):
        return None

    # Look for age pattern "XX лет/год/года"
    age_pattern = r'(\d{1,2})\s*(лет|год|года)'
    match = re.search(age_pattern, demo)

    if match:
        try:
            return int(match.group(1))
        except (ValueError, IndexError):
            pass

    # Alternative: calculate from birth year
    if 'родился' in demo:
        year_pattern = r'родился (\d{1,2}) (\w+) (\d{4})'
        match = re.search(year_pattern, demo)

        if match:
            try:
                birth_year = int(match.group(3))
                current_year = datetime.now().year
                return current_year - birth_year
            except (ValueError, IndexError):
                pass

    return None


def extract_gender(demo: str) -> str | None:
    """
    Extract gender from demographic string.

    Args:
        demo: String with demographic information

    Returns:
        'Мужчина', 'Женщина', or None if extraction fails
    """
    if not isinstance(demo, str):
        return None

    if 'Мужчина' in demo:
        return 'Мужчина'
    if 'Женщина' in demo:
        return 'Женщина'

    return None


def extract_salary(salary_str: str) -> int | None:
    """
    Extract numeric salary value from string.

    Args:
        salary_str: String with salary information (e.g., "27 000 руб.")

    Returns:
        Salary in numeric format or None if extraction fails
    """
    if not isinstance(salary_str, str):
        return None

    # Remove spaces and non-breaking spaces
    salary_str = salary_str.replace(' ', '').replace('\xa0', '')

    # Find all numbers in the string
    numbers = re.findall(r'\d+', salary_str)

    if not numbers:
        return None

    try:
        salary = int(numbers[0])

        # Validate realistic salary range
        if salary < MIN_SALARY or salary > MAX_SALARY:
            return None

        return salary

    except (ValueError, IndexError):
        return None


def extract_experience_months(exp_str: str) -> int:
    """
    Extract total work experience in months.

    Args:
        exp_str: String with experience description (e.g., "Опыт работы 6 лет 1 месяц")

    Returns:
        Total experience in months (0 if extraction fails)
    """
    if not isinstance(exp_str, str):
        return 0

    total_months = 0

    # Extract years
    years_pattern = r'(\d+)\s*(лет|год|года)'
    years_matches = re.findall(years_pattern, exp_str)

    for match in years_matches:
        try:
            years = int(match[0])
            total_months += years * 12
        except (ValueError, IndexError):
            continue

    # Extract months
    months_pattern = r'(\d+)\s*(месяц|месяца|месяцев)'
    months_matches = re.findall(months_pattern, exp_str)

    for match in months_matches:
        try:
            months = int(match[0])
            total_months += months
        except (ValueError, IndexError):
            continue

    return total_months


def extract_city(city_str: str) -> str | None:
    """
    Extract clean city name from location string.

    Args:
        city_str: String with city information (e.g., "Москва, не готов к переезду")

    Returns:
        Clean city name or None if extraction fails
    """
    if not isinstance(city_str, str):
        return None

    # Take part before first comma and strip whitespace
    city = city_str.split(',')[0].strip()
    return city if city else None


def is_it_position(position: str) -> bool:
    """
    Check if position is IT-related based on keywords.

    Args:
        position: Position title to check

    Returns:
        True if position is IT-related, False otherwise
    """
    if not isinstance(position, str):
        return False

    position_lower = position.lower()
    return any(keyword in position_lower for keyword in IT_KEYWORDS)


def determine_level(row: pd.Series) -> str:
    """
    Determine developer level based on position title and experience.

    Args:
        row: Pandas Series with 'position', 'last_position', 'experience_months'

    Returns:
        Developer level: 'junior', 'middle', 'senior', or 'unknown'
    """
    # Build combined position text
    position_text = _build_position_text(row)
    experience = row.get('experience_months', 0)

    # Check for level keywords in position
    level_from_position = _get_level_from_position(position_text, experience)

    if level_from_position != 'unknown':
        return level_from_position

    # Fallback to experience-based determination
    return _get_level_from_experience(experience)


def _build_position_text(row: pd.Series) -> str:
    """Combine position fields into a single lowercase string."""
    position_fields = ['position', 'last_position']
    text_parts = []

    for field in position_fields:
        field_value = row.get(field)
        if isinstance(field_value, str) and field_value.strip():
            text_parts.append(field_value.strip().lower())

    return ' '.join(text_parts)


def _get_level_from_position(position_text: str, experience: int) -> str:
    """Determine level from position keywords."""
    # Check senior keywords
    for keyword in SENIOR_KEYWORDS:
        if keyword in position_text:
            if experience >= JUNIOR_EXPERIENCE_MAX:
                return 'senior'

    # Check middle keywords
    for keyword in MIDDLE_KEYWORDS:
        if keyword in position_text:
            if experience >= JUNIOR_EXPERIENCE_MAX // 2:
                return 'middle'

    # Check junior keywords
    for keyword in JUNIOR_KEYWORDS:
        if keyword in position_text:
            return 'junior'

    return 'unknown'


def _get_level_from_experience(experience: int) -> str:
    """Determine level based on experience thresholds."""
    if experience <= JUNIOR_EXPERIENCE_MAX:
        return 'junior'
    elif experience <= MIDDLE_EXPERIENCE_MAX:
        return 'middle'
    else:
        return 'senior'


def encode_cities(df: pd.DataFrame, city_column: str) -> pd.DataFrame:
    """
    Encode cities, keeping only top N cities and grouping others.

    Args:
        df: DataFrame with city column
        city_column: Name of the city column

    Returns:
        DataFrame with encoded city column
    """
    df = df.copy()
    top_cities = df[city_column].value_counts().head(TOP_CITIES_COUNT).index.tolist()

    df['city_top'] = df[city_column].apply(
        lambda x: x if x in top_cities else 'Другие'
    )

    return df


def extract_all_features(df: pd.DataFrame) -> tuple:
    """
    Complete feature extraction pipeline.

    Args:
        df: Raw DataFrame

    Returns:
        Tuple of (X, y, feature_names, label_encoder)
    """
    # Extract basic features
    df['age'] = df['demographics'].apply(extract_age)
    df['gender'] = df['demographics'].apply(extract_gender)
    df['salary_numeric'] = df['salary'].apply(extract_salary)
    df['experience_months'] = df['experience'].apply(extract_experience_months)
    df['city_clean'] = df['city'].apply(extract_city)

    # Determine target variable
    df['level'] = df.apply(determine_level, axis=1)
    df = df[df['level'] != 'unknown']

    # Encode target
    label_encoder = LabelEncoder()
    df['level_encoded'] = label_encoder.fit_transform(df['level'])

    # Encode cities
    df = encode_cities(df, 'city_clean')

    # Prepare feature matrix
    X = df[['age', 'salary_numeric', 'experience_months', 'city_top', 'gender']].copy()
    y = df['level_encoded']

    # One-hot encoding
    X_encoded = pd.get_dummies(X, columns=['city_top', 'gender'], drop_first=True)

    return X_encoded, y, list(X_encoded.columns), label_encoder