#!/usr/bin/env python3
"""
Main entry point for IT Developer Level Classification.
"""

import argparse
import warnings

import pandas as pd

from src.data_preprocessing import (
    load_and_rename_columns
)
from src.feature_extraction import extract_all_features
from src.model import train_model, split_and_scale_data
from src.utils import (
    analyze_class_distribution, analyze_errors,
    generate_business_conclusion
)

warnings.filterwarnings('ignore')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='IT Developer Level Classification from hh.ru resumes'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to the input CSV file'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.3,
        help='Proportion of data for testing (default: 0.3)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    print("=" * 60)
    print("IT DEVELOPER LEVEL CLASSIFICATION")
    print("=" * 60)

    # Load and preprocess data
    print("\n📥 Loading data...")
    df = load_and_rename_columns(args.data)

    # Extract features
    print("\n🔧 Extracting features...")
    X, y, feature_names, label_encoder = extract_all_features(df)

    # Analyze class distribution
    print("\n📊 Class distribution:")
    level_counts = pd.Series(label_encoder.inverse_transform(y)).value_counts()
    print(level_counts)
    analyze_class_distribution(df)

    # Train model
    print("\n🧠 Training model...")
    model, scaler, metrics = train_model(X, y, feature_names, label_encoder)

    # Analyze errors
    X_train_scaled, X_test_scaled, y_train, y_test, _, _ = split_and_scale_data(X, y)
    y_pred = model.predict(X_test_scaled)
    test_results, error_analysis = analyze_errors(
        y_test, y_pred, X_test_scaled, label_encoder
    )

    # Generate conclusion
    generate_business_conclusion(metrics, error_analysis)

    print("\n✅ Pipeline completed successfully!")


if __name__ == "__main__":
    main()