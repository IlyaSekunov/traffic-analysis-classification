"""
Model training and evaluation functions.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE, TEST_SIZE, N_ESTIMATORS, MAX_DEPTH


def split_and_scale_data(X: pd.DataFrame, y: pd.Series):
    """
    Split data into train/test sets and scale numerical features.

    Args:
        X: Feature matrix
        y: Target vector

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, y_train, y_test, scaler, num_cols)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Scale numerical features
    num_cols = ['age', 'salary_numeric', 'experience_months']
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, num_cols


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Train Random Forest classifier with balanced class weights.

    Args:
        X_train: Training features
        y_train: Training targets

    Returns:
        Trained Random Forest model
    """
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series,
                   label_encoder, class_names=None) -> dict:
    """
    Evaluate model performance and display metrics.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        label_encoder: LabelEncoder for target variable
        class_names: Optional class names

    Returns:
        Dictionary with evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Get class names
    if class_names is None:
        class_names = label_encoder.inverse_transform([0, 1, 2])

    # Classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print("\nClassification Report:")
    print(report_df.round(3))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=14)
    plt.ylabel('True Class', fontsize=14)
    plt.tight_layout()
    plt.show()

    return {
        'report': report,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }


def analyze_feature_importance(model, feature_names: list, top_n: int = 10):
    """
    Analyze and visualize feature importance.

    Args:
        model: Trained Random Forest model
        feature_names: List of feature names
        top_n: Number of top features to display

    Returns:
        DataFrame with feature importances
    """
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Plot top features
    plt.figure(figsize=(10, 6))
    bars = plt.barh(feature_importance['feature'][:top_n],
                    feature_importance['importance'][:top_n])
    plt.xlabel('Feature Importance', fontsize=14)
    plt.title(f'Top-{top_n} Most Important Features', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    print(f"\nTop-{top_n} Most Important Features:")
    print(feature_importance.head(top_n))

    return feature_importance


def train_model(X, y, feature_names, label_encoder):
    """
    Complete model training pipeline.

    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        label_encoder: LabelEncoder for target

    Returns:
        Tuple of (model, scaler, metrics)
    """
    # Split and scale
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, _ = split_and_scale_data(X, y)

    # Train model
    model = train_random_forest(X_train_scaled, y_train)

    # Evaluate
    metrics = evaluate_model(model, X_test_scaled, y_test, label_encoder)

    # Feature importance
    feature_importance = analyze_feature_importance(model, feature_names)

    return model, scaler, metrics