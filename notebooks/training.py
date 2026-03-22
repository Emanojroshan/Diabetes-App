# =============================================================================
# Diabetes Prediction - Model Training Script
# =============================================================================

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ------------------------------ Paths ----------------------------------------

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, '..', 'data', 'diabetes.csv')
MODEL_DIR   = os.path.join(BASE_DIR, '..', 'model')
MODEL_PATH  = os.path.join(MODEL_DIR, 'model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

# ------------------------------ Functions ------------------------------------

def load_data(path):
    """Load dataset and return a DataFrame."""
    df = pd.read_csv(path)
    print(f"[1] Dataset loaded  → shape: {df.shape}")
    return df


def split_features_target(df, target_col='Outcome'):
    """Split DataFrame into feature matrix X and target vector y."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    print(f"[2] Features: {X.shape}  |  Target: {y.shape}")
    print(f"    Class distribution:\n{y.value_counts().to_string()}")
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Perform stratified train-test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(f"[3] Train samples: {X_train.shape[0]}  |  Test samples: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """Fit StandardScaler on train set and transform both sets."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    print("[4] Feature scaling applied (StandardScaler)")
    return X_train_scaled, X_test_scaled, scaler


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """Train and return a RandomForestClassifier."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("[5] Model training complete (RandomForestClassifier)")
    return model


def evaluate_model(model, X_test, y_test):
    """Predict and print accuracy score."""
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[6] Model Accuracy: {accuracy * 100:.2f}%")
    return accuracy


def save_artifacts(model, scaler, model_path, scaler_path, model_dir):
    """Persist model and scaler to disk using pickle."""
    os.makedirs(model_dir, exist_ok=True)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"[7] Model saved  → {model_path}")

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"    Scaler saved → {scaler_path}")

# ------------------------------ Main Pipeline --------------------------------

def main():
    print("=" * 55)
    print("   Diabetes Prediction — Training Pipeline")
    print("=" * 55)

    df                                  = load_data(DATA_PATH)
    X, y                                = split_features_target(df)
    X_train, X_test, y_train, y_test    = split_data(X, y)
    X_train_sc, X_test_sc, scaler       = scale_features(X_train, X_test)
    model                               = train_model(X_train_sc, y_train)
    evaluate_model(model, X_test_sc, y_test)
    save_artifacts(model, scaler, MODEL_PATH, SCALER_PATH, MODEL_DIR)

    print("=" * 55)
    print("   Pipeline finished successfully!")
    print("=" * 55)


if __name__ == '__main__':
    main()
