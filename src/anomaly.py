"""Autoencoder-style anomaly detection using a shallow MLP regressor.

Trains only on benign samples and flags anomalies based on reconstruction error.
If no dataset is provided, synthetic data is generated for a runnable demo.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


DEFAULT_DATA_PATH = "data/anomaly_traffic.csv"
DEFAULT_MODEL_PATH = "model/anomaly_model.pkl"


@dataclass
class AnomalyModel:
    scaler: StandardScaler
    model: MLPRegressor
    threshold: float
    feature_cols: List[str]


def _generate_synthetic_data(
    n_normal: int = 600,
    n_anomaly: int = 60,
    n_features: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    normal = rng.normal(loc=0.0, scale=1.0, size=(n_normal, n_features))
    anomalies = rng.normal(loc=4.0, scale=1.5, size=(n_anomaly, n_features))

    data = np.vstack([normal, anomalies])
    labels = np.hstack([np.zeros(n_normal, dtype=int), np.ones(n_anomaly, dtype=int)])

    columns = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(data, columns=columns)
    df["label"] = labels
    return df


def load_or_generate_data(path: str = DEFAULT_DATA_PATH) -> Tuple[pd.DataFrame, List[str]]:
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = _generate_synthetic_data()
        df.to_csv(path, index=False)

    feature_cols = [col for col in df.columns if col != "label"]
    return df, feature_cols


def train_autoencoder(
    data_path: str = DEFAULT_DATA_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
    hidden_layers: Tuple[int, int] = (32, 16),
    threshold_percentile: float = 95.0,
) -> str:
    df, feature_cols = load_or_generate_data(data_path)

    if "label" in df.columns:
        benign_df = df[df["label"] == 0]
    else:
        benign_df = df

    X = benign_df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42,
    )
    model.fit(X_scaled, X_scaled)

    reconstructed = model.predict(X_scaled)
    errors = np.mean(np.square(X_scaled - reconstructed), axis=1)
    threshold = float(np.percentile(errors, threshold_percentile))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    anomaly_model = AnomalyModel(
        scaler=scaler,
        model=model,
        threshold=threshold,
        feature_cols=feature_cols,
    )

    with open(model_path, "wb") as f:
        pickle.dump(anomaly_model, f)

    return model_path


def load_model(model_path: str = DEFAULT_MODEL_PATH) -> AnomalyModel:
    with open(model_path, "rb") as f:
        return pickle.load(f)


def detect_anomaly(features: Iterable[float], model_path: str = DEFAULT_MODEL_PATH) -> dict:
    anomaly_model = load_model(model_path)
    features_arr = np.array(list(features), dtype=float).reshape(1, -1)

    if features_arr.shape[1] != len(anomaly_model.feature_cols):
        raise ValueError(
            f"Expected {len(anomaly_model.feature_cols)} features, got {features_arr.shape[1]}."
        )

    scaled = anomaly_model.scaler.transform(features_arr)
    reconstructed = anomaly_model.model.predict(scaled)
    error = float(np.mean(np.square(scaled - reconstructed)))
    is_anomaly = error > anomaly_model.threshold

    return {
        "is_anomaly": is_anomaly,
        "reconstruction_error": error,
        "threshold": anomaly_model.threshold,
    }


if __name__ == "__main__":
    model_path = train_autoencoder()
    print(f"✅ Anomaly model trained and saved to {model_path}")

    sample = [0.1] * 10
    result = detect_anomaly(sample, model_path=model_path)
    print("Sample prediction:", result)
