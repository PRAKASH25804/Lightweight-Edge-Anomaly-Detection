"""
Edge-IIoTset Data Preprocessing Module
"Development of Lightweight Neural Network Architecture for Real-Time Anomaly Detection on Resource-Constrained Edge Devices"

Features:
- Robust loading and data inspection
- Removal of duplicates and invalid artifacts
- Strict prevention of data leakage (split before scaling / feature selection)
- Stratified splitting preserving attack subclass distribution
- Preprocessor artifact serialization for exact edge deployment alignment
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import config

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EdgeDataPreprocessor:
    """
    Standardized Preprocessor for Edge-IIoTset.
    Guarantees that training, validation, testing, and edge runtime inference
    use identical transformations and feature ordering.
    """

    METHOD_MAPPING = {"0": 0, "0.0": 0, "GET": 1, "POST": 2, "TRACE": 3, "OPTIONS": 4}
    VERSION_MAPPING = {"0": 0, "0.0": 0, "HTTP/1.1": 1, "HTTP/1.0": 2}

    def __init__(self):
        self.drop_columns = list(config.DROP_COLUMNS)
        self.active_features: List[str] = []
        self.zero_variance_cols: List[str] = []
        self.scaler: StandardScaler = StandardScaler()
        self.is_fitted: bool = False

    def clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies deterministic cleaning to candidate feature dataframe."""
        df_clean = df.copy()

        # Drop raw high-cardinality/payload/identifier columns if present
        cols_to_drop = [c for c in self.drop_columns if c in df_clean.columns]
        if cols_to_drop:
            df_clean = df_clean.drop(columns=cols_to_drop)

        # Categorical variable encoding
        if "http.request.method" in df_clean.columns:
            df_clean["http.request.method"] = (
                df_clean["http.request.method"]
                .astype(str)
                .map(lambda x: self.METHOD_MAPPING.get(x, 0))
                .astype(np.float32)
            )

        if "http.request.version" in df_clean.columns:
            df_clean["http.request.version"] = (
                df_clean["http.request.version"]
                .astype(str)
                .map(lambda x: self.VERSION_MAPPING.get(x, 0))
                .astype(np.float32)
            )

        if "mqtt.protoname" in df_clean.columns:
            df_clean["mqtt.protoname"] = (
                df_clean["mqtt.protoname"]
                .astype(str)
                .map(lambda x: 1.0 if "MQTT" in x else 0.0)
                .astype(np.float32)
            )

        if "mqtt.topic" in df_clean.columns:
            df_clean["mqtt.topic"] = (
                df_clean["mqtt.topic"]
                .astype(str)
                .map(lambda x: 1.0 if "Temperature" in x else 0.0)
                .astype(np.float32)
            )

        # Clean remaining columns to numeric float32
        for col in df_clean.columns:
            if df_clean[col].dtype == object or df_clean[col].dtype == "string":
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0.0).astype(np.float32)
            else:
                df_clean[col] = df_clean[col].fillna(0.0).astype(np.float32)

        return df_clean

    def fit(self, X_train: pd.DataFrame) -> "EdgeDataPreprocessor":
        """
        Fits preprocessor on training data ONLY to prevent data leakage.
        Identifies zero-variance features on train split and fits StandardScaler.
        """
        logger.info("Fitting preprocessor on training set...")
        X_clean = self.clean_features(X_train)

        # Detect zero-variance features strictly on X_train
        variances = X_clean.var(axis=0)
        self.zero_variance_cols = variances[variances == 0.0].index.tolist()
        self.active_features = [c for c in X_clean.columns if c not in self.zero_variance_cols]

        logger.info(f"Identified {len(self.zero_variance_cols)} zero-variance features on training set: {self.zero_variance_cols}")
        logger.info(f"Retained {len(self.active_features)} active candidate features.")

        X_active = X_clean[self.active_features]
        self.scaler.fit(X_active)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms input features using fitted scaler and active feature ordering."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform can be called.")

        X_clean = self.clean_features(X)

        # Align active features
        for col in self.active_features:
            if col not in X_clean.columns:
                X_clean[col] = 0.0

        X_active = X_clean[self.active_features]
        X_scaled = self.scaler.transform(X_active).astype(np.float32)
        return X_scaled

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

    def save(self, filepath: Path) -> None:
        """Serializes preprocessor state for edge deployment."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "drop_columns": self.drop_columns,
                "active_features": self.active_features,
                "zero_variance_cols": self.zero_variance_cols,
                "scaler": self.scaler,
                "method_mapping": self.METHOD_MAPPING,
                "version_mapping": self.VERSION_MAPPING,
            },
            filepath,
        )
        logger.info(f"Saved preprocessor artifact to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "EdgeDataPreprocessor":
        """Loads serialized preprocessor artifact."""
        state = joblib.load(filepath)
        preprocessor = cls()
        preprocessor.drop_columns = state["drop_columns"]
        preprocessor.active_features = state["active_features"]
        preprocessor.zero_variance_cols = state["zero_variance_cols"]
        preprocessor.scaler = state["scaler"]
        preprocessor.METHOD_MAPPING = state["method_mapping"]
        preprocessor.VERSION_MAPPING = state["version_mapping"]
        preprocessor.is_fitted = True
        return preprocessor


def profile_raw_dataset(csv_path: Path) -> Dict[str, Any]:
    """Generates a comprehensive dataset profile and verification metrics."""
    logger.info(f"Profiling raw dataset from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)

    num_rows, num_cols = df.shape
    num_duplicates = int(df.duplicated().sum())
    missing_total = int(df.isnull().sum().sum())

    binary_dist = df[config.TARGET_BINARY].value_counts().to_dict()
    multiclass_dist = df[config.TARGET_MULTICLASS].value_counts().to_dict()

    cat_cols = [c for c in df.columns if df[c].dtype == object]
    num_cols_list = [c for c in df.columns if df[c].dtype != object]

    normal_count = binary_dist.get(0, 0)
    anomaly_count = binary_dist.get(1, 0)
    imbalance_ratio = float(anomaly_count / max(1, normal_count))

    profile = {
        "dataset_name": "Edge-IIoTset (ML-EdgeIIoT-dataset)",
        "total_rows": num_rows,
        "total_columns": num_cols,
        "duplicate_rows": num_duplicates,
        "missing_values": missing_total,
        "categorical_columns_count": len(cat_cols),
        "numerical_columns_count": len(num_cols_list),
        "target_binary_distribution": {
            "Normal (0)": normal_count,
            "Anomaly (1)": anomaly_count,
        },
        "target_anomaly_percentage": float(anomaly_count / num_rows * 100),
        "target_normal_percentage": float(normal_count / num_rows * 100),
        "imbalance_ratio_anomaly_to_normal": round(imbalance_ratio, 2),
        "attack_types_count": len(multiclass_dist),
        "attack_types_distribution": multiclass_dist,
    }

    profile_path = config.RESULTS_DIR / "dataset_profile.json"
    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=4)
    logger.info(f"Dataset profile saved to {profile_path}")
    return profile


def prepare_datasets(
    csv_path: Path = config.RAW_DATA_PATH,
    random_seed: int = config.RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Full reproducible pipeline:
    1. Loads dataset
    2. Drops verified duplicate records
    3. Performs stratified Train/Val/Test split (70% / 15% / 15%)
    4. Fits preprocessor strictly on X_train
    5. Transforms X_train, X_val, X_test
    6. Saves preprocessor artifact and numpy cache
    """
    profile = profile_raw_dataset(csv_path)

    logger.info("Loading dataset for preprocessing...")
    df = pd.read_csv(csv_path, low_memory=False)
    initial_len = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {initial_len - len(df)} duplicate records. Clean records: {len(df)}")

    y_binary = df[config.TARGET_BINARY].values.astype(np.int64)
    y_multiclass = df[config.TARGET_MULTICLASS].values

    # Candidate feature columns
    drop_candidates = set(config.DROP_COLUMNS + [config.TARGET_BINARY, config.TARGET_MULTICLASS])
    feature_cols = [c for c in df.columns if c not in drop_candidates]
    X_df = df[feature_cols]

    # Stratified Train (70%) vs Temp (30%) based on multiclass labels for perfect subclass representation
    X_train_raw, X_temp_raw, y_train, y_temp, y_multi_train, y_multi_temp = train_test_split(
        X_df,
        y_binary,
        y_multiclass,
        test_size=(config.VAL_RATIO + config.TEST_RATIO),
        stratify=y_multiclass,
        random_state=random_seed,
    )

    # Stratified Val (15%) vs Test (15%)
    val_test_ratio = config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw,
        y_temp,
        test_size=val_test_ratio,
        stratify=y_multi_temp,
        random_state=random_seed,
    )

    logger.info(f"Split sizes: Train={len(X_train_raw)}, Val={len(X_val_raw)}, Test={len(X_test_raw)}")

    # Fit preprocessor ONLY on X_train_raw
    preprocessor = EdgeDataPreprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train_raw)
    X_val_scaled = preprocessor.transform(X_val_raw)
    X_test_scaled = preprocessor.transform(X_test_raw)

    feature_names = preprocessor.active_features

    # Save artifacts
    preprocessor_path = config.ARTIFACTS_DIR / "preprocessor.joblib"
    preprocessor.save(preprocessor_path)

    cache_path = config.ARTIFACTS_DIR / "processed_data.npz"
    np.savez_compressed(
        cache_path,
        X_train=X_train_scaled,
        y_train=y_train,
        X_val=X_val_scaled,
        y_val=y_val,
        X_test=X_test_scaled,
        y_test=y_test,
        feature_names=np.array(feature_names, dtype=str),
    )
    logger.info(f"Saved processed dataset cache to {cache_path}")

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, feature_names


if __name__ == "__main__":
    prepare_datasets()

