"""
flag_anomalies pattern — adapted for physiological stress detection.
Same core idea as your fraud/AML anomaly utilities: combine a supervised
classifier with an unsupervised anomaly score for robustness, then flag
episodes that cross a threshold.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import xgboost as xgb
import shap


# The columns available at real-time inference (a single API request gets
# one snapshot of mean/std per signal — not enough history to compute
# rolling slope/max, which only make sense over a batch time series).
# train.py trains on exactly these columns so the model matches what
# api.py can actually supply at prediction time.
API_FEATURES = ["EDA_mean", "EDA_std", "HR_mean", "HR_std", "TEMP_mean"]


def engineer_features(df: pd.DataFrame, window_seconds: int = 60, fs: int = 4) -> pd.DataFrame:
    """
    Roll physiological signals (EDA, HR, TEMP, ACC) into windowed features.
    Expects columns: EDA, HR, TEMP (WESAD-style, resampled to a common fs).
    """
    window = window_seconds * fs
    feats = pd.DataFrame(index=df.index)

    for col in ["EDA", "HR", "TEMP"]:
        if col not in df.columns:
            continue
        feats[f"{col}_mean"] = df[col].rolling(window, min_periods=1).mean()
        feats[f"{col}_std"] = df[col].rolling(window, min_periods=1).std()
        feats[f"{col}_slope"] = df[col].diff().rolling(window, min_periods=1).mean()
        feats[f"{col}_max"] = df[col].rolling(window, min_periods=1).max()

    return feats.bfill().fillna(0)


def flag_anomalies(
    X: pd.DataFrame,
    y: pd.Series = None,
    contamination: float = 0.1,
):
    """
    Two-track anomaly flagging:
    1. IsolationForest — unsupervised outlier score (catches novel stress
       patterns not seen in labeled training data).
    2. XGBoost classifier — supervised stress/no-stress prediction, if
       labels (y) are available (e.g. from WESAD's self-report labels).

    Returns a DataFrame with anomaly_score, stress_prob, and a combined
    `flagged` boolean — mirrors the reconciliation-style dual-check used
    in the fraud detection projects.
    """
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso_scores = iso.fit_predict(X)
    anomaly_score = iso.decision_function(X)

    result = pd.DataFrame(index=X.index)
    result["anomaly_score"] = anomaly_score
    result["iso_flag"] = iso_scores == -1

    if y is not None:
        clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42,
        )
        clf.fit(X, y)
        result["stress_prob"] = clf.predict_proba(X)[:, 1]
        result["clf_flag"] = result["stress_prob"] > 0.5
        result["flagged"] = result["iso_flag"] | result["clf_flag"]

        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X)
        result.attrs["shap_values"] = shap_values
        result.attrs["shap_features"] = X.columns.tolist()
        result.attrs["model"] = clf
    else:
        result["flagged"] = result["iso_flag"]

    return result


def top_stress_drivers(result: pd.DataFrame, row_idx: int, top_n: int = 3):
    """Return the top SHAP feature contributors for a single flagged episode."""
    if "shap_values" not in result.attrs:
        return []
    shap_values = result.attrs["shap_values"]
    features = result.attrs["shap_features"]
    contribs = list(zip(features, shap_values[row_idx]))
    contribs.sort(key=lambda x: abs(x[1]), reverse=True)
    return contribs[:top_n]