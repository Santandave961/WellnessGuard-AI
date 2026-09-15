"""
Training entry point.
Expects a CSV with EDA, HR, TEMP signal columns + a `label` column
(0 = baseline, 1 = stress) — this is the format after preprocessing
WESAD (see notebooks/01_wesad_preprocessing.ipynb, to be added once
you download the dataset).

WESAD download (manual step, ~2GB):
https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/
"""

import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from anomaly_utils import engineer_features, flag_anomalies, API_FEATURES


def main(data_path: str, output_path: str):
    df = pd.read_csv(data_path, parse_dates=["timestamp"] if "timestamp" in pd.read_csv(data_path, nrows=1).columns else None)

    y = df["label"]
    raw_signals = df[[c for c in ["EDA", "HR", "TEMP"] if c in df.columns]]
    X = engineer_features(raw_signals)[API_FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    result = flag_anomalies(X_train, y_train)
    model = result.attrs["model"]

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred_prob > 0.5))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_prob):.4f}")

    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/wesad_processed.csv")
    parser.add_argument("--output", default="models/stress_clf.joblib")
    args = parser.parse_args()
    main(args.data, args.output)