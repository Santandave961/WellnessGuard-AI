"""
Generates a small synthetic dataset in the same schema as
preprocess_wesad.py's output, so you can smoke-test train.py end-to-end
before you've downloaded/preprocessed real WESAD data.

This is NOT for evaluating real model performance — the labels are
generated from a simple rule, not real physiology. Swap for real WESAD
data (via preprocess_wesad.py) before drawing any conclusions or citing
metrics anywhere.

Usage:
    python src/make_synthetic_data.py --output data/wesad_processed.csv
"""

import argparse
import os

import numpy as np
import pandas as pd


def main(output_path: str, n: int = 5000, seed: int = 42):
    rng = np.random.default_rng(seed)

    timestamp = pd.date_range("2024-01-01", periods=n, freq="250ms")
    eda = 2 + rng.normal(0, 0.3, n).cumsum() * 0.01
    temp = 33 + rng.normal(0, 0.3, n)
    hr = 70 + rng.normal(0, 5, n).cumsum() * 0.02

    # Simple synthetic rule: "stress" when HR and EDA are both elevated
    stress_signal = (hr > np.percentile(hr, 60)) & (eda > np.percentile(eda, 60))
    label = stress_signal.astype(int)

    df = pd.DataFrame({
        "timestamp": timestamp,
        "EDA": eda,
        "TEMP": temp,
        "HR": hr,
        "subject": "synthetic",
        "label": label,
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} synthetic rows to {output_path}")
    print(f"Stress ratio: {df['label'].mean():.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/wesad_processed.csv")
    parser.add_argument("--n", type=int, default=5000)
    args = parser.parse_args()
    main(args.output, args.n)