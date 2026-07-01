"""
predict.py
-----------
Command-line batch scoring script - supports Scenario 2 from the brief
(a compliance officer batch-screening a list of applicants).

Usage:
    python src/predict.py --input path/to/applicants.csv --output scored.csv

The input CSV must contain the same raw columns as application_record.csv
(CODE_GENDER, FLAG_OWN_CAR, ..., DAYS_BIRTH, DAYS_EMPLOYED, ...).
The output CSV adds two columns: PRED_DECISION (APPROVED/REJECTED) and
PRED_REJECT_PROBABILITY.
"""

import argparse
import json
import os
import pickle
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import add_derived_features, encode_features  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_artifacts():
    with open(os.path.join(MODELS_DIR, "best_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "training_columns.json")) as f:
        training_columns = json.load(f)
    with open(os.path.join(MODELS_DIR, "model_meta.json")) as f:
        meta = json.load(f)
    return model, scaler, training_columns, meta


def main():
    parser = argparse.ArgumentParser(description="Batch-score applicants for credit card approval.")
    parser.add_argument("--input", required=True, help="Path to input CSV of applicants")
    parser.add_argument("--output", required=True, help="Path to write scored CSV")
    args = parser.parse_args()

    model, scaler, training_columns, meta = load_artifacts()

    df = pd.read_csv(args.input)
    engineered = add_derived_features(df)
    X = encode_features(engineered, training_columns=training_columns)

    X_input = scaler.transform(X) if meta.get("uses_scaled_input") else X

    preds = model.predict(X_input)
    probs = model.predict_proba(X_input)[:, 1]

    df["PRED_DECISION"] = ["REJECTED" if p == 1 else "APPROVED" for p in preds]
    df["PRED_REJECT_PROBABILITY"] = (probs * 100).round(1)

    df.to_csv(args.output, index=False)
    n_reject = (df["PRED_DECISION"] == "REJECTED").sum()
    print(f"Scored {len(df)} applicants -> {n_reject} flagged high-risk / rejected.")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
