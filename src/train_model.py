"""
train_model.py
----------------
Loads application_record.csv + credit_record.csv, builds features/labels,
trains four classifiers:
    - Logistic Regression
    - Random Forest
    - XGBoost (Gradient Boosting)
    - Decision Tree

Evaluates each on a held-out test set (Accuracy, Precision, Recall,
F1, ROC-AUC), saves a comparison bar chart with matplotlib, and
persists the best-performing model (by F1 score) plus the scaler and
the training column list to the `models/` folder for use by the Flask app.
"""

import os
import json
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    # Fallback so the pipeline still runs end-to-end even if the xgboost
    # package isn't installed in this environment. Install xgboost
    # (see requirements.txt) to use real XGBoost instead.
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier  # type: ignore
    XGBOOST_AVAILABLE = False

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import merge_and_label, encode_features  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def load_data():
    app_path = os.path.join(DATA_DIR, "application_record.csv")
    credit_path = os.path.join(DATA_DIR, "credit_record.csv")

    if not (os.path.exists(app_path) and os.path.exists(credit_path)):
        print("Raw data not found - generating synthetic dataset first...")
        from generate_data import generate_application_record, generate_credit_record  # type: ignore
        app_df = generate_application_record()
        credit_df = generate_credit_record(app_df)
        app_df.to_csv(app_path, index=False)
        credit_df.to_csv(credit_path, index=False)
    else:
        app_df = pd.read_csv(app_path)
        credit_df = pd.read_csv(credit_path, dtype={"STATUS": str})

    return app_df, credit_df


def main():
    print("Loading data...")
    app_df, credit_df = load_data()

    print("Merging + labeling (multi-class STATUS -> binary TARGET)...")
    merged = merge_and_label(app_df, credit_df)
    print(f"Dataset shape after merge: {merged.shape}")
    print("Target distribution:\n", merged["TARGET"].value_counts(normalize=True))

    y = merged["TARGET"]
    X = encode_features(merged)
    training_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # NOTE: TARGET==1 means "high risk" (bank would REJECT the applicant)
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "XGBoost": (
            XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.08,
                scale_pos_weight=pos_weight, eval_metric="logloss",
                random_state=42, use_label_encoder=False
            ) if XGBOOST_AVAILABLE else
            XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.08,
                random_state=42
            )
        ),
    }

    results = {}
    fitted_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        # Tree-based models don't need scaling but it doesn't hurt them;
        # Logistic Regression benefits from it.
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        results[name] = metrics
        fitted_models[name] = model

        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1       : {metrics['f1']:.4f}")
        print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
        print("  Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # ---- Pick the best model by F1 score ----
    best_name = max(results, key=lambda n: results[n]["f1"])
    best_model = fitted_models[best_name]
    print(f"\nBest model: {best_name} (F1={results[best_name]['f1']:.4f})")

    # ---- Save comparison chart ----
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metric_names))
    width = 0.2
    for i, (name, m) in enumerate(results.items()):
        values = [m[k] for k in metric_names]
        ax.bar(x + i * width, values, width, label=name)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metric_names])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison - Credit Card Approval Prediction")
    ax.legend()
    plt.tight_layout()
    chart_path = os.path.join(OUTPUTS_DIR, "model_comparison.png")
    plt.savefig(chart_path, dpi=150)
    print(f"Saved comparison chart to {chart_path}")

    # ---- Persist artifacts for the Flask app ----
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    with open(os.path.join(MODELS_DIR, "training_columns.json"), "w") as f:
        json.dump(training_columns, f)

    meta = {
        "best_model_name": best_name,
        "uses_scaled_input": best_name == "Logistic Regression",
        "metrics": results,
    }
    with open(os.path.join(MODELS_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved best model ({best_name}) and artifacts to {MODELS_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
