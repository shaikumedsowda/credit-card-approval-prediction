"""
app.py
-------
Flask web application for Credit Card Approval Prediction.

Run with:
    python app/app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import os
import sys
import json
import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

from feature_engineering import add_derived_features, encode_features  # noqa: E402

app = Flask(__name__)

# ---- Load trained artifacts at startup ----
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
COLUMNS_PATH = os.path.join(MODELS_DIR, "training_columns.json")
META_PATH = os.path.join(MODELS_DIR, "model_meta.json")

model = None
scaler = None
training_columns = None
model_meta = {}

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(COLUMNS_PATH, "r") as f:
        training_columns = json.load(f)
    if os.path.exists(META_PATH):
        with open(META_PATH, "r") as f:
            model_meta = json.load(f)

FORM_FIELDS = {
    "CODE_GENDER": ["M", "F"],
    "FLAG_OWN_CAR": ["Y", "N"],
    "FLAG_OWN_REALTY": ["Y", "N"],
    "NAME_INCOME_TYPE": ["Working", "Commercial associate", "Pensioner", "State servant", "Student"],
    "NAME_EDUCATION_TYPE": ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"],
    "NAME_FAMILY_STATUS": ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
    "NAME_HOUSING_TYPE": ["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment", "Co-op apartment"],
    "OCCUPATION_TYPE": ["Laborers", "Core staff", "Sales staff", "Managers", "Drivers", "High skill tech staff", "Accountants", "Medicine staff", "Cooking staff", "Security staff"],
}


def build_input_row(form):
    """Convert raw HTML form data into a single-row engineered dataframe."""
    age_years = float(form["age_years"])
    years_employed = float(form["years_employed"])
    is_pensioner = 1 if form["NAME_INCOME_TYPE"] == "Pensioner" else 0

    raw = {
        "CODE_GENDER": form["CODE_GENDER"],
        "FLAG_OWN_CAR": form["FLAG_OWN_CAR"],
        "FLAG_OWN_REALTY": form["FLAG_OWN_REALTY"],
        "CNT_CHILDREN": int(form["CNT_CHILDREN"]),
        "AMT_INCOME_TOTAL": float(form["AMT_INCOME_TOTAL"]),
        "NAME_INCOME_TYPE": form["NAME_INCOME_TYPE"],
        "NAME_EDUCATION_TYPE": form["NAME_EDUCATION_TYPE"],
        "NAME_FAMILY_STATUS": form["NAME_FAMILY_STATUS"],
        "NAME_HOUSING_TYPE": form["NAME_HOUSING_TYPE"],
        "DAYS_BIRTH": -int(age_years * 365.25),
        "DAYS_EMPLOYED": 365243 if is_pensioner else -int(years_employed * 365.25),
        "FLAG_MOBIL": 1,
        "FLAG_WORK_PHONE": int(form.get("FLAG_WORK_PHONE", 0)),
        "FLAG_PHONE": int(form.get("FLAG_PHONE", 0)),
        "FLAG_EMAIL": int(form.get("FLAG_EMAIL", 0)),
        "OCCUPATION_TYPE": form["OCCUPATION_TYPE"],
        "CNT_FAM_MEMBERS": int(form["CNT_FAM_MEMBERS"]),
    }
    df = pd.DataFrame([raw])
    df = add_derived_features(df)
    return df


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", fields=FORM_FIELDS, model_ready=model is not None)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template(
            "result.html",
            error="No trained model found. Run `python src/train_model.py` first.",
        )

    try:
        row = build_input_row(request.form)
        X = encode_features(row, training_columns=training_columns)

        if model_meta.get("uses_scaled_input"):
            X_input = scaler.transform(X)
        else:
            X_input = X

        pred = int(model.predict(X_input)[0])
        proba = float(model.predict_proba(X_input)[0][1])  # P(high risk / reject)

        decision = "REJECTED" if pred == 1 else "APPROVED"
        confidence = proba if pred == 1 else 1 - proba

        return render_template(
            "result.html",
            decision=decision,
            confidence=round(confidence * 100, 1),
            reject_probability=round(proba * 100, 1),
            model_name=model_meta.get("best_model_name", "Unknown"),
        )
    except Exception as e:
        return render_template("result.html", error=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
