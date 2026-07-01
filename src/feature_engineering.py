"""
feature_engineering.py
-----------------------
Shared feature-engineering logic used both at training time and at
prediction time inside the Flask app, so the two never drift apart.

Key steps:
1. Merge application_record.csv with credit_record.csv on ID.
2. Convert the multi-class STATUS payment codes (0-5, C, X) from
   credit_record.csv into a binary TARGET label:
     - STATUS in {2,3,4,5}  -> 1 (high risk / past-due -> "reject")
     - STATUS in {0,1,C,X}  -> 0 (low risk -> "approve")
   An applicant is labeled high-risk (1) if ANY of their historical
   months shows a status of 2+ (60+ days past due).
3. Engineer derived features (AGE_YEARS, YEARS_EMPLOYED, INCOME_PER_FAM
   etc.) from the raw application fields.
4. One-hot encode categorical columns.
"""

import numpy as np
import pandas as pd

CATEGORICAL_COLS = [
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
]

NUMERIC_COLS = [
    "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AGE_YEARS", "YEARS_EMPLOYED",
    "FLAG_MOBIL", "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL",
    "CNT_FAM_MEMBERS", "INCOME_PER_FAM_MEMBER", "IS_PENSIONER",
]

HIGH_RISK_STATUS_CODES = {"2", "3", "4", "5"}


def build_target(credit_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse multi-class STATUS codes into a binary TARGET per ID."""
    credit_df = credit_df.copy()
    credit_df["STATUS"] = credit_df["STATUS"].astype(str)
    credit_df["IS_HIGH_RISK_MONTH"] = credit_df["STATUS"].isin(HIGH_RISK_STATUS_CODES).astype(int)

    # An applicant is only labeled high-risk (TARGET=1) if they had 2 or
    # more months of 60+ days past-due status, rather than just one -
    # this keeps the class balance closer to real-world approval rates
    # (most applicants get approved; a minority are flagged high-risk).
    high_risk_months = credit_df.groupby("ID")["IS_HIGH_RISK_MONTH"].sum()
    target = (
        (high_risk_months >= 2)
        .astype(int)
        .reset_index()
        .rename(columns={"IS_HIGH_RISK_MONTH": "TARGET"})
    )
    # TARGET = 1 -> high risk -> would be REJECTED
    # TARGET = 0 -> low risk  -> would be APPROVED
    return target


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365.25).round(1)

    # DAYS_EMPLOYED sentinel (365243) means "not currently employed" (pensioners)
    df["IS_PENSIONER"] = (df["DAYS_EMPLOYED"] > 0).astype(int)
    years_employed = (-df["DAYS_EMPLOYED"] / 365.25).round(1)
    years_employed[df["IS_PENSIONER"] == 1] = 0
    df["YEARS_EMPLOYED"] = years_employed

    df["CNT_FAM_MEMBERS"] = df["CNT_FAM_MEMBERS"].fillna(1).clip(lower=1)
    df["INCOME_PER_FAM_MEMBER"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]

    return df


def merge_and_label(app_df: pd.DataFrame, credit_df: pd.DataFrame) -> pd.DataFrame:
    target = build_target(credit_df)
    merged = app_df.merge(target, on="ID", how="inner")
    merged = add_derived_features(merged)
    return merged


def encode_features(df: pd.DataFrame, training_columns=None):
    """
    One-hot encode categorical columns. If `training_columns` is provided
    (a list of column names from the training set), align the resulting
    dataframe to exactly those columns (fills missing dummy columns with 0
    and drops any extras) - this is essential for consistent predictions
    on new, single-row applicant data in the Flask app.
    """
    df = df.copy()
    keep_cols = [c for c in NUMERIC_COLS + CATEGORICAL_COLS if c in df.columns]
    df = df[keep_cols]

    df = pd.get_dummies(df, columns=[c for c in CATEGORICAL_COLS if c in df.columns])

    if training_columns is not None:
        for col in training_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[training_columns]

    return df
