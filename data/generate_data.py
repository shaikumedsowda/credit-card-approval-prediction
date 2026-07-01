"""
generate_data.py
-----------------
Generates synthetic 'application_record.csv' and 'credit_record.csv' files
that mirror the schema of the well-known Kaggle "Credit Card Approval
Prediction" dataset (rikdifos/credit-card-approval-prediction).

If you have the real Kaggle dataset, simply drop your own
application_record.csv and credit_record.csv into this `data/` folder
(same column names) and skip running this script - the rest of the
pipeline will work unchanged.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N_APPLICANTS = 6000
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_application_record(n=N_APPLICANTS):
    ids = np.arange(5000000, 5000000 + n)

    gender = np.random.choice(["M", "F"], size=n, p=[0.45, 0.55])
    own_car = np.random.choice(["Y", "N"], size=n, p=[0.4, 0.6])
    own_realty = np.random.choice(["Y", "N"], size=n, p=[0.6, 0.4])
    cnt_children = np.random.choice([0, 1, 2, 3, 4], size=n,
                                     p=[0.55, 0.22, 0.15, 0.06, 0.02])

    income_type = np.random.choice(
        ["Working", "Commercial associate", "Pensioner", "State servant", "Student"],
        size=n, p=[0.5, 0.23, 0.17, 0.09, 0.01]
    )
    education_type = np.random.choice(
        ["Secondary / secondary special", "Higher education",
         "Incomplete higher", "Lower secondary", "Academic degree"],
        size=n, p=[0.65, 0.25, 0.06, 0.03, 0.01]
    )
    family_status = np.random.choice(
        ["Married", "Single / not married", "Civil marriage",
         "Separated", "Widow"],
        size=n, p=[0.6, 0.18, 0.1, 0.08, 0.04]
    )
    housing_type = np.random.choice(
        ["House / apartment", "With parents", "Municipal apartment",
         "Rented apartment", "Office apartment", "Co-op apartment"],
        size=n, p=[0.78, 0.09, 0.06, 0.04, 0.02, 0.01]
    )
    occupation_type = np.random.choice(
        ["Laborers", "Core staff", "Sales staff", "Managers", "Drivers",
         "High skill tech staff", "Accountants", "Medicine staff",
         "Cooking staff", "Security staff"],
        size=n
    )

    # Income: log-normal, adjusted a bit by income type
    base_income = np.random.lognormal(mean=11.2, sigma=0.45, size=n)
    income_bump = pd.Series(income_type).map({
        "Working": 1.0, "Commercial associate": 1.25, "Pensioner": 0.6,
        "State servant": 1.05, "Student": 0.3
    }).values
    amt_income_total = np.round(base_income * income_bump, -2)

    # DAYS_BIRTH: negative = days before today. Age range ~21-70
    age_years = np.random.randint(21, 70, size=n)
    days_birth = -(age_years * 365 + np.random.randint(0, 365, size=n))

    # DAYS_EMPLOYED: negative = employed X days ago. Positive huge number = pensioner (not employed)
    days_employed = np.where(
        income_type == "Pensioner",
        365243,  # Kaggle dataset's real sentinel value for "not employed"
        -np.random.randint(30, 365 * 40, size=n)
    )

    flag_mobil = np.ones(n, dtype=int)
    flag_work_phone = np.random.choice([0, 1], size=n, p=[0.75, 0.25])
    flag_phone = np.random.choice([0, 1], size=n, p=[0.6, 0.4])
    flag_email = np.random.choice([0, 1], size=n, p=[0.8, 0.2])

    cnt_fam_members = cnt_children + np.where(
        np.isin(family_status, ["Married", "Civil marriage"]), 2, 1
    )

    df = pd.DataFrame({
        "ID": ids,
        "CODE_GENDER": gender,
        "FLAG_OWN_CAR": own_car,
        "FLAG_OWN_REALTY": own_realty,
        "CNT_CHILDREN": cnt_children,
        "AMT_INCOME_TOTAL": amt_income_total,
        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": education_type,
        "NAME_FAMILY_STATUS": family_status,
        "NAME_HOUSING_TYPE": housing_type,
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "FLAG_MOBIL": flag_mobil,
        "FLAG_WORK_PHONE": flag_work_phone,
        "FLAG_PHONE": flag_phone,
        "FLAG_EMAIL": flag_email,
        "OCCUPATION_TYPE": occupation_type,
        "CNT_FAM_MEMBERS": cnt_fam_members,
    })
    return df


def generate_credit_record(app_df):
    """
    For each applicant, generate a history of MONTHS_BALANCE / STATUS rows.
    STATUS codes (as in the real Kaggle dataset):
      0: 1-29 days past due
      1: 30-59 days past due
      2: 60-89 days past due
      3: 90-119 days past due
      4: 120-149 days past due
      5: overdue more than 150 days / bad debt
      C: paid off that month
      X: no loan for that month
    We bias "risk" by income and employment status so the model has
    real signal to learn from.
    """
    rows = []
    for _, r in app_df.iterrows():
        months = np.random.randint(6, 36)  # length of credit history
        # base risk: lower income & unemployed (pensioner sentinel) -> higher risk
        risk_score = 0.15
        if r["AMT_INCOME_TOTAL"] < 90000:
            risk_score += 0.15
        if r["DAYS_EMPLOYED"] > 0:  # pensioner / not currently employed
            risk_score += 0.05
        if r["NAME_INCOME_TYPE"] == "Student":
            risk_score += 0.1
        if r["CNT_CHILDREN"] >= 3:
            risk_score += 0.05

        for m in range(months):
            u = np.random.rand()
            if u < risk_score * 0.4:
                status = np.random.choice(["2", "3", "4", "5"])
            elif u < risk_score:
                status = np.random.choice(["0", "1"])
            elif u < risk_score + 0.35:
                status = "C"
            else:
                status = "X"
            rows.append((r["ID"], -m, status))

    return pd.DataFrame(rows, columns=["ID", "MONTHS_BALANCE", "STATUS"])


if __name__ == "__main__":
    app_df = generate_application_record()
    credit_df = generate_credit_record(app_df)

    app_path = os.path.join(OUT_DIR, "application_record.csv")
    credit_path = os.path.join(OUT_DIR, "credit_record.csv")

    app_df.to_csv(app_path, index=False)
    credit_df.to_csv(credit_path, index=False)

    print(f"Saved {len(app_df)} applicant records to {app_path}")
    print(f"Saved {len(credit_df)} credit history rows to {credit_path}")
