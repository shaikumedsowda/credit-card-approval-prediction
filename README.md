# Credit Card Approval Prediction

Automates credit card approval/rejection decisions using machine learning,
trained on historical applicant data (income, employment, credit history,
demographics). Four classifiers are trained and compared — **Logistic
Regression**, **Random Forest**, **XGBoost**, and **Decision Tree** — and
the best performer is deployed behind a Flask web app for real-time
predictions. An IBM Watson Machine Learning deployment script is included
for cloud hosting.

## Project structure

```
credit-card-approval-prediction/
├── data/
│   ├── generate_data.py        # synthetic dataset generator (Kaggle-schema compatible)
│   ├── application_record.csv  # generated on first run
│   └── credit_record.csv       # generated on first run
├── src/
│   ├── feature_engineering.py  # shared feature engineering (training + inference)
│   ├── train_model.py          # trains & compares all 4 models, saves the best
│   └── predict.py              # CLI batch scoring (compliance / bulk screening)
├── app/
│   ├── app.py                  # Flask web application
│   ├── templates/
│   │   ├── index.html          # applicant intake form
│   │   └── result.html         # decision + confidence gauge
│   └── static/style.css
├── models/                     # best_model.pkl, scaler.pkl, etc. (created by training)
├── outputs/                    # model_comparison.png (created by training)
├── deployment/
│   └── watson_deploy.py        # IBM Watson ML deployment pipeline
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Generate data (optional — auto-runs on first training if skipped)

No real dataset is bundled with this zip. Running the training script will
automatically generate a realistic **synthetic** dataset that mirrors the
schema of the well-known Kaggle "Credit Card Approval Prediction" dataset
(`application_record.csv` + `credit_record.csv`).

**To use your own real data instead**, just drop your own
`application_record.csv` and `credit_record.csv` (same column names) into
the `data/` folder before training — the rest of the pipeline is unchanged.

```bash
python data/generate_data.py
```

## 2. Train the models

```bash
python src/train_model.py
```

This will:
- Merge the application + credit history data
- Convert the multi-class `STATUS` payment codes (0–5, C, X) into a binary
  `TARGET` label (2+ months of 60-day-past-due history → high risk)
- Train Logistic Regression, Decision Tree, Random Forest, and XGBoost
- Print Accuracy / Precision / Recall / F1 / ROC-AUC for each
- Save a comparison bar chart to `outputs/model_comparison.png`
- Save the **best model by F1 score** + scaler + training columns to `models/`

## 3. Run the web app

```bash
python app/app.py
```

Then open **http://127.0.0.1:5000** in your browser, fill in an applicant's
profile, and submit to get an instant APPROVED / REJECTED decision with the
model's confidence.

## 4. Batch scoring (compliance / bulk screening use case)

```bash
python src/predict.py --input path/to/applicants.csv --output scored.csv
```

Adds `PRED_DECISION` and `PRED_REJECT_PROBABILITY` columns to your CSV.

## 5. (Optional) Deploy to IBM Watson Machine Learning

```bash
export WML_API_KEY=your-ibm-cloud-api-key
export WML_URL=https://us-south.ml.cloud.ibm.com
export WML_SPACE_ID=your-deployment-space-id
pip install ibm-watson-machine-learning
python deployment/watson_deploy.py
```

This publishes `models/best_model.pkl` to a Watson ML deployment space and
prints a scoring REST endpoint you can call from anywhere.

## Notes on the synthetic dataset

Since no real dataset was provided, `data/generate_data.py` builds a
synthetic dataset with the same structure as the real Kaggle dataset,
including realistic distributions for income, age, employment, family
status, and a credit-history simulator with built-in risk correlations
(lower income / no current employment / more dependents → higher chance of
past-due months) so the models have genuine signal to learn from. Swap in
real `application_record.csv` / `credit_record.csv` files at any time —
no code changes required.

## Publishing to GitHub

```bash
git init
git add .
git commit -m "Credit card approval prediction - ML pipeline + Flask app"
git branch -M main
git remote add origin <your-empty-github-repo-url>
git push -u origin main
```

Then paste your repo URL into the **GitHub** field, and your local
`http://127.0.0.1:5000` (or a deployed URL, e.g. via Render/Heroku/PythonAnywhere)
into the **Demo** field on your workspace.
