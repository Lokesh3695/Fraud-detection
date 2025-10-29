
# Fraud-detection

A self-contained example project for synthetic credit-card transaction fraud detection. This repository includes code to generate synthetic transaction data, preprocess it, train multiple detection models (Random Forest, Logistic Regression, Isolation Forest), evaluate performance, and test new transactions. The goal is to provide a clear, reproducible starting point for experimenting with fraud detection ideas and pipelines.

## ✅ Highlights
- Synthetic dataset generator with realistic-ish normal and fraudulent transaction patterns.
- Data preprocessing: label encoding, feature engineering, log transformation, scaling.
- Handling class imbalance using SMOTE.
- Models trained: Random Forest, Logistic Regression, Isolation Forest (unsupervised).
- Evaluation: classification report, confusion matrix, ROC/AUC and feature importance.
- Example script to score new transactions.

---

## Table of contents
- [Quick start](#quick-start)
- [What’s included](#whats-included)
- [How it works (high level)](#how-it-works-high-level)
- [Usage](#usage)
- [Interpreting results](#interpreting-results)
- [Improving the model](#improving-the-model)
- [Reproducibility notes](#reproducibility-notes)
- [Contributing](#contributing)
- [License & Contact](#license--contact)

---

## Quick start

1. Clone the repo:
   git clone https://github.com/Lokesh3695/Fraud-detection.git
2. Create and activate a virtual environment (recommended):
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate      # Windows
3. Install dependencies:
   pip install -r requirements.txt
4. Run the main script or notebook (example):
   python fraud_detection.py

(If you don't yet have a `requirements.txt`, see the Requirements section below and install packages manually.)

---

## What's included
- fraud_detection.py (or notebook): script that:
  - generates synthetic transactions using generate_fraud_data(...)
  - preprocesses and engineers features
  - splits data, applies SMOTE, scales features
  - trains Random Forest, Logistic Regression, Isolation Forest
  - evaluates performance and plots ROC & feature importance
  - demonstrates predictions for a few new sample transactions
- README.md (this file)
- (Optional) example assets / notebooks / sample outputs (add as you expand)

---

## How it works (high level)
1. Data generation:
   - Normal transactions (90%) and fraud (10%) with different distributions for amount, time, location risk, velocity, etc.
2. Preprocessing:
   - Label encoding of categorical variables (merchant_category).
   - Feature engineering: log(amount), is_night, high_risk_location, high_velocity, etc.
3. Train / test split with stratification to preserve class ratio.
4. Apply SMOTE to training set to reduce class imbalance.
5. Standard scaling of features.
6. Train models:
   - Random Forest (supervised)
   - Logistic Regression (supervised)
   - Isolation Forest (unsupervised anomaly detection)
7. Evaluate with classification reports, confusion matrices, ROC/AUC curves, and feature importance.

---

## Usage

Example (high-level steps in script):

- Generate data:
  df = generate_fraud_data(n_samples=5000)

- Preprocess:
  - Encode merchant categories
  - Add amount_log, is_night, high_risk_location, high_velocity

- Split:
  X_train, X_test = train_test_split(..., stratify=y)

- Balance (SMOTE):
  X_train_balanced, y_train_balanced = smote.fit_resample(...)

- Scale:
  scaler.fit_transform(X_train_balanced)

- Train:
  rf_model.fit(X_train_scaled, y_train_balanced)
  lr_model.fit(X_train_scaled, y_train_balanced)
  iso_model.fit(X_train_scaled)

- Evaluate:
  - classification_report(y_test, preds)
  - roc_auc_score and roc_curve for ROC plots
  - feature_importances_ for Random Forest

- Score new transactions:
  - Preprocess new transactions the same way
  - Use scaler.transform and model.predict / predict_proba

---

## Interpreting results
- AUC (ROC) is used to compare classifiers' ability to rank fraud vs non-fraud.
- Precision / recall and confusion matrix reveal trade-offs (especially important in fraud detection — false positives vs false negatives).
- Feature importance from Random Forest gives an indication of which engineered features the model relied on most (use with caution — correlated features and synthetic data can mislead).

---

## Improving the model (recommended next steps)
1. Replace or augment synthetic data with real transaction data (properly anonymized and privacy-compliant).
2. Add more features:
   - Device ID, IP location, velocity across accounts, merchant risk scores, historical user patterns.
3. Use time-series / sequential models for user-level/session-level patterns (RNNs/transformers or aggregated features).
4. Tune hyperparameters (GridSearchCV / RandomizedSearchCV) and add cross-validation with time-aware splits.
5. Use calibrated probability outputs (calibration or isotonic regression) for actionable thresholds.
6. Deploy real-time scoring with streaming (Kafka / serverless) and add monitoring/alerting for model drift.
7. Use more robust anomaly detection ensembles or one-class classifiers for unseen fraud patterns.

---

## Requirements
Minimum packages used in the script:
- python >= 3.8
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn

Suggested install:
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn

(If you want, I can add a requirements.txt file to the repo.)

---

## Reproducibility
- The provided code uses np.random.seed(42) to make the synthetic data reproducible. If you change seeds or sampling, results will differ.
- When using real data, ensure stable preprocessing pipelines (fit transformers only on training data) and save scalers/encoders for inference.

---

## Files & structure (suggested)
- fraud_detection.py           # main script demonstrating the full pipeline
- README.md                    # project overview and instructions
- requirements.txt             # pinned dependency list (optional)
- notebooks/analysis.ipynb      # interactive exploration (optional)
- data/                        # place to store datasets (if added later)

---

## Contributing
Feel free to open issues or PRs to:
- Add a requirements.txt and CI
- Improve the synthetic data generator
- Add real-world dataset connectors or example notebooks
- Improve evaluation and visualization

When contributing, please include tests or example notebook outputs.

---

## License
Add your preferred license (e.g., MIT). If you want, I can add a LICENSE file.

---

## Contact
Maintainer: Lokesh3695

If you'd like, I can:
- create a requirements.txt with pinned versions,
- add a Jupyter notebook that walks through the steps interactively,
- or convert the script into a lightweight package with entry points for training and scoring.
