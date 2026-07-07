Flipkart GridLock Hackathon 2.0 — Traffic Demand Forecasting
Leaderboard Score: 91.39

Problem

Forecast real-world traffic/ride demand using historical geospatial and temporal signals, under strict data-leakage constraints (train/test split by day).

Approach

Feature Engineering: per-geohash missing value imputation (train-only statistics, zero leakage), time-based features (day, hour, minute, time buckets), and ordinal encodings for categorical signals.
Target Encoding: out-of-fold (OOF) 5-fold encoding on training data to prevent target leakage, with full-train encoding applied to validation/test.
Modeling: stacked ensemble of LightGBM, XGBoost, and CatBoost, each independently hyperparameter-tuned via Optuna (80/conditional/50 trial budgets respectively, tuned for validation R² stability).
Ensembling: weighted blend of the three models' predictions, with blend weights selected via validation performance rather than fixed averaging.
Validation Strategy: temporal split (day48 → train, day49 → validation) to simulate real-world forecasting rather than random shuffling, which would leak future information into training.


Tech Stack

Python · LightGBM · XGBoost · CatBoost · Optuna · Pandas · NumPy

Key Result

Achieved a 91.39 leaderboard score on real-world traffic demand forecasting by combining careful leakage-safe feature engineering with an Optuna-tuned stacked ensemble.

Notebook

See Gridlock_Hackathon.ipynb for the full pipeline — from data loading through final submission generation.
