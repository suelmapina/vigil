"""Project-wide paths and constants."""

import os
from pathlib import Path

root = Path(__file__).resolve().parents[2]
datadir = root / "data"
rawdir = datadir / "raw"
processeddir = datadir / "processed"
modelsdir = root / "models"
reportsdir = root / "reports"

seed = 42

# Bank Account Fraud dataset suite (Feedzai, NeurIPS 2022), CC BY-NC-SA 4.0
# https://github.com/feedzai/bank-account-fraud
kagglehandle = "sgpjesus/bank-account-fraud-dataset-neurips-2022"
basecsv = "Base.csv"
baseparquet = processeddir / "baf_base.parquet"

target = "fraud_bool"
monthcol = "month"

# temporal split, as in production: fit on the oldest months, tune on the most recent one,
# report on months the model has never seen (same 6 + 2 split as the paper)
fitmonths = range(0, 5)
validmonth = 5
testmonths = range(6, 8)
trainmonths = range(0, 6)  # fit + valid

agegroup = 50  # fairness groups: applicants under / over 50, as in the paper
targetfpr = 0.05  # the paper's operating point: recall at 5% false positive rate

mlflowuri = f"sqlite:///{root / 'mlflow.db'}"

# artefacts shared by the pipeline, the API and the demo
modelpath = Path(os.environ.get("VIGIL_MODEL", modelsdir / "model.joblib"))
scorespath = processeddir / "test_scores.parquet"
validpath = processeddir / "valid_scores.parquet"
