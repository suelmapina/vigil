"""Project-wide paths and constants."""

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
trainmonths = range(0, 6)  # first six months, as in the paper
testmonths = range(6, 8)  # last two months are never seen in training
