"""Load and clean the BAF base dataset.

Missing values are coded as -1 in some columns. They are informative (an applicant with no
previous address on file is about 4.6x more likely to be fraudulent), so they become NaN and the
models see missingness explicitly rather than as a small number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vigil import config

missingcodes = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]
constant = ["device_fraud_count"]  # always 0 in the base dataset
categorical = ["payment_type", "employment_status", "housing_status", "source", "device_os"]
protected = ["customer_age", "income", "employment_status"]
# age is kept in the data to measure fairness, never used to score: dropping it costs nothing
# (see vigil.fairness) and avoids treating applicants differently because of their age
notfeatures = [config.target, config.monthcol, "customer_age"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in constant if c in df.columns]).copy()
    for c in missingcodes:
        df[c] = df[c].where(df[c] != -1, np.nan).astype(float)
    for c in categorical:
        df[c] = df[c].astype("category")
    return df


def load() -> pd.DataFrame:
    return clean(pd.read_parquet(config.baseparquet))


def features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in notfeatures]


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """fit (months 0 to 4), valid (month 5), test (months 6 and 7)."""
    month = df[config.monthcol]
    fit = df[month.isin(config.fitmonths)]
    valid = df[month == config.validmonth]
    test = df[month.isin(config.testmonths)]
    return fit, valid, test


def older(df: pd.DataFrame) -> np.ndarray:
    return (df["customer_age"] >= config.agegroup).to_numpy()
