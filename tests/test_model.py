"""Data cleaning, drift and metric functions on small synthetic inputs."""

import numpy as np
import pandas as pd
import pytest

from vigil import config, drift
from vigil.data import prepare
from vigil.model import metrics


def raw(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.integers(-1, 30, n) for c in prepare.missingcodes})
    for c in prepare.categorical:
        df[c] = rng.choice(["AA", "AB"], n)
    df["device_fraud_count"] = 0
    df["customer_age"] = rng.choice([20, 30, 50, 60], n)
    df[config.monthcol] = rng.integers(0, 8, n)
    df[config.target] = rng.integers(0, 2, n)
    return df


def test_clean_turns_missing_codes_into_nan():
    df = raw()
    out = prepare.clean(df)
    for c in prepare.missingcodes:
        assert (out[c] == -1).sum() == 0
        assert out[c].isna().sum() == (df[c] == -1).sum()
    assert "device_fraud_count" not in out
    assert all(isinstance(out[c].dtype, pd.CategoricalDtype) for c in prepare.categorical)


def test_split_is_temporal_and_disjoint():
    fit, valid, test = prepare.split(prepare.clean(raw()))
    assert fit[config.monthcol].max() < valid[config.monthcol].min()
    assert valid[config.monthcol].max() < test[config.monthcol].min()
    assert set(fit.index).isdisjoint(test.index)


def test_psi_is_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(size=20000))
    assert drift.psi(a, pd.Series(rng.normal(size=20000))) < 0.01
    assert drift.psi(a, pd.Series(rng.normal(1.5, 1, size=20000))) > 0.25


def test_psi_handles_categories_and_missing():
    a = pd.Series(["x", "y", None] * 100, dtype="category")
    b = pd.Series(["x"] * 300, dtype="category")
    assert drift.psi(a, a) == pytest.approx(0, abs=1e-9)
    assert drift.psi(a, b) > 0.25


def test_threshold_hits_the_target_fpr():
    rng = np.random.default_rng(2)
    y = rng.random(50000) < 0.02
    score = rng.random(50000) + y
    threshold = metrics.threshold_at_fpr(y, score, 0.05)
    recall, fpr = metrics.rates(y, score, threshold)
    assert fpr == pytest.approx(0.05, abs=0.002)
    assert recall > 0.9


def test_recall_at_fpr_extremes():
    y = np.array([0] * 95 + [1] * 5)
    perfect = y.astype(float)
    assert metrics.recall_at_fpr(y, perfect) == 1.0
    assert metrics.recall_at_fpr(y, -perfect) == 0.0


def test_fpr_ratio_is_one_when_groups_are_treated_alike():
    rng = np.random.default_rng(3)
    y = np.zeros(20000, dtype=int)
    score = rng.random(20000)
    group = rng.random(20000) < 0.3
    assert metrics.fpr_ratio(y, score, group, 0.9) == pytest.approx(1, abs=0.1)
    assert metrics.fpr_ratio(y, score + group, group, 0.9) < 0.2


def test_bootstrap_reference_difference_is_zero():
    rng = np.random.default_rng(4)
    y = rng.random(3000) < 0.05
    out = metrics.bootstrap(y, {"a": rng.random(3000) + y, "b": rng.random(3000)}, "b", reps=20)
    assert out["b"]["pr_auc"]["diff"] == 0
    assert out["a"]["recall_at_5fpr"]["diff_low"] > 0
