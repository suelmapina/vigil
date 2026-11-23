"""Reason codes and fairness helpers."""

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from vigil import explain, fairness
from vigil.decision import engine


@pytest.fixture(scope="module")
def bundle():
    rng = np.random.default_rng(0)
    x = pd.DataFrame(
        {"income": rng.choice([0.1, 0.5, 0.9], 2000), "name_email_similarity": rng.random(2000)}
    )
    y = (x["name_email_similarity"] < 0.2) & (rng.random(2000) < 0.7)
    model = lgb.LGBMClassifier(n_estimators=30, verbose=-1).fit(x, y)
    return {"model": model, "features": list(x.columns)}, x


def test_contributions_add_up_to_the_raw_score(bundle):
    b, x = bundle
    shap = explain.contributions(b, x.head(50))
    full = b["model"].booster_.predict(x.head(50), pred_contrib=True)
    raw = b["model"].booster_.predict(x.head(50), raw_score=True)
    assert np.allclose(shap.sum(axis=1) + full[:, -1], raw)


def test_reasons_point_at_the_driver(bundle):
    b, x = bundle
    risky = x[x["name_email_similarity"] < 0.05].head(5)
    for r in explain.reasons(b, risky, k=1):
        assert r[0]["feature"] == "name_email_similarity"
        assert r[0]["impact"] > 0


def test_describe_is_readable():
    assert explain.describe("income", 0.9) == "Income decile: 9 of 10"
    assert explain.describe("prev_address_months_count", np.nan).endswith("not provided")
    assert explain.describe("proposed_credit_limit", 2000.0) == "Proposed credit limit: 2000"


def test_burden_by_group():
    y = np.array([0, 0, 0, 0, 1, 1])
    older = np.array([True, True, False, False, True, False])
    choice = np.array(
        [
            engine.review,
            engine.approve,
            engine.approve,
            engine.approve,
            engine.decline,
            engine.approve,
        ]
    )
    b = fairness.burden(y, older, choice)
    assert b["over50"]["good_stopped"] == 0.5
    assert b["under50"]["good_stopped"] == 0.0
    assert b["over50"]["fraud_stopped"] == 1.0
    assert b["ratio"] == 0.0


def test_scaled_only_touches_the_older_group():
    p = np.array([0.2, 0.2])
    assert list(fairness.scaled(p, np.array([True, False]), 0.5)) == [0.1, 0.2]
