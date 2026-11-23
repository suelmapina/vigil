"""End-to-end API tests. Need the trained model (make model); skipped otherwise."""

import json

import numpy as np
import pandas as pd
import pytest

from vigil import config
from vigil.model.train import modelpath, scorespath

pytestmark = pytest.mark.skipif(not modelpath.exists(), reason="run make model first")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from vigil.api.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample():
    df = pd.read_parquet(config.baseparquet)
    test = df[df[config.monthcol].isin(config.testmonths)].reset_index(drop=True)
    rows = test.drop(columns=[config.target, config.monthcol])
    return json.loads(rows.head(200).to_json(orient="records"))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_score_shape(client, sample):
    r = client.post("/score", json=sample[0])
    assert r.status_code == 200
    body = r.json()
    assert body["action"] in ("approve", "review", "decline")
    assert 0 <= body["probability"] <= 1
    assert set(body["expectedcost"]) == {"approve", "review", "decline"}
    assert len(body["reasons"]) <= 3
    assert "x-process-ms" in r.headers


def test_reasons_for_reviews_and_on_request(client, sample):
    decisions = client.post("/score-batch", json=sample).json()
    for d in decisions:
        if d["action"] == "approve":
            assert d["reasons"] == []
        else:
            assert d["reasons"]
    approved = next(i for i, d in enumerate(decisions) if d["action"] == "approve")
    forced = client.post("/score?explain=true", json=sample[approved]).json()
    assert forced["reasons"] and forced["probability"] == decisions[approved]["probability"]


def test_api_matches_the_offline_pipeline(client, sample):
    """Same probabilities as the batch scoring used in every report (no training/serving skew)."""
    r = client.post("/score-batch", json=sample)
    online = np.array([d["probability"] for d in r.json()])
    offline = pd.read_parquet(scorespath)["probability"].head(len(sample)).to_numpy()
    assert np.allclose(online, offline, atol=1e-4)


def test_missing_as_null_or_minus_one_is_the_same(client, sample):
    item = dict(sample[0], prev_address_months_count=-1)
    a = client.post("/score", json=item).json()
    b = client.post("/score", json=dict(item, prev_address_months_count=None)).json()
    assert a["probability"] == b["probability"]


def test_age_is_ignored(client, sample):
    a = client.post("/score", json=dict(sample[1], customer_age=20)).json()
    b = client.post("/score", json=dict(sample[1], customer_age=80)).json()
    assert a == b


def test_bad_input_is_rejected(client, sample):
    assert client.post("/score", json=dict(sample[0], income=5)).status_code == 422
    assert client.post("/score", json={"income": 0.5}).status_code == 422
