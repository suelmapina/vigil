"""Cost-based decision engine."""

import numpy as np
import pytest

from vigil.decision import engine


def test_low_risk_is_approved_and_near_certain_fraud_is_declined():
    choice = engine.decide(np.array([0.001, 0.99]), np.array([500, 500]), capacity=1.0)
    assert list(choice) == [engine.approve, engine.decline]


def test_larger_credit_line_is_reviewed_at_lower_risk():
    p = np.array([0.03, 0.03])
    choice = engine.decide(p, np.array([200, 2000]), capacity=1.0)
    assert list(choice) == [engine.approve, engine.review]


def test_decision_is_the_cheapest_expected_action_without_capacity_limit():
    rng = np.random.default_rng(0)
    p, limit = rng.random(1000) ** 4, rng.choice([200, 500, 1000, 2000], 1000)
    choice = engine.decide(p, limit, capacity=1.0)
    e = engine.expected_costs(p, limit)
    assert np.allclose(e[np.arange(1000), choice], e.min(axis=1))


def test_reviews_never_exceed_capacity_and_go_to_the_biggest_savings():
    rng = np.random.default_rng(1)
    p, limit = rng.uniform(0.05, 0.3, 1000), np.full(1000, 2000.0)
    choice = engine.decide(p, limit, capacity=0.01)
    assert (choice == engine.review).sum() <= 10
    e = engine.expected_costs(p, limit)
    reviewed = choice == engine.review
    fallback = np.minimum(e[:, engine.approve], e[:, engine.decline])
    saving = fallback - e[:, engine.review]
    assert saving[reviewed].min() >= saving[~reviewed & (saving > 0)].max()


def test_realised_cost_by_outcome():
    y = np.array([1, 0, 1, 0, 1, 0])
    limit = np.full(6, 1000.0)
    choice = np.array([0, 0, 1, 1, 2, 2])
    cost = engine.realised(y, limit, choice)
    c = engine.defaultcosts
    assert cost[0] == 1000 * c["drawshare"]
    assert cost[1] == 0
    assert cost[2] == pytest.approx(c["reviewcost"] + 1000 * (1 - c["catchrate"]))
    assert cost[3] == c["reviewcost"] + c["friction"]
    assert cost[4] == 0
    assert cost[5] == c["lostcustomer"]


def test_outcome_counts():
    y = np.array([1, 0, 1, 0])
    choice = np.array([2, 2, 1, 0])
    out = engine.outcome(y, np.full(4, 500.0), choice)
    assert out["declined"] == 2 and out["good_declined"] == 1 and out["reviews"] == 1
    assert out["fraud_stopped"] == pytest.approx((1 + engine.defaultcosts["catchrate"]) / 2)
