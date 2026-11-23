"""Cost-based three-way decision: approve, send to an analyst, or decline.

For an application with fraud probability p and proposed credit limit L, the expected cost of each
action is

    approve  p × L × drawshare
    review   reviewcost + p × L × drawshare × (1 − catchrate) + (1 − p) × friction
    decline  (1 − p) × lostcustomer

and the engine picks the cheapest. Analysts are a fixed resource: when more applications would be
worth reviewing than there are analyst hours, the ones where a review saves the most go first and
the rest fall back to the better of approve or decline.
"""

from __future__ import annotations

import numpy as np

approve, review, decline = 0, 1, 2
actions = ["approve", "review", "decline"]

defaultcosts = {
    "drawshare": 1.0,  # share of the credit line a fraudster draws before the account is closed
    "reviewcost": 20.0,  # analyst time per application, €
    "catchrate": 0.9,  # share of fraud an analyst stops
    "friction": 2.0,  # cost of delaying a legitimate customer by a review, €
    "lostcustomer": 100.0,  # value lost when a legitimate applicant is declined, €
}
defaultcapacity = 0.025  # analysts can review 2.5 % of daily applications


def expected_costs(p: np.ndarray, limit: np.ndarray, costs: dict | None = None) -> np.ndarray:
    """(n, 3) expected cost of approve, review and decline for each application."""
    c = defaultcosts | (costs or {})
    p, limit = np.asarray(p, float), np.asarray(limit, float)
    loss = p * limit * c["drawshare"]
    return np.column_stack(
        [
            loss,
            c["reviewcost"] + loss * (1 - c["catchrate"]) + (1 - p) * c["friction"],
            (1 - p) * c["lostcustomer"],
        ]
    )


def decide(
    p: np.ndarray,
    limit: np.ndarray,
    capacity: float = defaultcapacity,
    costs: dict | None = None,
) -> np.ndarray:
    """Cheapest action per application, with reviews capped at `capacity` × volume."""
    e = expected_costs(p, limit, costs)
    choice = e.argmin(axis=1)
    fallback = np.where(e[:, approve] <= e[:, decline], approve, decline)
    slots = int(np.floor(capacity * len(choice)))
    wants = np.flatnonzero(choice == review)
    if len(wants) > slots:
        saving = e[wants, fallback[wants]] - e[wants, review]
        overflow = wants[np.argsort(-saving)[slots:]]
        choice[overflow] = fallback[overflow]
    return choice


def realised(
    y: np.ndarray, limit: np.ndarray, choice: np.ndarray, costs: dict | None = None
) -> np.ndarray:
    """Cost of each decision once the true outcome is known (analyst catch rate in expectation)."""
    c = defaultcosts | (costs or {})
    y, limit = np.asarray(y, bool), np.asarray(limit, float)
    loss = limit * c["drawshare"]
    cost = np.zeros(len(y))
    cost[choice == approve] = np.where(y, loss, 0)[choice == approve]
    reviewed = np.where(
        y, c["reviewcost"] + loss * (1 - c["catchrate"]), c["reviewcost"] + c["friction"]
    )
    cost[choice == review] = reviewed[choice == review]
    cost[choice == decline] = np.where(y, 0, c["lostcustomer"])[choice == decline]
    return cost


def outcome(y: np.ndarray, limit: np.ndarray, choice: np.ndarray, costs: dict | None = None):
    """Business summary of a set of decisions."""
    c = defaultcosts | (costs or {})
    y = np.asarray(y, bool)
    limit = np.asarray(limit, float)
    stopped = (choice == decline) | (choice == review)
    stoppedshare = np.where(choice == review, c["catchrate"], 1.0) * stopped
    return {
        "cost": float(realised(y, limit, choice, costs).sum()),
        "fraud_loss": float((limit * c["drawshare"] * y * (1 - stoppedshare)).sum()),
        "fraud_stopped": float(stoppedshare[y].sum() / max(y.sum(), 1)),
        "reviews": int((choice == review).sum()),
        "declined": int((choice == decline).sum()),
        "good_declined": int(((choice == decline) & ~y).sum()),
        "good_reviewed": int(((choice == review) & ~y).sum()),
    }
