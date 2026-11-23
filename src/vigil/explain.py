"""Reason codes for every decision, from exact TreeSHAP contributions.

LightGBM computes SHAP values natively (pred_contrib), so no extra dependency is needed: each
application gets the features that pushed its score towards fraud, in plain words, ready for the
analyst who reviews it.

uv run python -m vigil.explain
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from vigil import config
from vigil.data import prepare
from vigil.decision import engine
from vigil.model.train import modelpath, scorespath

labels = {
    "income": "Income decile",
    "name_email_similarity": "Name and email similarity",
    "prev_address_months_count": "Months at previous address",
    "current_address_months_count": "Months at current address",
    "customer_age": "Age group",
    "days_since_request": "Days since request",
    "intended_balcon_amount": "Initial transfer amount",
    "payment_type": "Payment plan",
    "zip_count_4w": "Applications from the same zip code, 4 weeks",
    "velocity_6h": "Application velocity, 6 hours",
    "velocity_24h": "Application velocity, 24 hours",
    "velocity_4w": "Application velocity, 4 weeks",
    "bank_branch_count_8w": "Applications at the same branch, 8 weeks",
    "date_of_birth_distinct_emails_4w": "Emails sharing this birth date, 4 weeks",
    "employment_status": "Employment status",
    "credit_risk_score": "Credit risk score",
    "email_is_free": "Free email provider",
    "housing_status": "Housing status",
    "phone_home_valid": "Valid home phone",
    "phone_mobile_valid": "Valid mobile phone",
    "bank_months_count": "Months with previous bank",
    "has_other_cards": "Has other cards",
    "proposed_credit_limit": "Proposed credit limit",
    "foreign_request": "Request from abroad",
    "source": "Application channel",
    "session_length_in_minutes": "Session length, minutes",
    "device_os": "Device operating system",
    "keep_alive_session": "Kept session alive",
    "device_distinct_emails_8w": "Emails used on this device, 8 weeks",
}


def load():
    return joblib.load(modelpath)


def contributions(bundle: dict, x: pd.DataFrame) -> pd.DataFrame:
    """SHAP value of each feature, in log-odds, for each row (the bias column is dropped)."""
    cols = bundle["features"]
    values = bundle["model"].booster_.predict(x[cols], pred_contrib=True)
    return pd.DataFrame(values[:, :-1], columns=cols, index=x.index)


def describe(feature: str, value) -> str:
    label = labels.get(feature, feature)
    if pd.isna(value):
        return f"{label}: not provided"
    if feature == "income":
        return f"{label}: {round(float(value) * 10)} of 10"
    if isinstance(value, float) and not value.is_integer():
        return f"{label}: {value:.2f}"
    if isinstance(value, float):
        value = int(value)
    return f"{label}: {value}"


def reasons(bundle: dict, x: pd.DataFrame, k: int = 3) -> list[list[dict]]:
    """Top k features pushing each application towards fraud, with their values."""
    shap = contributions(bundle, x)
    out = []
    for i, row in shap.iterrows():
        top = row[row > 0].sort_values(ascending=False).head(k)
        out.append(
            [
                {"feature": f, "text": describe(f, x.at[i, f]), "impact": round(float(v), 3)}
                for f, v in top.items()
            ]
        )
    return out


def main(sample: int = 20000) -> None:
    bundle = load()
    df = prepare.load()
    test = prepare.split(df)[2].reset_index(drop=True)
    scored = pd.read_parquet(scorespath)

    rows = test.sample(sample, random_state=config.seed)
    shap = contributions(bundle, rows)
    importance = shap.abs().mean().sort_values(ascending=False)

    out = config.reportsdir / "explain"
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(
        {
            "label": [labels.get(f, f) for f in importance.index],
            "mean_abs_shap": importance.round(4),
        }
    )
    table.to_csv(out / "importance.csv")

    choice = engine.decide(scored["probability"], scored["proposed_credit_limit"])
    reviewed = np.flatnonzero(choice == engine.review)
    rng = np.random.default_rng(config.seed)
    examples = []
    for i in rng.choice(reviewed, 5, replace=False):
        examples.append(
            {
                "probability": round(float(scored.at[i, "probability"]), 3),
                "credit_limit": float(scored.at[i, "proposed_credit_limit"]),
                "fraud": bool(scored.at[i, config.target]),
                "reasons": reasons(bundle, test.loc[[i]])[0],
            }
        )

    (out / "examples.json").write_text(json.dumps({"examples": examples}, indent=2))
    print(table.head(12).to_string())
    for e in examples[:3]:
        print(e)


if __name__ == "__main__":
    main()
