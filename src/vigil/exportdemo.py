"""Write the small files the demo needs into app/data, so it runs without data or a model.

uv run python -m vigil.exportdemo     (after make model and make fairness)
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from vigil import config, explain
from vigil.data import prepare
from vigil.decision import engine
from vigil.fairness import withagepath

appdata = config.root / "app" / "data"
profilefields = [
    "proposed_credit_limit",
    "income",
    "name_email_similarity",
    "email_is_free",
    "phone_home_valid",
    "foreign_request",
    "prev_address_months_count",
    "housing_status",
    "device_os",
    "source",
]


def scores() -> pd.DataFrame:
    test = pd.read_parquet(config.scorespath)
    out = pd.DataFrame(
        {
            "month": test[config.monthcol].astype("int8"),
            config.target: test[config.target].astype("int8"),
            "customer_age": test["customer_age"].astype("int8"),
            "proposed_credit_limit": test["proposed_credit_limit"].astype("float32"),
            "score_rules": test["score_rules"].astype("float32"),
            "score_lightgbm_stable": test["score_lightgbm_stable"].astype("float64"),
            "probability": test["probability"].astype("float64"),
            "probability_withage": pd.read_parquet(withagepath)["probability_withage"].astype(
                "float64"
            ),
        }
    )
    return out


def valid() -> pd.DataFrame:
    v = pd.read_parquet(config.validpath)
    return pd.DataFrame(
        {
            config.target: v[config.target].astype("int8"),
            "customer_age": v["customer_age"].astype("int8"),
            "proposed_credit_limit": v["proposed_credit_limit"].astype("float32"),
            "score": v["score"].astype("float64"),
            "probability": v["probability"].astype("float64"),
        }
    )


def cases(scored: pd.DataFrame, n: dict | None = None) -> pd.DataFrame:
    """A sample of real test applications with their reasons, for the review screen."""
    n = n or {engine.review: 170, engine.decline: 60, engine.approve: 70}
    choice = engine.decide(scored["probability"].to_numpy(), scored["proposed_credit_limit"])
    rng = np.random.default_rng(config.seed)
    picked = np.concatenate(
        [rng.choice(np.flatnonzero(choice == a), k, replace=False) for a, k in n.items()]
    )
    test = prepare.split(prepare.load())[2].reset_index(drop=True).loc[picked]
    bundle = joblib.load(config.modelpath)
    shap = explain.contributions(bundle, test)
    rows = []
    for i in picked:
        contrib = shap.loc[i].sort_values(key=np.abs, ascending=False).head(6)
        rows.append(
            {
                "row": int(i),
                "month": int(test.at[i, config.monthcol]),
                "probability": float(scored.at[i, "probability"]),
                "proposed_credit_limit": float(test.at[i, "proposed_credit_limit"]),
                "fraud": bool(test.at[i, config.target]),
                "drivers": json.dumps(
                    [
                        {
                            "label": explain.labels.get(f, f),
                            "text": explain.describe(f, test.at[i, f]),
                            "impact": round(float(v), 4),
                        }
                        for f, v in contrib.items()
                    ]
                ),
                "profile": json.dumps([explain.describe(f, test.at[i, f]) for f in profilefields]),
            }
        )
    out = pd.DataFrame(rows).sort_values("probability", ascending=False)
    out.insert(0, "application", [f"A-{r:06d}" for r in out["row"]])
    return out.reset_index(drop=True)


def main() -> None:
    appdata.mkdir(parents=True, exist_ok=True)
    scored = scores()
    files = {"scores.parquet": scored, "valid.parquet": valid(), "cases.parquet": cases(scored)}
    for name, df in files.items():
        path = appdata / name
        df.to_parquet(path, index=False, compression="zstd")
        print(f"✓ {name}: {len(df):,} rows, {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
