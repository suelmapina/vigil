"""Who carries the cost of false alarms, and what it costs to share it more evenly.

Fraud is about 3x more common among applicants aged 50 or more, so a model that is well
calibrated for both groups will also flag more of their legitimate applicants. Equal error rates
and equal calibration cannot both hold when base rates differ, so any fix is a choice. This script
prices three options on the unseen test months:

1. With the age feature: the same model, retrained with customer_age as an input.
2. Vigil: age is never used to score ("fairness through unawareness"). This is what ships.
3. Equal burden: scale down the risk of the older group until both groups have the same share of
   legitimate applicants reviewed or declined. Set on the validation month, applied unchanged.

uv run python -m vigil.fairness
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from vigil import config
from vigil.data import prepare
from vigil.decision import engine
from vigil.model import metrics, train


def burden(y: np.ndarray, older: np.ndarray, choice: np.ndarray) -> dict:
    """Share of legitimate applicants reviewed or declined, and of fraud stopped, by group."""
    legit = y == 0
    stopped = choice != engine.approve
    out = {}
    for name, g in [("under50", ~older), ("over50", older)]:
        out[name] = {
            "good_stopped": float(stopped[g & legit].mean()),
            "good_declined": float((choice == engine.decline)[g & legit].mean()),
            "fraud_stopped": float(stopped[g & ~legit].mean()),
        }
    a, b = out["under50"]["good_stopped"], out["over50"]["good_stopped"]
    out["ratio"] = float(min(a, b) / max(a, b)) if max(a, b) > 0 else 1.0
    return out


def scaled(p: np.ndarray, older: np.ndarray, k: float) -> np.ndarray:
    return np.where(older, p * k, p)


def equalising(valid: pd.DataFrame) -> float:
    """Multiplier on the older group probabilities that equalises burden on the validation month."""
    y, p = valid[config.target].to_numpy(), valid["probability"].to_numpy()
    limit, older = valid["proposed_credit_limit"].to_numpy(), prepare.older(valid)
    low, high = np.log(0.01), 0.0
    for _ in range(40):
        k = np.exp((low + high) / 2)
        b = burden(y, older, engine.decide(scaled(p, older, k), limit))
        if b["over50"]["good_stopped"] > b["under50"]["good_stopped"]:
            high = np.log(k)
        else:
            low = np.log(k)
    return float(np.exp((low + high) / 2))


def withage() -> tuple[np.ndarray, dict]:
    """Retrain the shipped model with customer_age added; calibrated test probabilities."""
    bundle = train.joblib.load(train.modelpath)
    cols = bundle["features"] + ["customer_age"]
    fit, valid, test = prepare.split(prepare.load())
    model = train.gbm(fit, valid, cols)
    validscore = model.predict_proba(valid[cols])[:, 1]
    testscore = model.predict_proba(test[cols])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(validscore, valid[config.target])
    y = test[config.target].to_numpy()
    paper = metrics.summary(y, testscore, prepare.older(test))
    return iso.predict(testscore), paper


def main() -> None:
    test = pd.read_parquet(train.scorespath)
    valid = pd.read_parquet(train.validpath)
    y, limit = test[config.target].to_numpy(), test["proposed_credit_limit"].to_numpy()
    older = prepare.older(test)

    print("retraining with the age feature, for comparison...")
    agep, agepaper = withage()
    k = equalising(valid)
    options = {
        "With the age feature": agep,
        "Vigil": test["probability"].to_numpy(),
        "Equal burden across age groups": scaled(test["probability"].to_numpy(), older, k),
    }
    choices = {name: engine.decide(p, limit) for name, p in options.items()}
    percase = {name: engine.realised(y, limit, c) for name, c in choices.items()}

    rng = np.random.default_rng(config.seed)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(200)]
    rows, groups = [], []
    for name, choice in choices.items():
        b = burden(y, older, choice)
        diff = np.array([(percase[name][i] - percase["Vigil"][i]).sum() for i in idx])
        rows.append(
            {
                "option": name,
                "cost": float(percase[name].sum()),
                "extra_cost": float((percase[name] - percase["Vigil"]).sum()),
                "extra_low": float(np.quantile(diff, 0.025)),
                "extra_high": float(np.quantile(diff, 0.975)),
                "fraud_stopped": engine.outcome(y, limit, choice)["fraud_stopped"],
                "burden_ratio": b["ratio"],
            }
        )
        for g in ["under50", "over50"]:
            groups.append({"option": name, "group": g} | b[g])

    paper = {
        "with_age": agepaper["fpr_ratio_age"],
        "without_age": metrics.summary(y, test["score_lightgbm_stable"], older)["fpr_ratio_age"],
    }
    out = config.reportsdir / "fairness"
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).set_index("option")
    table.round(4).to_csv(out / "options.csv")
    pd.DataFrame(groups).round(4).to_csv(out / "groups.csv", index=False)
    report = {
        "fraud_rate": {"under50": float(y[~older].mean()), "over50": float(y[older].mean())},
        "share_over50": float(older.mean()),
        "older_multiplier": k,
        "paper_fpr_ratio_at_5fpr": paper,
        "recall_at_5fpr_with_age": agepaper["recall_at_5fpr"],
        "options": table.round(4).to_dict(orient="index"),
        "groups": groups,
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2))

    pd.set_option("display.width", 200)
    print(table.round(4).to_string())
    print(pd.DataFrame(groups).round(4).to_string(index=False))
    print(
        json.dumps(
            {
                k2: report[k2]
                for k2 in [
                    "fraud_rate",
                    "older_multiplier",
                    "paper_fpr_ratio_at_5fpr",
                    "recall_at_5fpr_with_age",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
