"""Compare screening policies on the unseen test months, in euros.

Every policy gets the same analyst capacity. Thresholds come from the validation month, so
nothing is tuned on the months being scored.

uv run python -m vigil.decision.simulate
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from vigil import config
from vigil.decision import engine
from vigil.model import metrics

modelscore = "score_lightgbm_stable"


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_parquet(config.scorespath), pd.read_parquet(config.validpath)


def topshare(score: np.ndarray, share: float) -> np.ndarray:
    """Send the highest `share` of scores to review, approve the rest."""
    choice = np.full(len(score), engine.approve)
    k = int(np.floor(share * len(score)))
    choice[np.argsort(-score)[:k]] = engine.review
    return choice


def policies(test: pd.DataFrame, valid: pd.DataFrame, capacity: float, costs: dict | None):
    y, limit = test[config.target].to_numpy(), test["proposed_credit_limit"].to_numpy()
    cut = metrics.threshold_at_fpr(
        valid[config.target].to_numpy(), valid["score"].to_numpy(), config.targetfpr
    )
    model = test[modelscore].to_numpy()
    return (
        y,
        limit,
        {
            "Approve everyone": np.full(len(test), engine.approve),
            "Rules: review top alerts": topshare(test["score_rules"].to_numpy(), capacity),
            "Model: review top scores": topshare(model, capacity),
            "Model: decline at 5% FPR": np.where(model > cut, engine.decline, engine.approve),
            "Vigil, uncalibrated scores": engine.decide(model, limit, capacity, costs),
            "Vigil: cost-based decision": engine.decide(
                test["probability"], limit, capacity, costs
            ),
        },
    )


def compare(test, valid, capacity=engine.defaultcapacity, costs=None, reps=200, seed=42):
    y, limit, choices = policies(test, valid, capacity, costs)
    percase = {n: engine.realised(y, limit, c, costs) for n, c in choices.items()}
    reference = percase["Vigil: cost-based decision"]
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(reps)]
    rows = []
    for name, choice in choices.items():
        row = {"policy": name} | engine.outcome(y, limit, choice, costs)
        diff = np.array([(percase[name][i] - reference[i]).sum() for i in idx])
        row |= {
            "vs_vigil": float((percase[name] - reference).sum()),
            "vs_vigil_low": float(np.quantile(diff, 0.025)),
            "vs_vigil_high": float(np.quantile(diff, 0.975)),
        }
        rows.append(row)
    return pd.DataFrame(rows).set_index("policy")


def sensitivity(test, valid) -> pd.DataFrame:
    rows = []
    for lost in [25, 50, 100, 200, 400]:
        for capacity in [0.005, 0.01, 0.025, 0.05, 0.1]:
            y, limit, choices = policies(test, valid, capacity, {"lostcustomer": lost})
            for name in [
                "Model: review top scores",
                "Model: decline at 5% FPR",
                "Vigil: cost-based decision",
            ]:
                cost = engine.realised(y, limit, choices[name], {"lostcustomer": lost}).sum()
                rows.append(
                    {
                        "lostcustomer": lost,
                        "capacity": capacity,
                        "policy": name,
                        "cost": float(cost),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    test, valid = load()
    table = compare(test, valid)
    sens = sensitivity(test, valid)

    out = config.reportsdir / "decision"
    out.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out / "policies.csv")
    sens.to_csv(out / "sensitivity.csv", index=False)
    report = {
        "costs": engine.defaultcosts,
        "capacity": engine.defaultcapacity,
        "applications": len(test),
        "frauds": int(test[config.target].sum()),
        "policies": table.round(4).to_dict(orient="index"),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2))

    pd.set_option("display.width", 220)
    show = table.copy()
    for c in ["cost", "fraud_loss", "vs_vigil", "vs_vigil_low", "vs_vigil_high"]:
        show[c] = (show[c] / 1000).round(1)
    show["fraud_stopped"] = (show["fraud_stopped"] * 100).round(1)
    print(
        f"{len(test):,} applications, {int(test[config.target].sum()):,} frauds, "
        f"costs in €k, capacity {engine.defaultcapacity:.1%}"
    )
    print(show.to_string())
    best = sens.loc[sens.groupby(["lostcustomer", "capacity"])["cost"].idxmin()]
    print("\ncheapest policy per scenario:")
    print(best.pivot(index="lostcustomer", columns="capacity", values="policy").to_string())


if __name__ == "__main__":
    main()
