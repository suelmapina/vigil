"""Monitoring after go-live: what a monthly model report would have shown, and when to retrain.

Three parts:

1. Production report for the shipped model (fit on months 0 to 4, live from month 6). Signals
   that need no labels (score and feature drift, share flagged) are available on day one; signals
   that need labels (fraud caught, cost, fairness) only once fraud outcomes mature.
2. Backtest of two retraining policies over months 3 to 7: a model trained once and never
   touched, against a monthly retrain on all data up to the previous month.
3. Shadow test: the retrained challenger scores month 7 alongside the champion without acting,
   and is promoted only if it is cheaper with confidence and not less fair.

uv run python -m vigil.monitor
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from vigil import config, drift
from vigil.data import prepare
from vigil.decision import engine
from vigil.fairness import burden
from vigil.model import metrics, train

fastparams = {"learning_rate": 0.06, "n_estimators": 1500}
alerts = {
    # no labels needed: available the day the applications arrive
    "score_psi": 0.10,  # model output shifted
    "flagged_over_reference": 1.15,  # share above the threshold up 15 % on the tuning month
    "feature_psi": 0.25,  # an input the model uses shifted a lot
    # need fraud outcomes, so they arrive once labels mature
    "fpr_over_target": 1.2,  # legitimate customers flagged 20 % above plan
    "fairness_drop": 0.8,  # balance between age groups fell by a fifth
}


def package(model, features, calibrator, fit, valid) -> dict:
    score = model.predict_proba(valid[features])[:, 1]
    y = valid[config.target].to_numpy()
    if calibrator is None:
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(score, y)
    return {
        "model": model,
        "features": features,
        "calibrator": calibrator,
        "threshold": metrics.threshold_at_fpr(y, score, config.targetfpr),
        "reference": score,
        "fit": fit,
    }


def build(df: pd.DataFrame, fitmonths, validmonth: int, params: dict | None = fastparams):
    """Train the same pipeline as the shipped model on a different window."""
    month = df[config.monthcol]
    fit, valid = df[month.isin(fitmonths)], df[month == validmonth]
    cols = prepare.features(df)
    stable = [c for c in cols if c not in train.unstable(fit, valid, cols)]
    model = train.gbm(fit, valid, stable, params)
    return package(model, stable, None, fit, valid)


def shipped(df: pd.DataFrame) -> dict:
    """The model actually in production (vigil.model.train), with its month 5 references."""
    bundle = train.joblib.load(train.modelpath)
    month = df[config.monthcol]
    fit, valid = df[month.isin(config.fitmonths)], df[month == config.validmonth]
    return package(bundle["model"], bundle["features"], bundle["calibrator"], fit, valid)


def evaluate(m: dict, data: pd.DataFrame) -> dict:
    """Everything a monthly report needs for one month of applications."""
    y = data[config.target].to_numpy()
    limit = data["proposed_credit_limit"].to_numpy()
    older = prepare.older(data)
    score = m["model"].predict_proba(data[m["features"]])[:, 1]
    choice = engine.decide(m["calibrator"].predict(score), limit)
    recall, fpr = metrics.rates(y, score, m["threshold"])
    featurepsi = {c: drift.psi(m["fit"][c], data[c]) for c in m["features"]}
    worst = max(featurepsi, key=featurepsi.get)
    return {
        "applications": len(data),
        "fraud_rate": float(y.mean()),
        "score_psi": drift.psi(pd.Series(m["reference"]), pd.Series(score)),
        "worst_feature": worst,
        "worst_feature_psi": featurepsi[worst],
        "flagged_share": float((score > m["threshold"]).mean()),
        "fpr_fixed_threshold": fpr,
        "recall_fixed_threshold": recall,
        "recall_at_5fpr": metrics.recall_at_fpr(y, score),
        "cost_per_1k": float(engine.realised(y, limit, choice).sum() / len(y) * 1000),
        "fairness_balance": burden(y, older, choice)["ratio"],
        "percase": engine.realised(y, limit, choice),
    }


def flags(row: dict, reference: dict) -> list[str]:
    out = []
    if row["score_psi"] > alerts["score_psi"]:
        out.append("score drift")
    if row["flagged_share"] > alerts["flagged_over_reference"] * reference["flagged_share"]:
        out.append("more applications flagged")
    if row["worst_feature_psi"] > alerts["feature_psi"]:
        out.append(f"input drift ({row['worst_feature']})")
    if row["fpr_fixed_threshold"] > alerts["fpr_over_target"] * config.targetfpr:
        out.append("false positives over plan")
    if row["fairness_balance"] < alerts["fairness_drop"] * reference["fairness_balance"]:
        out.append("fairness gap widened")
    return out


def strip(row: dict) -> dict:
    return {k: v for k, v in row.items() if k != "percase"}


def main() -> None:
    df = prepare.load()
    month = df[config.monthcol]
    out = config.reportsdir / "monitor"
    out.mkdir(parents=True, exist_ok=True)

    print("1. production report for the shipped model...")
    live = shipped(df)
    reference = evaluate(live, df[month == config.validmonth])
    report = []
    for m in [config.validmonth, *config.testmonths]:
        row = evaluate(live, df[month == m])
        report.append({"month": m} | strip(row) | {"alerts": "; ".join(flags(row, reference))})
    report = pd.DataFrame(report).set_index("month")
    report.round(4).to_csv(out / "production.csv")

    print("2. backtest: trained once vs retrained monthly...")
    static = build(df, range(0, 2), 2)
    rows = []
    for m in range(3, 8):
        current = df[month == m]
        monthly = build(df, range(0, m - 1), m - 1)
        result = {"static": evaluate(static, current), "monthly": evaluate(monthly, current)}
        for name, row in result.items():
            rows.append({"month": m, "policy": name} | strip(row))
        print(
            f"   month {m}: cost per 1k static €{result['static']['cost_per_1k']:,.0f} "
            f"vs monthly €{result['monthly']['cost_per_1k']:,.0f}"
        )
    current = df[month == 7]
    champion = evaluate(live, current)  # in production since month 6
    # same training setup as the champion, one more month of data, tuned on month 6
    challenger = evaluate(build(df, range(0, 6), 6, params=None), current)
    backtest = pd.DataFrame(rows)
    backtest.round(4).to_csv(out / "backtest.csv", index=False)

    print("3. shadow test on month 7...")
    rng = np.random.default_rng(config.seed)
    diff = champion["percase"] - challenger["percase"]
    draws = np.array([diff[rng.integers(0, len(diff), len(diff))].sum() for _ in range(200)])
    saving = float(diff.sum())
    low, high = float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
    promote = low > 0 and challenger["fairness_balance"] >= 0.95 * champion["fairness_balance"]
    shadow = {
        "month": 7,
        "champion": "fit on months 0 to 4, tuned on 5",
        "challenger": "fit on months 0 to 5, tuned on 6",
        "champion_cost": float(champion["percase"].sum()),
        "challenger_cost": float(challenger["percase"].sum()),
        "saving": saving,
        "saving_low": low,
        "saving_high": high,
        "champion_recall_at_5fpr": champion["recall_at_5fpr"],
        "challenger_recall_at_5fpr": challenger["recall_at_5fpr"],
        "champion_fairness": champion["fairness_balance"],
        "challenger_fairness": challenger["fairness_balance"],
        "promote": bool(promote),
    }

    totals = backtest.groupby("policy")[["cost_per_1k", "recall_at_5fpr"]].mean()
    summary = {
        "alerts": alerts,
        "production": report.round(4).reset_index().to_dict(orient="records"),
        "backtest_mean": totals.round(4).to_dict(orient="index"),
        "shadow": shadow,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))

    pd.set_option("display.width", 220)
    print(report.round(4).to_string())
    print(
        backtest.pivot(
            index="month",
            columns="policy",
            values=["cost_per_1k", "recall_at_5fpr", "fpr_fixed_threshold", "fairness_balance"],
        )
        .round(3)
        .to_string()
    )
    print(json.dumps(shadow, indent=2))


if __name__ == "__main__":
    main()
