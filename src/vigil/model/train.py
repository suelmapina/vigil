"""Fit and compare fraud models on a temporal split.

fit on months 0 to 4, tune (early stopping, calibration, thresholds) on month 5, report on the
unseen months 6 and 7. Every run is logged to MLflow.

uv run python -m vigil.model.train
"""

from __future__ import annotations

import json
import time

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from vigil import config, drift
from vigil.data import prepare
from vigil.model import metrics

scorespath = config.processeddir / "test_scores.parquet"
validpath = config.processeddir / "valid_scores.parquet"
modelpath = config.modelsdir / "model.joblib"
unstablepsi = 0.25

treeparams = dict(
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=300,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=5.0,
    random_state=config.seed,
    verbose=-1,
)


def rules(df: pd.DataFrame) -> np.ndarray:
    """A typical hand-written rules engine: count red flags, break ties by name/email mismatch."""
    flags = (
        df["prev_address_months_count"].isna().astype(int)
        + df["email_is_free"]
        + df["foreign_request"]
        + (df["keep_alive_session"] == 0).astype(int)
        + (df["phone_home_valid"] == 0).astype(int)
        + (df["device_distinct_emails_8w"] >= 2).astype(int)
        + (df["name_email_similarity"] < 0.2).astype(int)
    )
    return (flags + 0.5 * (1 - df["name_email_similarity"])).to_numpy()


def logistic(fit: pd.DataFrame, cols: list[str]):
    numeric = [c for c in cols if c not in prepare.categorical]
    pre = ColumnTransformer(
        [
            (
                "num",
                make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True), StandardScaler()
                ),
                numeric,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), prepare.categorical),
        ]
    )
    model = make_pipeline(pre, LogisticRegression(max_iter=2000, C=0.5))
    return model.fit(fit[cols], fit[config.target])


def gbm(fit: pd.DataFrame, valid: pd.DataFrame, cols: list[str]) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**treeparams)
    model.fit(
        fit[cols],
        fit[config.target],
        eval_set=[(valid[cols], valid[config.target])],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    return model


def unstable(fit: pd.DataFrame, valid: pd.DataFrame, cols: list[str]) -> list[str]:
    """Features whose distribution already moved by the validation month (test is not used)."""
    return [c for c in cols if drift.psi(fit[c], valid[c]) > unstablepsi]


def calibration(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "brier": float(brier_score_loss(y, p)),
        "mean_predicted": float(p.mean()),
        "observed_rate": float(y.mean()),
    }


def main() -> None:
    df = prepare.load()
    fit, valid, test = prepare.split(df)
    cols = prepare.features(df)
    ytest, yvalid = test[config.target].to_numpy(), valid[config.target].to_numpy()
    older = prepare.older(test)
    print(f"fit {len(fit):,} · valid {len(valid):,} · test {len(test):,} rows")

    moved = unstable(fit, valid, cols)
    stable = [c for c in cols if c not in moved]
    print(f"unstable by month 5 (PSI > {unstablepsi}): {moved}")

    mlflow.set_tracking_uri(config.mlflowuri)
    mlflow.set_experiment("vigil-model")

    models, validscores, testscores, timing = {}, {}, {}, {}

    validscores["rules"], testscores["rules"] = rules(valid), rules(test)
    timing["rules"] = 0.0

    start = time.time()
    models["logistic"] = logistic(fit, cols)
    timing["logistic"] = time.time() - start
    for name, cs in [("lightgbm", cols), ("lightgbm_stable", stable)]:
        start = time.time()
        models[name] = gbm(fit, valid, cs)
        timing[name] = time.time() - start
    for name, m in models.items():
        cs = stable if name == "lightgbm_stable" else cols
        validscores[name] = m.predict_proba(valid[cs])[:, 1]
        testscores[name] = m.predict_proba(test[cs])[:, 1]

    results, threshold = {}, {}
    for name, s in testscores.items():
        res = metrics.summary(ytest, s, older)
        # threshold chosen on month 5 and kept fixed, as it would be in production
        threshold[name] = metrics.threshold_at_fpr(yvalid, validscores[name], config.targetfpr)
        recall, fpr = metrics.rates(ytest, s, threshold[name])
        res |= {"recall_fixed_threshold": recall, "fpr_fixed_threshold": fpr}
        for month in config.testmonths:
            inmonth = (test[config.monthcol] == month).to_numpy()
            res[f"fpr_fixed_month{month}"] = metrics.rates(
                ytest[inmonth], s[inmonth], threshold[name]
            )[1]
        res["fit_seconds"] = round(timing[name], 1)
        results[name] = res
        with mlflow.start_run(run_name=name):
            mlflow.log_params(
                {"model": name, "features": len(stable if "stable" in name else cols)}
            )
            if name in models and "lightgbm" in name:
                mlflow.log_params({k: v for k, v in treeparams.items() if k != "verbose"})
                mlflow.log_param("best_iteration", models[name].best_iteration_)
            mlflow.log_metrics({k: v for k, v in res.items() if isinstance(v, float)})
        print(f"{name:16s} " + "  ".join(f"{k} {v:.3f}" for k, v in res.items()))

    # design rule fixed before looking at test: do not ship features that already drifted by the
    # validation month, since a threshold set today would not hold next month
    best = "lightgbm_stable"
    iso = IsotonicRegression(out_of_bounds="clip").fit(validscores[best], yvalid)
    calibrated = iso.predict(testscores[best])
    calib = {
        "raw": calibration(ytest, testscores[best]),
        "isotonic": calibration(ytest, calibrated),
    }

    print("bootstrap (200 paired resamples)...")
    boot = metrics.bootstrap(ytest, testscores, reference="logistic")

    out = config.reportsdir / "model"
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "fit_rows": len(fit),
        "valid_rows": len(valid),
        "test_rows": len(test),
        "test_fraud_rate": float(ytest.mean()),
        "unstable_features": moved,
        "best_model": best,
        "models": results,
        "bootstrap_vs_logistic": boot,
        "calibration": calib,
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(results).T.round(4).to_csv(out / "models.csv")

    keep = [config.monthcol, config.target, "customer_age", "proposed_credit_limit"]
    scored = test[keep].reset_index(drop=True)
    for name, s in testscores.items():
        scored["score_" + name] = s
    scored["probability"] = calibrated
    scored.to_parquet(scorespath, index=False)
    validframe = valid[keep].reset_index(drop=True)
    validframe["score"] = validscores[best]
    validframe["probability"] = iso.predict(validscores[best])
    validframe.to_parquet(validpath, index=False)

    config.modelsdir.mkdir(parents=True, exist_ok=True)
    features = stable if best == "lightgbm_stable" else cols
    joblib.dump(
        {"model": models[best], "calibrator": iso, "features": features, "name": best}, modelpath
    )

    print(json.dumps({"best": best, "calibration": calib}, indent=2))
    for name, b in boot.items():
        r = b["recall_at_5fpr"]
        print(
            f"{name:16s} recall@5%fpr [{r['low']:.3f}, {r['high']:.3f}] "
            f"vs logistic {r['diff']:+.3f} [{r['diff_low']:+.3f}, {r['diff_high']:+.3f}]"
        )


if __name__ == "__main__":
    main()
