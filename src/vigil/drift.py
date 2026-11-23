"""Population stability index (PSI) per feature and month, against the months the model is fit on.

Rule of thumb from credit risk: PSI below 0.1 is stable, 0.1 to 0.25 a moderate shift, above 0.25
a large shift.

uv run python -m vigil.drift
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from vigil import config
from vigil.data import prepare

eps = 1e-4


def iscategorical(values: pd.Series) -> bool:
    return isinstance(values.dtype, pd.CategoricalDtype) or values.dtype == object


def shares(values: pd.Series, edges: np.ndarray | None, levels: list | None) -> np.ndarray:
    """Share of rows in each bin, with missing values as their own bin."""
    present = values.dropna()
    if levels is not None:
        counts = present.astype(object).value_counts().reindex(levels, fill_value=0).to_numpy()
    else:
        bins = np.searchsorted(edges, present.to_numpy(float), side="right")
        counts = np.bincount(bins, minlength=len(edges) + 1)
    total = max(len(values), 1)
    return np.append(counts / total, values.isna().sum() / total)


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    if iscategorical(reference):
        levels, edges = list(reference.astype(object).dropna().unique()), None
    else:
        cuts = np.linspace(0, 1, bins + 1)[1:-1]
        edges, levels = np.unique(np.nanquantile(reference.to_numpy(float), cuts)), None
    expected = np.clip(shares(reference, edges, levels), eps, None)
    actual = np.clip(shares(current, edges, levels), eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def table(df: pd.DataFrame) -> pd.DataFrame:
    reference = df[df[config.monthcol].isin(config.fitmonths)]
    rows = []
    for month, current in df.groupby(config.monthcol):
        for c in prepare.features(df):
            rows.append({"month": month, "feature": c, "psi": psi(reference[c], current[c])})
    return pd.DataFrame(rows)


def monthly(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(config.monthcol)
    return pd.DataFrame(
        {
            "applications": g.size(),
            "fraud_rate": g[config.target].mean(),
            "share_over50": g["customer_age"].apply(lambda a: (a >= config.agegroup).mean()),
        }
    )


def main() -> None:
    df = prepare.load()
    out = config.reportsdir / "data"
    out.mkdir(parents=True, exist_ok=True)

    months = monthly(df)
    months.to_csv(out / "monthly.csv")
    wide = table(df).pivot(index="feature", columns="month", values="psi")
    wide.to_csv(out / "psi.csv")

    test = wide[list(config.testmonths)].max(axis=1).sort_values(ascending=False)
    month = df[config.monthcol]
    summary = {
        "rows": len(df),
        "fraud_rate": float(df[config.target].mean()),
        "fraud_rate_fit": float(df[month.isin(config.fitmonths)][config.target].mean()),
        "fraud_rate_test": float(df[month.isin(config.testmonths)][config.target].mean()),
        "features_psi_over_0.1_in_test": int((test > 0.1).sum()),
        "features_psi_over_0.25_in_test": int((test > 0.25).sum()),
        "top_shift_in_test": test.head(6).round(3).to_dict(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(months.round(4).to_string())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
