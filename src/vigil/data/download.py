"""Download the BAF base dataset from Kaggle and store it as parquet.

Needs a Kaggle API token: ~/.kaggle/kaggle.json, or KAGGLE_API_TOKEN in the environment.

uv run python -m vigil.data.download
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from vigil import config


def fetch(filename: str = config.basecsv) -> Path:
    import kagglehub

    cached = Path(kagglehub.dataset_download(config.kagglehandle, path=filename))
    config.rawdir.mkdir(parents=True, exist_ok=True)
    local = config.rawdir / filename
    if not local.exists():
        shutil.copy(cached, local)
    return local


def to_parquet(csv: Path) -> Path:
    df = pd.read_csv(csv)
    config.processeddir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.baseparquet, index=False)
    return config.baseparquet


def main() -> None:
    csv = fetch()
    out = to_parquet(csv)
    df = pd.read_parquet(out)
    rate = df[config.target].mean()
    print(f"✓ {out.name}: {len(df):,} rows, {df.shape[1]} columns, fraud rate {rate:.2%}")
    print(df.groupby(config.monthcol)[config.target].agg(["size", "mean"]).round(4))


if __name__ == "__main__":
    main()
