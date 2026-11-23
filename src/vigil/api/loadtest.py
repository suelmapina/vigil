"""Latency of POST /score under concurrent load, with real applications from the test months.

Start the API first (make api, or uvicorn without --reload), then:
uv run python -m vigil.api.loadtest --requests 3000 --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import httpx
import numpy as np
import pandas as pd

from vigil import config


def payloads(n: int) -> list[dict]:
    df = pd.read_parquet(config.baseparquet)
    df = df[df[config.monthcol].isin(config.testmonths)].sample(n, random_state=config.seed)
    df = df.drop(columns=[config.target, config.monthcol])
    return json.loads(df.to_json(orient="records"))


async def run(url: str, items: list[dict], concurrency: int) -> tuple[np.ndarray, np.ndarray]:
    gate = asyncio.Semaphore(concurrency)
    times, server = [], []

    async def one(client: httpx.AsyncClient, item: dict) -> None:
        async with gate:
            start = time.perf_counter()
            r = await client.post(url, json=item)
            r.raise_for_status()
            times.append(time.perf_counter() - start)
            server.append(float(r.headers["x-process-ms"]))

    async with httpx.AsyncClient(timeout=30) as client:
        for item in items[:50]:  # warm up
            await client.post(url, json=item)
        start = time.perf_counter()
        await asyncio.gather(*(one(client, item) for item in items))
        wall = time.perf_counter() - start
    print(f"{len(items):,} requests in {wall:.1f} s: {len(items) / wall:,.0f} requests per second")
    return np.array(times) * 1000, np.array(server)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/score")
    parser.add_argument("--requests", type=int, default=3000)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    ms, server = asyncio.run(run(args.url, payloads(args.requests), args.concurrency))
    result = {"requests": len(ms), "concurrency": args.concurrency}
    for name, values in [("server", server), ("end_to_end", ms)]:
        for q in (50, 95, 99):
            result[f"{name}_p{q}_ms"] = round(float(np.percentile(values, q)), 2)
    out = config.reportsdir / "api"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"latency_c{args.concurrency}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
