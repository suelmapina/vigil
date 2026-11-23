"""Scoring API: one application in, a decision with its reasons out.

uv run uvicorn vigil.api.main:app            # docs on http://localhost:8000/docs

The API recommends the cheapest action for one application. Analyst capacity is a queue concern,
so each response also carries `reviewsaving`: how much a review is expected to save over the next
best action. The case management system can sort its queue by it, which is exactly how the batch
engine fills limited analyst capacity.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from vigil import config, explain, version
from vigil.data import prepare
from vigil.decision import engine

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["bundle"] = joblib.load(config.modelpath)
    state["codes"] = encoder(state["bundle"])
    yield
    state.clear()


app = FastAPI(
    title="Vigil",
    version=version,
    summary="Cost-aware fraud screening for online account opening.",
    lifespan=lifespan,
)


class application(BaseModel):
    """One account opening request, with the fields of the BAF dataset. Missing: -1 or null."""

    model_config = ConfigDict(extra="ignore")

    income: float = Field(ge=0, le=1, description="Income decile, 0.1 to 0.9")
    name_email_similarity: float = Field(ge=0, le=1)
    prev_address_months_count: int | None = None
    current_address_months_count: int | None = None
    days_since_request: float = Field(ge=0)
    intended_balcon_amount: float
    payment_type: str
    zip_count_4w: int = Field(ge=0)
    bank_branch_count_8w: int = Field(ge=0)
    date_of_birth_distinct_emails_4w: int = Field(ge=0)
    employment_status: str
    credit_risk_score: int
    email_is_free: int = Field(ge=0, le=1)
    housing_status: str
    phone_home_valid: int = Field(ge=0, le=1)
    phone_mobile_valid: int = Field(ge=0, le=1)
    bank_months_count: int | None = None
    has_other_cards: int = Field(ge=0, le=1)
    proposed_credit_limit: float = Field(gt=0)
    foreign_request: int = Field(ge=0, le=1)
    source: str
    session_length_in_minutes: float | None = None
    device_os: str
    keep_alive_session: int = Field(ge=0, le=1)
    device_distinct_emails_8w: int | None = None


class reason(BaseModel):
    text: str
    impact: float = Field(description="Contribution to the fraud score, in log-odds")


class decision(BaseModel):
    action: str = Field(description="approve, review or decline")
    probability: float = Field(description="Calibrated fraud probability")
    expectedcost: dict[str, float] = Field(description="Expected cost of each action, €")
    reviewsaving: float = Field(description="€ a review saves over the next best action")
    reasons: list[reason]
    model: str


def encoder(bundle: dict) -> dict:
    """Category codes exactly as LightGBM learned them from the pandas training frame."""
    cats = [f for f in bundle["features"] if f in prepare.categorical]
    learned = bundle["model"].booster_.pandas_categorical
    return {
        f: {v: float(i) for i, v in enumerate(levels)}
        for f, levels in zip(cats, learned, strict=True)
    }


def value(feature: str, raw):
    """Same cleaning as vigil.data.prepare, for one field: -1 or null means missing."""
    if raw is None or (feature in prepare.missingcodes and raw == -1):
        return np.nan
    return raw


def check(rows: list[dict]) -> None:
    """Reject categories the model never saw, instead of silently scoring them as missing."""
    for i, row in enumerate(rows):
        for f, known in state["codes"].items():
            if row[f] not in known:
                where = f"item {i}: " if len(rows) > 1 else ""
                raise HTTPException(
                    status_code=422,
                    detail=f"{where}unknown {f} '{row[f]}', expected one of {sorted(known)}",
                )


def matrix(rows: list[dict]) -> np.ndarray:
    features, codes = state["bundle"]["features"], state["codes"]
    x = np.empty((len(rows), len(features)))
    for i, row in enumerate(rows):
        for j, f in enumerate(features):
            v = value(f, row[f])
            x[i, j] = codes[f].get(v, np.nan) if f in codes else v
    return x


def reasonsfor(x: np.ndarray, rows: list[dict]) -> list[list[reason]]:
    """Top three features pushing each application towards fraud (exact TreeSHAP)."""
    features = state["bundle"]["features"]
    contrib = state["bundle"]["model"].booster_.predict(x, pred_contrib=True, num_threads=1)
    out = []
    for shap, row in zip(contrib[:, :-1], rows, strict=True):
        top = [j for j in np.argsort(-shap)[:3] if shap[j] > 0]
        out.append(
            [
                reason(
                    text=explain.describe(features[j], value(features[j], row[features[j]])),
                    impact=round(float(shap[j]), 3),
                )
                for j in top
            ]
        )
    return out


def score(items: list[application], always: bool = False) -> list[decision]:
    """Reasons are computed for reviews and declines, the decisions someone has to act on.

    TreeSHAP is about 40x slower than the score itself, and nobody reads the reasons for an
    approval, so approvals skip it unless `always` is set.
    """
    bundle = state["bundle"]
    rows = [a.model_dump() for a in items]
    check(rows)
    x = matrix(rows)
    raw = bundle["model"].booster_.predict(x, num_threads=1)
    p = bundle["calibrator"].predict(raw)
    limit = np.array([r["proposed_credit_limit"] for r in rows], float)
    e = engine.expected_costs(p, limit)
    choice = engine.decide(p, limit, capacity=1.0)
    explained = np.flatnonzero((choice != engine.approve) | always)
    why = [[] for _ in rows]
    if len(explained):
        for i, r in zip(
            explained, reasonsfor(x[explained], [rows[i] for i in explained]), strict=True
        ):
            why[i] = r
    out = []
    for i in range(len(rows)):
        alternative = min(e[i, engine.approve], e[i, engine.decline])
        out.append(
            decision(
                action=engine.actions[choice[i]],
                probability=round(float(p[i]), 4),
                expectedcost={a: round(float(e[i, k]), 2) for k, a in enumerate(engine.actions)},
                reviewsaving=round(float(alternative - e[i, engine.review]), 2),
                reasons=why[i],
                model=bundle["name"],
            )
        )
    return out


class timer:
    """Server-side processing time in a response header (plain ASGI, no extra overhead)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        start = time.perf_counter()

        async def stamped(message):
            if message["type"] == "http.response.start":
                ms = f"{(time.perf_counter() - start) * 1000:.3f}".encode()
                message["headers"] = [*message.get("headers", []), (b"x-process-ms", ms)]
            await send(message)

        await self.app(scope, receive, stamped)


app.add_middleware(timer)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": state["bundle"]["name"], "version": version}


@app.post("/score", response_model=decision)
async def scoreone(item: application, explain: bool = False) -> decision:
    """Score one application. Set explain=true to get reasons for approvals too."""
    return score([item], always=explain)[0]


@app.post("/score-batch", response_model=list[decision])
async def scoremany(items: list[application], explain: bool = False) -> list[decision]:
    return score(items, always=explain) if items else []
