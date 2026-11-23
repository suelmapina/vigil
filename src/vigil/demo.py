"""Vigil interactive demo.

uv run streamlit run src/vigil/demo.py
"""

from __future__ import annotations

import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from vigil import config, theme
from vigil.decision import engine, simulate
from vigil.explain import labels
from vigil.fairness import burden, equalising, scaled

repourl = "https://github.com/suelmapina/vigil"
appdata = config.root / "app" / "data"
reports = config.reportsdir
capacitygrid = [0.005, 0.01, 0.025, 0.05, 0.1]
ours = "Vigil: cost-based decision"
display = {
    "Approve everyone": "Approve everyone",
    "Rules: review top alerts": "Rules engine, analysts on top alerts",
    "Model: review top scores": "Model score, analysts on top scores",
    "Model: decline at 5% FPR": "Model score, decline above 5% FPR",
    ours: ours,
}
actioncolor = {"approve": theme.context, "review": theme.secondary, "decline": theme.primary}

st.set_page_config(
    page_title="Vigil · Fraud screening", page_icon=":material/shield:", layout="wide"
)
st.markdown(theme.css, unsafe_allow_html=True)
st.markdown(
    f"""<style>
.badge.approve {{ color: {theme.light}; background: {theme.muted}; }}
.badge.review {{ color: {theme.background}; background: {theme.secondary};
  border-color: {theme.secondary}; }}
.badge.decline {{ color: {theme.foreground}; background: rgba(140, 71, 209, 0.45);
  border-color: {theme.primary}; }}
.chip {{ display: inline-block; font-size: 0.76rem; padding: 0.12rem 0.5rem;
  margin: 0 0.3rem 0.3rem 0; border-radius: 6px; border: 1px solid {theme.border};
  color: {theme.light}; }}
.chip.alert {{ border-color: rgba(140, 71, 209, 0.6); background: rgba(140, 71, 209, 0.14); }}
.assumptions {{ font-family: '{theme.mono}', monospace; font-size: 0.68rem; letter-spacing: 0.1em;
  text-transform: uppercase; color: {theme.mutedforeground}; margin: 1.6rem 0 0.2rem; }}
.stTabs [data-baseweb="tab-highlight"] {{ height: 2px; }}
.report {{ display: grid; grid-template-columns: 0.8fr 3.2fr repeat(5, 1fr);
  gap: 0.55rem 1.2rem; align-items: center; color: {theme.light}; font-size: 0.92rem; }}
.report .chip {{ margin-bottom: 0.15rem; }}
table.steps tr.chosen td {{ color: {theme.foreground}; font-weight: 600; }}
.split {{ display: flex; gap: 6px; margin: 0.4rem 0 0.2rem; }}
.split div {{ border-radius: 6px; padding: 0.5rem 0.6rem; font-size: 0.8rem; color: {theme.light};
  border: 1px solid {theme.border}; }}
.split .fit {{ flex: 5; background: {theme.card}; }}
.split .tune {{ flex: 1; background: rgba(38, 202, 217, 0.12); border-color: {theme.secondary}; }}
.split .test {{ flex: 2; background: rgba(140, 71, 209, 0.18); border-color: {theme.primary}; }}
</style>""",
    unsafe_allow_html=True,
)
alt.theme.register("portfolio", enable=True)(theme.altair_theme)


# --- helpers -------------------------------------------------------------------------------
def html(text: str) -> None:
    st.markdown(text, unsafe_allow_html=True)


def money(v: float, decimals: int = 0) -> str:
    sign = "−" if v < 0 else ""
    v = abs(v)
    if v >= 1e6:
        return f"{sign}€{v / 1e6:,.2f}M"
    if v >= 10000 and decimals == 0:
        return f"{sign}€{v / 1000:,.0f}k"
    return f"{sign}€{v:,.{decimals}f}"


def risk(p: float) -> str:
    return ">99%" if p >= 0.995 else f"{p:.1%}"


def kpi(label: str, value: str, note: str, accent: bool = False, highlight: bool = False) -> str:
    return (
        f'<div class="card{" highlight" if highlight else ""}">'
        f'<div class="kpilabel">{label}</div>'
        f'<div class="kpivalue{" accent" if accent else ""}">{value}</div>'
        f'<div class="kpinote">{note}</div></div>'
    )


def section(title: str, note: str) -> None:
    html(f'<div class="sectiontitle">{title}</div><div class="sectionnote">{note}</div>')


eurolabels = (
    "datum.value == 0 ? '€0' : (abs(datum.value) >= 1e6 ? "
    "'€' + format(datum.value / 1e6, ',.1f') + 'M' : "
    "'€' + format(datum.value / 1000, ',.0f') + 'k')"
)


# --- data ----------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def data():
    test = pd.read_parquet(appdata / "scores.parquet")
    valid = pd.read_parquet(appdata / "valid.parquet")
    cases = pd.read_parquet(appdata / "cases.parquet")
    return test, valid, cases


@st.cache_data(show_spinner=False)
def evidence() -> dict:
    out = {
        name: json.loads((reports / name / "metrics.json").read_text())
        for name in ("model", "decision", "fairness", "monitor")
    }
    out["monthly"] = pd.read_csv(reports / "data" / "monthly.csv")
    out["psi"] = pd.read_csv(reports / "data" / "psi.csv", index_col=0)
    out["production"] = pd.read_csv(reports / "monitor" / "production.csv")
    out["backtest"] = pd.read_csv(reports / "monitor" / "backtest.csv")
    out["latency"] = json.loads((reports / "api" / "latency_c8.json").read_text())
    return out


@st.cache_data(show_spinner=False)
def comparison(lost: float, reviewcost: float, capacity: float) -> pd.DataFrame:
    test, valid, _ = data()
    costs = {"lostcustomer": lost, "reviewcost": reviewcost}
    table = simulate.compare(test, valid, capacity, costs, reps=100)
    table = table.drop(index="Vigil, uncalibrated scores")
    table.index = [display[n] for n in table.index]
    return table


@st.cache_data(show_spinner=False)
def capacitycurve(lost: float, reviewcost: float) -> pd.DataFrame:
    test, valid, _ = data()
    costs = {"lostcustomer": lost, "reviewcost": reviewcost}
    rows = []
    for capacity in capacitygrid:
        y, limit, choices = simulate.policies(test, valid, capacity, costs)
        for name in (ours, "Model: review top scores", "Rules: review top alerts"):
            cost = engine.realised(y, limit, choices[name], costs).sum()
            rows.append({"capacity": capacity * 100, "policy": display[name], "cost": cost})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def vigilchoice(lost: float, reviewcost: float, capacity: float) -> np.ndarray:
    test, _, _ = data()
    costs = {"lostcustomer": lost, "reviewcost": reviewcost}
    return engine.decide(
        test["probability"].to_numpy(), test["proposed_credit_limit"].to_numpy(), capacity, costs
    )


@st.cache_data(show_spinner=False)
def fairnessoptions(lost: float, reviewcost: float, capacity: float) -> pd.DataFrame:
    test, valid, _ = data()
    costs = {"lostcustomer": lost, "reviewcost": reviewcost}
    y = test[config.target].to_numpy()
    limit = test["proposed_credit_limit"].to_numpy()
    older = (test["customer_age"] >= config.agegroup).to_numpy()
    k = equalising(valid, costs, capacity)
    options = {
        "Age used as a model input": test["probability_withage"].to_numpy(),
        "Vigil: age never used": test["probability"].to_numpy(),
        "Equal burden across ages": scaled(test["probability"].to_numpy(), older, k),
    }
    base = engine.realised(
        y, limit, engine.decide(options["Vigil: age never used"], limit, capacity, costs), costs
    ).sum()
    rows = []
    for name, p in options.items():
        choice = engine.decide(p, limit, capacity, costs)
        b = burden(y, older, choice)
        cost = engine.realised(y, limit, choice, costs).sum()
        rows.append(
            {
                "option": name,
                "under50": b["under50"]["good_stopped"],
                "over50": b["over50"]["good_stopped"],
                "fraud_under50": b["under50"]["fraud_stopped"],
                "fraud_over50": b["over50"]["fraud_stopped"],
                "balance": b["ratio"],
                "cost": cost,
                "extra": cost - base,
                "fraud_stopped": engine.outcome(y, limit, choice, costs)["fraud_stopped"],
            }
        )
    return pd.DataFrame(rows).set_index("option")


test, valid, cases = data()
proof = evidence()
frauds = int(test[config.target].sum())

# --- header --------------------------------------------------------------------------------
html(
    '<div class="eyebrow">Vigil · fraud screening case study</div>'
    '<div class="headline">Approve, review or decline?</div>'
    '<div class="lede">For every new account application, Vigil takes the action that is '
    "expected to cost the least: fraud let through, analyst time, or a good customer lost.</div>"
)
st.write("")

html('<div class="assumptions">Assumptions · change them and every number below updates</div>')
c1, c2, c3 = st.columns(3, gap="large")
lost = c1.select_slider(
    "Value of a good customer turned away (€)",
    [25, 50, 100, 200, 400],
    value=100,
    help="Lifetime margin lost when a legitimate applicant is declined.",
)
reviewcost = c2.slider("Analyst cost per review (€)", 5, 50, 20, 5)
capacity = (
    c3.select_slider(
        "Analysts can review",
        [0.5, 1.0, 2.5, 5.0, 10.0],
        value=2.5,
        format_func=lambda v: f"{v:g}% of applications",
        help="How many applications the fraud team can review. Every policy gets the same.",
    )
    / 100
)
costs = {"lostcustomer": float(lost), "reviewcost": float(reviewcost)}

with st.spinner("Scoring 205,011 applications under every policy…"):
    table = comparison(float(lost), float(reviewcost), capacity)
best = table.loc[ours]
rules = table.loc[display["Rules: review top alerts"]]
saved = rules["vs_vigil"]

legit = len(test) - frauds
st.write("")
kpis = st.container()
with kpis:
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(
        kpi(
            "Saved vs rules engine",
            money(saved),
            f"{saved / rules['cost']:.0%} less, same analysts",
            accent=True,
            highlight=True,
        ),
        unsafe_allow_html=True,
    )
    k2.markdown(
        kpi("Total cost", money(best["cost"]), "fraud, analysts and lost customers"),
        unsafe_allow_html=True,
    )
    k3.markdown(
        kpi("Fraud stopped", f"{best['fraud_stopped']:.0%}", f"of {frauds:,} frauds"),
        unsafe_allow_html=True,
    )
    k4.markdown(
        kpi(
            "Good customers declined",
            f"{int(best['good_declined']):,}",
            f"{best['good_declined'] / legit:.2%} of genuine applicants",
        ),
        unsafe_allow_html=True,
    )
st.write("")
st.write("")

policytab, reviewtab, drifttab, fairtab, methodtab = st.tabs(
    ["Policy comparison", "Application review", "Drift and monitoring", "Fairness", "Method"]
)

# --- policies ------------------------------------------------------------------------------
with policytab:
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        section(
            "Which policy costs the least?",
            "Total cost over the two test months, scored on what actually happened.",
        )
        bars = table.reset_index(names="policy")
        bars["ours"] = bars["policy"] == ours
        bars["text"] = [money(v) for v in bars["cost"]]
        bars["extra"] = [("+" + money(v)) if v > 0 else "" for v in bars["vs_vigil"]]
        order = bars.sort_values("cost")["policy"].tolist()
        base = alt.Chart(bars).encode(
            y=alt.Y(
                "policy:N", sort=order, title=None, axis=alt.Axis(labelLimit=280, labelPadding=8)
            )
        )
        chart = (
            base.mark_bar(height=20, cornerRadiusEnd=4).encode(
                x=alt.X(
                    "cost:Q",
                    title=None,
                    scale=alt.Scale(domain=[0, bars["cost"].max() * 1.18]),
                    axis=alt.Axis(labelExpr=eurolabels, tickCount=5),
                ),
                color=alt.condition(
                    "datum.ours", alt.value(theme.primary), alt.value(theme.context)
                ),
                tooltip=[
                    alt.Tooltip("policy:N", title="Policy"),
                    alt.Tooltip("cost:Q", format=",.0f", title="Total cost (€)"),
                    alt.Tooltip("vs_vigil:Q", format="+,.0f", title="vs Vigil (€)"),
                    alt.Tooltip("vs_vigil_low:Q", format="+,.0f", title="95% CI low"),
                    alt.Tooltip("vs_vigil_high:Q", format="+,.0f", title="95% CI high"),
                    alt.Tooltip("fraud_stopped:Q", format=".0%", title="Fraud stopped"),
                    alt.Tooltip("good_declined:Q", format=",.0f", title="Good declined"),
                ],
            )
            + base.mark_text(align="left", dx=8, fontSize=12).encode(
                x="cost:Q",
                text="text:N",
                color=alt.condition(
                    "datum.ours", alt.value(theme.foreground), alt.value(theme.mutedforeground)
                ),
            )
        ).properties(height=260)
        st.altair_chart(chart, width="stretch")

    with right:
        section(
            "How the bill responds to analyst capacity",
            "Vigil uses extra analysts only where a review pays for itself.",
        )
        curve = capacitycurve(float(lost), float(reviewcost))
        domain = [ours, display["Model: review top scores"], display["Rules: review top alerts"]]
        lines = (
            alt.Chart(curve)
            .mark_line(point=alt.OverlayMarkDef(size=40), strokeWidth=2)
            .encode(
                x=alt.X(
                    "capacity:Q",
                    scale=alt.Scale(type="log", domain=[0.45, 11], nice=False),
                    title="Analyst capacity (%)",
                    axis=alt.Axis(values=[v * 100 for v in capacitygrid], format="~g"),
                ),
                y=alt.Y(
                    "cost:Q",
                    title=None,
                    scale=alt.Scale(zero=False),
                    axis=alt.Axis(labelExpr=eurolabels, tickCount=5),
                ),
                color=alt.Color(
                    "policy:N",
                    scale=alt.Scale(
                        domain=domain, range=[theme.primary, theme.secondary, theme.mutedforeground]
                    ),
                    legend=alt.Legend(
                        orient="bottom",
                        columns=1,
                        title=None,
                        labelLimit=300,
                        symbolType="stroke",
                        symbolStrokeWidth=3,
                    ),
                ),
                strokeDash=alt.condition(
                    alt.datum.policy == domain[2], alt.value([5, 3]), alt.value([1, 0])
                ),
                tooltip=[
                    alt.Tooltip("policy:N", title="Policy"),
                    alt.Tooltip("capacity:Q", title="Capacity (%)"),
                    alt.Tooltip("cost:Q", format=",.0f", title="Total cost (€)"),
                ],
            )
        )
        marker = (
            alt.Chart(pd.DataFrame({"capacity": [capacity * 100]}))
            .mark_rule(color=theme.mutedforeground, strokeDash=[2, 3])
            .encode(x="capacity:Q")
        )
        st.altair_chart((lines + marker).properties(height=300), width="stretch")

    st.write("")
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        section(
            "Where the money goes", "The cheapest policy is not the one that stops the most fraud."
        )
        parts = []
        for name in [
            ours,
            display["Model: decline at 5% FPR"],
            display["Model: review top scores"],
            display["Rules: review top alerts"],
        ]:
            r = table.loc[name]
            for label, v in [
                ("Fraud losses", r["fraud_loss"]),
                ("Analyst time", r["reviews"] * reviewcost),
                ("Good customers turned away", r["good_declined"] * lost),
                ("Delays for good customers", r["good_reviewed"] * engine.defaultcosts["friction"]),
            ]:
                parts.append({"policy": name, "part": label, "cost": v})
        parts = pd.DataFrame(parts)
        partorder = [
            "Fraud losses",
            "Analyst time",
            "Good customers turned away",
            "Delays for good customers",
        ]
        stack = (
            alt.Chart(parts)
            .mark_bar(height=20, stroke=theme.background, strokeWidth=2)
            .encode(
                y=alt.Y(
                    "policy:N",
                    sort=parts["policy"].unique().tolist(),
                    title=None,
                    axis=alt.Axis(labelLimit=280, labelPadding=8),
                ),
                x=alt.X(
                    "cost:Q",
                    title=None,
                    stack="zero",
                    axis=alt.Axis(labelExpr=eurolabels, tickCount=5),
                ),
                color=alt.Color(
                    "part:N",
                    scale=alt.Scale(
                        domain=partorder,
                        range=[theme.primary, theme.secondary, theme.context, theme.light],
                    ),
                    legend=alt.Legend(orient="bottom", columns=2, title=None, labelLimit=260),
                ),
                order=alt.Order("partindex:Q"),
                tooltip=[
                    alt.Tooltip("policy:N", title="Policy"),
                    alt.Tooltip("part:N", title="Cost"),
                    alt.Tooltip("cost:Q", format=",.0f", title="€"),
                ],
            )
            .transform_calculate(partindex=f"indexof({json.dumps(partorder)}, datum.part)")
            .properties(height=300)
        )
        st.altair_chart(stack, width="stretch")

    with right:
        section(
            "Bigger credit lines get a closer look",
            "Share of applications sent to analysts or declined, by credit line requested.",
        )
        choice = vigilchoice(float(lost), float(reviewcost), capacity)
        bands = pd.cut(
            test["proposed_credit_limit"],
            [0, 300, 700, 1200, 3000],
            labels=["€190 to €210", "€490 to €510", "€990 to €1,000", "€1,500 to €2,100"],
        )
        share = pd.DataFrame({"band": bands, "action": np.array(engine.actions)[choice]})
        share = pd.crosstab(share["band"], share["action"], normalize="index").reset_index()
        share = share.melt(
            id_vars="band",
            value_vars=[a for a in ("review", "decline") if a in share],
            var_name="action",
            value_name="share",
        )
        grouped = (
            alt.Chart(share)
            .mark_bar(cornerRadiusEnd=4, width=alt.RelativeBandSize(0.9))
            .encode(
                x=alt.X(
                    "band:N", title="Credit line requested", sort=None, axis=alt.Axis(labelAngle=0)
                ),
                xOffset=alt.XOffset("action:N", sort=["review", "decline"]),
                y=alt.Y("share:Q", title=None, axis=alt.Axis(format=".0%", tickCount=4)),
                color=alt.Color(
                    "action:N",
                    scale=alt.Scale(
                        domain=["review", "decline"], range=[theme.secondary, theme.primary]
                    ),
                    legend=alt.Legend(
                        orient="bottom",
                        title=None,
                        labelExpr="datum.label == 'review' ? 'Sent to analysts' : 'Declined'",
                    ),
                ),
                tooltip=[
                    alt.Tooltip("band:N", title="Credit line"),
                    alt.Tooltip("action:N", title="Action"),
                    alt.Tooltip("share:Q", format=".1%", title="Share"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(grouped, width="stretch")

    st.caption(
        "Thresholds come from month 5; months 6 and 7 were never used to fit or tune anything. "
        "Every policy gets the same analyst capacity. Intervals: 100 paired bootstrap resamples. "
        "Analysts are assumed to stop 90% of the fraud they review; a review delays a genuine "
        "customer at €2."
    )

# --- application review --------------------------------------------------------------------
with reviewtab:
    c = cases.copy()
    e = engine.expected_costs(c["probability"], c["proposed_credit_limit"], costs)
    c["action"] = np.array(engine.actions)[e.argmin(axis=1)]
    saving = np.minimum(e[:, engine.approve], e[:, engine.decline]) - e[:, engine.review]
    c["saving"] = [
        money(v) if a == "review" else "" for v, a in zip(saving, c["action"], strict=True)
    ]
    c["risk"] = c["probability"].map(risk)
    c["credit"] = c["proposed_credit_limit"].map(money)
    c["decision"] = c["action"].str.capitalize()
    c["outcome"] = np.where(c["fraud"], "Fraud", "Genuine")

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        section(
            "Applications from the test months",
            "300 real applications. Select one to see how its decision is made.",
        )
        show = st.segmented_control(
            "Show",
            ["All", "Review", "Decline", "Approve"],
            default="All",
            label_visibility="collapsed",
        )
        view = c if show in (None, "All") else c[c["action"] == show.lower()]
        view = view.reset_index(drop=True)
        picked = st.dataframe(
            view[["application", "credit", "risk", "decision", "saving", "outcome"]],
            hide_index=True,
            width="stretch",
            height=420,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "application": "Application",
                "credit": "Credit line",
                "risk": st.column_config.TextColumn(
                    "Fraud risk", help="Calibrated probability that the application is fraud."
                ),
                "decision": "Decision",
                "saving": st.column_config.TextColumn(
                    "Review saves",
                    help="For reviews: expected saving over the next best action. The analyst "
                    "queue is sorted by it.",
                ),
                "outcome": st.column_config.TextColumn(
                    "Outcome", help="Known weeks later; not available when deciding."
                ),
            },
        )
    rows = picked.selection.rows if picked else []
    if len(view) == 0:
        st.info("No application in this sample gets that decision with the current costs.")
    else:
        reviews = np.flatnonzero(view["action"].to_numpy() == "review")
        first = int(reviews[0]) if len(reviews) else 0  # open on a review: the hard call
        row = view.iloc[rows[0]] if rows else view.iloc[first]
        ecost = engine.expected_costs([row["probability"]], [row["proposed_credit_limit"]], costs)[
            0
        ]
        with right:
            section(
                row["application"],
                f"Month {row['month']} · credit line {money(row['proposed_credit_limit'])} · "
                f"fraud risk {risk(row['probability'])}",
            )
            steps = [
                ("approve", "approve: fraud loss if it is fraud"),
                ("review", "review: analyst + fraud missed + delay"),
                ("decline", "decline: good customer lost if genuine"),
            ]
            body = "".join(
                f'<tr class="{"chosen" if a == row["action"] else ""}"><td></td><td>{text}</td>'
                f"<td>{money(ecost[k], 2)}</td></tr>"
                for k, (a, text) in enumerate(steps)
            )
            html(
                '<div class="card"><div class="kpilabel">Expected cost of each action</div>'
                f'<table class="steps" style="margin-top:0.5rem">{body}</table>'
                f'<div style="margin-top:0.8rem"><span class="badge {row["action"]}">'
                f"{row['action'].upper()}</span>"
                f'<span class="kpinote" style="margin-left:0.6rem">outcome, known later: '
                f"{row['outcome'].lower()}</span></div></div>"
            )
            st.write("")
            drivers = pd.DataFrame(json.loads(row["drivers"]))
            drivers["direction"] = np.where(
                drivers["impact"] > 0, "Towards fraud", "Towards genuine"
            )
            html(
                '<div class="sectiontitle" style="font-size:1.02rem">Why</div>'
                '<div class="sectionnote">Largest contributions to this score (TreeSHAP, '
                "log-odds).</div>"
            )
            why = (
                alt.Chart(drivers)
                .mark_bar(height=16, cornerRadiusEnd=4)
                .encode(
                    y=alt.Y(
                        "text:N",
                        sort=None,
                        title=None,
                        axis=alt.Axis(labelLimit=320, labelPadding=8),
                    ),
                    x=alt.X("impact:Q", title=None, axis=alt.Axis(tickCount=4, format="+.1f")),
                    color=alt.Color(
                        "direction:N",
                        scale=alt.Scale(
                            domain=["Towards fraud", "Towards genuine"],
                            range=[theme.primary, theme.secondary],
                        ),
                        legend=alt.Legend(orient="bottom", title=None),
                    ),
                    tooltip=[
                        alt.Tooltip("text:N", title="Feature"),
                        alt.Tooltip("impact:Q", format="+.3f", title="Impact"),
                    ],
                )
            )
            zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=theme.border).encode(x="x:Q")
            st.altair_chart((zero + why).properties(height=270), width="stretch")
            st.caption(
                "Categories such as housing status are anonymised in the source "
                "dataset (BA, CA, ...). One application at a time, as the API decides; "
                "the batch engine also caps reviews at analyst capacity."
            )

# --- drift and monitoring ------------------------------------------------------------------
with drifttab:
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        section(
            "Fraud keeps rising after the model is built",
            "Fraud rate per month; the model is fit on 0 to 4, tuned on 5, live from 6.",
        )
        monthly = proof["monthly"].assign(
            stage=lambda d: np.select([d["month"] <= 4, d["month"] == 5], ["Fit", "Tune"], "Live")
        )
        stages = pd.DataFrame(
            {"start": [-0.5, 4.5, 5.5], "end": [4.5, 5.5, 7.5], "stage": ["Fit", "Tune", "Live"]}
        )
        bands = (
            alt.Chart(stages)
            .mark_rect(opacity=0.14)
            .encode(
                x="start:Q",
                x2="end:Q",
                color=alt.Color(
                    "stage:N",
                    scale=alt.Scale(
                        domain=["Fit", "Tune", "Live"],
                        range=[theme.context, theme.secondary, theme.primary],
                    ),
                    legend=None,
                ),
            )
        )
        stagelabels = (
            alt.Chart(stages.assign(mid=lambda d: (d["start"] + d["end"]) / 2, y=0.0156))
            .mark_text(fontSize=11, color=theme.mutedforeground, baseline="top")
            .encode(
                x="mid:Q", y=alt.Y("y:Q", scale=alt.Scale(domain=[0.007, 0.016])), text="stage:N"
            )
        )
        rate = (
            alt.Chart(monthly)
            .mark_line(
                point=alt.OverlayMarkDef(size=48, filled=True), strokeWidth=2, color=theme.light
            )
            .encode(
                x=alt.X(
                    "month:Q",
                    title="Month",
                    axis=alt.Axis(values=list(range(8)), format="d"),
                    scale=alt.Scale(domain=[-0.5, 7.5]),
                ),
                y=alt.Y(
                    "fraud_rate:Q",
                    title=None,
                    axis=alt.Axis(format=".1%", tickCount=4),
                    scale=alt.Scale(domain=[0.007, 0.016]),
                ),
                tooltip=[
                    alt.Tooltip("month:Q", title="Month"),
                    alt.Tooltip("fraud_rate:Q", format=".2%", title="Fraud rate"),
                    alt.Tooltip("applications:Q", format=",.0f", title="Applications"),
                ],
            )
        )
        st.altair_chart((bands + stagelabels + rate).properties(height=260), width="stretch")

    with right:
        section(
            "Which inputs moved",
            "Population stability index against the fit months; above 0.25 is a large "
            "shift. Velocity features were dropped from the model for this reason.",
        )
        psi = proof["psi"]
        psi.columns = [int(c) for c in psi.columns]
        top = psi[[6, 7]].max(axis=1).sort_values(ascending=False).head(9).index
        heat = (
            psi.loc[top]
            .reset_index(names="feature")
            .melt(id_vars="feature", var_name="month", value_name="psi")
        )

        heat["label"] = heat["feature"].map(lambda f: labels.get(f, f))
        heat["shown"] = heat["psi"].clip(upper=1.0)
        heat["text"] = np.where(heat["psi"] >= 0.25, heat["psi"].map(lambda v: f"{v:.2f}"), "")
        grid = alt.Chart(heat).encode(
            x=alt.X("month:O", title="Month", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "label:N",
                sort=[labels.get(f, f) for f in top],
                title=None,
                axis=alt.Axis(labelLimit=300),
            ),
        )
        cells = grid.mark_rect(stroke=theme.background, strokeWidth=2, cornerRadius=3).encode(
            color=alt.Color(
                "shown:Q",
                scale=alt.Scale(domain=[0, 1], range=[theme.card, theme.primaryglow]),
                legend=alt.Legend(title="PSI", orient="right", format=".1f"),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Feature"),
                alt.Tooltip("month:O", title="Month"),
                alt.Tooltip("psi:Q", format=".3f", title="PSI"),
            ],
        )
        text = grid.mark_text(fontSize=11, color=theme.foreground).encode(text="text:N")
        st.altair_chart((cells + text).properties(height=260), width="stretch")

    st.write("")
    section(
        "What the monthly report would have said",
        "The share of flagged applications needs no labels and caught month 6 on the day; "
        "the false positive overrun was confirmed only once fraud outcomes matured.",
    )
    prod = proof["production"]
    header = "".join(
        f'<div class="kpilabel">{h}</div>'
        for h in [
            "month",
            "alerts",
            "flagged¹",
            "score drift¹",
            "largest input drift¹",
            "false positive rate²",
            "cost per 1,000²",
        ]
    )
    body = ""
    for _, r in prod.iterrows():
        alerts = [a for a in str(r["alerts"]).split("; ") if a and a != "nan"]
        alerts = [
            a.replace(f"({r['worst_feature']})", f"· {labels.get(r['worst_feature'], '')}")
            for a in alerts
        ]
        chips = (
            "".join(f'<span class="chip alert">{a}</span>' for a in alerts)
            or '<span class="chip">no alerts</span>'
        )
        stage = "tuning" if r["month"] == config.validmonth else "live"
        body += (
            f"<div>{int(r['month'])} <span class='kpinote'>· {stage}</span></div>"
            f"<div>{chips}</div>"
            f"<div>{r['flagged_share']:.1%}</div>"
            f"<div>{r['score_psi']:.3f}</div>"
            f"<div>{r['worst_feature_psi']:.2f}</div>"
            f"<div>{r['fpr_fixed_threshold']:.1%}</div>"
            f"<div>{money(r['cost_per_1k'])}</div>"
        )
    html(
        '<div class="card" style="min-height:0"><div class="report">'
        f"{header}{body}</div>"
        '<div class="kpinote" style="margin-top:0.6rem">¹ available the day applications '
        "arrive, no labels needed · ² needs fraud outcomes, known weeks later</div></div>"
    )

    st.write("")
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        section(
            "Retrain monthly, or train once?",
            "Cost per 1,000 applications in a backtest over months 3 to 7.",
        )
        bt = proof["backtest"].assign(
            policy=lambda d: d["policy"].map(
                {"static": "Trained once", "monthly": "Retrained monthly"}
            )
        )
        backtest = (
            alt.Chart(bt)
            .mark_line(point=alt.OverlayMarkDef(size=44), strokeWidth=2)
            .encode(
                x=alt.X("month:O", title="Month", axis=alt.Axis(labelAngle=0)),
                y=alt.Y(
                    "cost_per_1k:Q",
                    title=None,
                    scale=alt.Scale(zero=False),
                    axis=alt.Axis(labelExpr="'€' + format(datum.value, ',.0f')", tickCount=4),
                ),
                color=alt.Color(
                    "policy:N",
                    scale=alt.Scale(
                        domain=["Retrained monthly", "Trained once"],
                        range=[theme.secondary, theme.mutedforeground],
                    ),
                    legend=alt.Legend(
                        orient="bottom", title=None, symbolType="stroke", symbolStrokeWidth=3
                    ),
                ),
                strokeDash=alt.condition(
                    alt.datum.policy == "Trained once", alt.value([5, 3]), alt.value([1, 0])
                ),
                tooltip=[
                    alt.Tooltip("policy:N", title="Policy"),
                    alt.Tooltip("month:O"),
                    alt.Tooltip("cost_per_1k:Q", format=",.0f", title="€ per 1,000"),
                    alt.Tooltip("fpr_fixed_threshold:Q", format=".1%", title="False positive rate"),
                ],
            )
            .properties(height=230)
        )
        st.altair_chart(backtest, width="stretch")
    with right:
        sh = proof["monitor"]["shadow"]
        verdict = "promoted" if sh["promote"] else "not promoted"
        section(
            "Shadow test before any swap",
            "A challenger trained on one more month scored month 7 without acting.",
        )
        html(
            '<div class="card"><table class="steps">'
            f"<tr><td></td><td>live model, cost</td><td>{money(sh['champion_cost'])}</td></tr>"
            f"<tr><td></td><td>challenger, cost</td><td>{money(sh['challenger_cost'])}</td></tr>"
            f"<tr><td></td><td>challenger saving, 95% CI</td><td>{money(sh['saving_low'])} to "
            f"{money(sh['saving_high'])}</td></tr>"
            f"<tr><td></td><td>fairness balance, live vs challenger</td>"
            f"<td>{sh['champion_fairness']:.2f} vs {sh['challenger_fairness']:.2f}</td></tr>"
            f'<tr class="total"><td></td><td>decision</td><td>{verdict}</td></tr>'
            "</table></div>"
        )
        st.caption(
            "Rule: promote only if cheaper with 95% confidence and not less fair. The "
            "challenger is fairer; if that should count, the rule must say so."
        )

# --- fairness ------------------------------------------------------------------------------
with fairtab:
    with st.spinner("Pricing three fairness options…"):
        fair = fairnessoptions(float(lost), float(reviewcost), capacity)
    fr = proof["fairness"]["fraud_rate"]
    section(
        "Who carries the false alarms?",
        f"Fraud is {fr['over50'] / fr['under50']:.1f} times more common among applicants "
        f"aged 50 or more ({fr['over50']:.1%} vs {fr['under50']:.1%}), so getting the risk "
        "right for both groups also stops more of their genuine applicants. Equal error "
        "rates and equal calibration cannot both hold when base rates differ: any fix is a "
        "choice, so each one is priced.",
    )
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        bars = fair.reset_index().melt(
            id_vars="option", value_vars=["under50", "over50"], var_name="group", value_name="share"
        )
        bars["group"] = bars["group"].map({"under50": "Under 50", "over50": "50 or older"})
        chart = (
            alt.Chart(bars)
            .mark_bar(cornerRadiusEnd=4, width=alt.RelativeBandSize(0.9))
            .encode(
                x=alt.X(
                    "option:N",
                    sort=list(fair.index),
                    title=None,
                    axis=alt.Axis(labelAngle=0, labelLimit=200),
                ),
                xOffset=alt.XOffset("group:N", sort=["Under 50", "50 or older"]),
                y=alt.Y(
                    "share:Q",
                    title="Genuine applicants stopped",
                    axis=alt.Axis(format=".0%", tickCount=4),
                ),
                color=alt.Color(
                    "group:N",
                    scale=alt.Scale(
                        domain=["Under 50", "50 or older"], range=[theme.secondary, theme.primary]
                    ),
                    legend=alt.Legend(orient="bottom", title=None),
                ),
                tooltip=[
                    alt.Tooltip("option:N", title="Option"),
                    alt.Tooltip("group:N"),
                    alt.Tooltip("share:Q", format=".2%", title="Genuine applicants stopped"),
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, width="stretch")
    with right:
        for name, r in fair.iterrows():
            shipped = name == "Vigil: age never used"
            extra = (
                "the reference"
                if shipped
                else (("+" if r["extra"] > 0 else "") + money(r["extra"]) + " vs Vigil")
            )
            html(
                f'<div class="card{" highlight" if shipped else ""}" '
                'style="min-height:0;margin-bottom:0.6rem">'
                f'<div class="kpilabel">{name}</div>'
                '<div style="display:flex;gap:2rem;margin-top:0.35rem;flex-wrap:wrap">'
                f'<div><div class="kpinote">balance (1 = equal)</div>'
                f'<div class="kpivalue" style="font-size:1.35rem">{r["balance"]:.2f}</div></div>'
                f'<div><div class="kpinote">cost</div>'
                f'<div class="kpivalue" style="font-size:1.35rem">{extra}</div></div>'
                f'<div><div class="kpinote">fraud stopped, 50+</div>'
                f'<div class="kpivalue" style="font-size:1.35rem">{r["fraud_over50"]:.0%}</div>'
                "</div></div></div>"
            )
    st.markdown(
        "- **Dropping age costs nothing measurable and narrows the gap**, so Vigil never uses it "
        "to score. Other features still carry part of the age signal.\n"
        "- **Equal burden closes most of the gap at a price**, and stops less fraud among older "
        "applicants. It also means using age *in the decision*, which is differential treatment: "
        "a call for legal and compliance, not for the model. This page gives them the price.\n"
        "- **Fairness drifts too.** The equal burden factor is set on month 5 and does not stay "
        "exact on later months, so group error rates are monitored like any other metric."
    )

# --- method --------------------------------------------------------------------------------
with methodtab:
    m = proof["model"]
    boot = m["bootstrap_vs_logistic"]["lightgbm_stable"]["recall_at_5fpr"]
    lat = proof["latency"]
    html(
        '<div class="formula" style="margin:0 0 0.9rem 0">cheapest of · <b>approve</b> '
        "p × credit line · <b>review</b> analyst time + fraud the analyst misses · "
        "<b>decline</b> (1 − p) × value of a good customer</div>"
    )
    html(
        '<div class="split"><div class="fit">Fit · months 0 to 4 · 676k applications</div>'
        '<div class="tune">Tune · month 5</div>'
        '<div class="test">Test · months 6 and 7 · never seen</div></div>'
    )
    st.write("")
    c1, c2, c3 = st.columns(3, gap="medium")
    c1.markdown(
        kpi(
            "Model",
            f"+{boot['diff'] * 100:.1f} pp",
            "more fraud caught than logistic regression at 5% false positives "
            f"(95% CI +{boot['diff_low'] * 100:.1f} to +{boot['diff_high'] * 100:.1f}); LightGBM "
            "without drifting features or age",
        ),
        unsafe_allow_html=True,
    )
    c2.markdown(
        kpi(
            "Decision",
            "Expected cost",
            "three actions priced per application, analyst capacity filled where a review saves "
            "the most; thresholds never tuned on test months",
        ),
        unsafe_allow_html=True,
    )
    c3.markdown(
        kpi(
            "Serving",
            f"{lat['server_p99_ms']:.0f} ms p99",
            f"server time at {lat['concurrency']} concurrent requests; FastAPI, reasons from "
            "TreeSHAP for reviews and declines",
        ),
        unsafe_allow_html=True,
    )
    st.write("")
    names = {
        "rules": "Rules engine (7 red flags)",
        "logistic": "Logistic regression",
        "lightgbm": "LightGBM, all features",
        "lightgbm_stable": "LightGBM, stable (shipped)",
    }
    models = pd.DataFrame(m["models"]).T.rename(index=names)
    for col in ("recall_at_5fpr", "fpr_fixed_threshold"):
        models[col] = models[col].astype(float) * 100
    st.dataframe(
        models[["recall_at_5fpr", "pr_auc", "roc_auc", "fpr_fixed_threshold"]],
        width="stretch",
        column_config={
            "recall_at_5fpr": st.column_config.NumberColumn(
                "Fraud caught at 5% FPR", format="%.1f%%"
            ),
            "pr_auc": st.column_config.NumberColumn("PR-AUC", format="%.3f"),
            "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.3f"),
            "fpr_fixed_threshold": st.column_config.NumberColumn(
                "FPR, threshold fixed on month 5",
                format="%.1f%%",
                help="Target 5%. Shows how much the drifting features move the error rate.",
            ),
        },
    )
    with st.expander("Assumptions and limitations"):
        st.markdown(
            "- **The data is synthetic**, generated from real bank data with privacy protection "
            "(Feedzai BAF, NeurIPS 2022). Categories are anonymised.\n"
            "- **Costs are illustrative.** A fraudster is assumed to draw the full credit line; "
            "analysts stop 90% of the fraud they review. Change the main costs above.\n"
            "- **Labels are assumed to mature** before the monthly report; in production fraud "
            "outcomes arrive with a delay, which is why label-free signals matter.\n"
            "- **Analyst capacity is a share of volume.** A real queue would also model "
            "arrival times and service levels."
        )
    st.markdown(
        f"Code, full write-up and tests: [{repourl.removeprefix('https://')}]({repourl}) · "
        "Data: Bank Account Fraud dataset suite, Feedzai (NeurIPS 2022), CC BY-NC-SA 4.0."
    )
