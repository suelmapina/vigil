"""README figures, rendered from saved results only (no retraining).

uv run python -m vigil.figures
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vigil import config, theme

out = config.reportsdir / "figures"
names = {
    "Vigil: cost-based decision": "Vigil: cost-based decision",
    "Model: review top scores": "Model score, analysts on top scores",
    "Model: decline at 5% FPR": "Model score, decline above 5% FPR",
    "Rules: review top alerts": "Rules engine, analysts on top alerts",
    "Approve everyone": "Approve everyone",
}


def money(v: float) -> str:
    return f"€{v / 1e6:.2f}M" if v >= 1e6 else f"€{v / 1e3:.0f}k"


def policies() -> None:
    p = json.loads((config.reportsdir / "decision" / "metrics.json").read_text())["policies"]
    rows = [(names[k], v["cost"]) for k, v in p.items() if k in names]
    rows.sort(key=lambda r: r[1], reverse=True)
    labels, costs = zip(*rows, strict=True)
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    colors = [theme.primary if lab.startswith("Vigil") else theme.context for lab in labels]
    ax.barh(labels, costs, height=0.56, color=colors)
    for i, c in enumerate(costs):
        ax.text(
            c + 25e3,
            i,
            money(c),
            va="center",
            fontsize=9,
            color=theme.foreground if labels[i].startswith("Vigil") else theme.mutedforeground,
        )
    ax.set_xlim(0, max(costs) * 1.16)
    ax.xaxis.set_major_formatter(
        lambda v, _: "€0" if v == 0 else (f"€{v / 1e6:.1f}M" if v >= 1e6 else f"€{v / 1e3:.0f}k")
    )
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_title("Total cost of fraud and screening, two unseen months", pad=28)
    theme.subtitle(ax, "205,011 applications, same analyst capacity for every policy")
    fig.tight_layout()
    fig.savefig(out / "policies.png")
    plt.close(fig)


def monthly() -> None:
    m = pd.read_csv(config.reportsdir / "data" / "monthly.csv")
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    for start, end, label, color in [
        (-0.5, 4.5, "fit", theme.context),
        (4.5, 5.5, "tune", theme.secondary),
        (5.5, 7.5, "live", theme.primary),
    ]:
        ax.axvspan(start, end, color=color, alpha=0.14, lw=0)
        ax.text(
            (start + end) / 2,
            0.0157,
            label,
            ha="center",
            va="top",
            fontsize=9,
            color=theme.mutedforeground,
        )
    ax.plot(m["month"], m["fraud_rate"], color=theme.light, lw=2, marker="o", ms=5)
    ax.set_ylim(0.007, 0.016)
    ax.set_xlim(-0.5, 7.5)
    ax.set_xticks(range(8))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1%}")
    ax.set_xlabel("Month")
    ax.set_title("Fraud keeps rising after the model is built", pad=28)
    theme.subtitle(ax, "Fraud rate per month; fit on months 0 to 4, tuned on 5, live from 6")
    fig.tight_layout()
    fig.savefig(out / "monthly.png")
    plt.close(fig)


def fairness() -> None:
    f = json.loads((config.reportsdir / "fairness" / "metrics.json").read_text())
    groups = pd.DataFrame(f["groups"])
    order = ["With the age feature", "Vigil", "Equal burden across age groups"]
    shown = ["Age used as an input", "Vigil: age never used", "Equal burden across ages"]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    for offset, group, color, label in [
        (-0.18, "under50", theme.secondary, "Under 50"),
        (0.18, "over50", theme.primary, "50 or older"),
    ]:
        values = [
            groups.loc[(groups["option"] == o) & (groups["group"] == group), "good_stopped"].item()
            for o in order
        ]
        ax.bar(x + offset, values, width=0.34, color=color, label=label)
        for xi, v in zip(x + offset, values, strict=True):
            ax.text(
                xi, v + 0.0015, f"{v:.1%}", ha="center", fontsize=8.5, color=theme.mutedforeground
            )
    ax.set_xticks(x, shown)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylim(0, 0.085)
    ax.legend(loc="upper right")
    ax.set_title("Genuine applicants reviewed or declined, by age", pad=28)
    theme.subtitle(ax, "Fraud is 2.7 times more common at 50+, so equal treatment has a price")
    fig.tight_layout()
    fig.savefig(out / "fairness.png")
    plt.close(fig)


def main() -> None:
    out.mkdir(parents=True, exist_ok=True)
    theme.use_matplotlib()
    policies()
    monthly()
    fairness()
    print(f"✓ figures in {out}")


if __name__ == "__main__":
    main()
