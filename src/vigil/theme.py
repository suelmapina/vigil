"""Portfolio visual identity (same tokens as suelmapina.github.io) for figures and the demo."""

from __future__ import annotations

background = "#0d0f16"
card = "#181b25"
muted = "#272935"
border = "#2b2e3b"
foreground = "#f1f5f9"
mutedforeground = "#8596ad"
primary = "#8c47d1"
primaryglow = "#bb67e4"
secondary = "#26cad9"
destructive = "#ef4444"
context = "#4a5068"  # de-emphasised series
light = "#cbd5e1"

sans = "Space Grotesk"
mono = "JetBrains Mono"
googlefonts = (
    "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700"
    "&family=JetBrains+Mono:wght@400;500&display=swap"
)


def use_matplotlib() -> None:
    """Dark, minimal matplotlib style for README figures."""
    from pathlib import Path

    import matplotlib as mpl
    from matplotlib import font_manager

    fontdir = Path(__file__).resolve().parents[2] / "assets" / "fonts"
    for fontfile in fontdir.glob("SpaceGrotesk-*.ttf"):
        font_manager.fontManager.addfont(str(fontfile))

    mpl.rcParams.update(
        {
            "figure.facecolor": background,
            "savefig.facecolor": background,
            "axes.facecolor": background,
            "axes.edgecolor": border,
            "axes.labelcolor": mutedforeground,
            "axes.titlecolor": foreground,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.titlelocation": "left",
            "axes.titlepad": 14,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": border,
            "grid.linewidth": 0.8,
            "xtick.color": mutedforeground,
            "ytick.color": mutedforeground,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.color": foreground,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": light,
            "font.family": "sans-serif",
            "font.sans-serif": [sans, "DejaVu Sans", "Arial"],
            "figure.dpi": 100,
            "savefig.dpi": 160,
        }
    )


def subtitle(ax, text: str) -> None:
    """Grey line under the title, for the 'how to read this' sentence."""
    ax.text(0, 1.02, text, transform=ax.transAxes, fontsize=9, color=mutedforeground, va="bottom")


def altair_theme() -> dict:
    return {
        "config": {
            "background": "transparent",
            "font": sans,
            "view": {"stroke": "transparent"},
            "axis": {
                "domainColor": border,
                "gridColor": border,
                "tickColor": border,
                "labelColor": mutedforeground,
                "titleColor": mutedforeground,
                "labelFont": sans,
                "titleFont": sans,
                "titleFontWeight": 500,
                "labelFontSize": 12,
                "titleFontSize": 12,
            },
            "legend": {"labelColor": light, "titleColor": mutedforeground, "labelFont": sans},
            "title": {"color": foreground, "font": sans, "anchor": "start", "fontSize": 15},
            "text": {"color": foreground, "font": sans},
        }
    }


css = f"""
<style>
@import url('{googlefonts}');

html, body, .stApp, .stMarkdown, button, input, textarea, [data-testid="stWidgetLabel"] p {{
  font-family: '{sans}', sans-serif;
}}
.stApp {{
  background:
    radial-gradient(1100px 520px at 0% -10%, rgba(140, 71, 209, 0.14), transparent 60%),
    radial-gradient(900px 480px at 100% 0%, rgba(38, 202, 217, 0.07), transparent 60%),
    {background};
}}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ display: none; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{
  max-width: 1480px; padding: 2.6rem clamp(1rem, 3vw, 3rem) 4rem clamp(1rem, 3vw, 3rem);
}}

.eyebrow {{
  font-family: '{mono}', monospace; font-size: 0.72rem; letter-spacing: 0.14em;
  text-transform: uppercase; color: {secondary}; margin-bottom: 0.4rem;
}}
.headline {{
  font-size: 2.7rem; font-weight: 700; line-height: 1.1; margin: 0 0 0.6rem 0;
  background: linear-gradient(135deg, {primary}, {secondary});
  -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.lede {{ color: {mutedforeground}; font-size: 1.05rem; line-height: 1.6; max-width: 900px; }}
.formula {{
  display: inline-block; font-family: '{mono}', monospace; font-size: 0.85rem;
  color: {foreground}; background: {card}; border: 1px solid {border};
  border-radius: 8px; padding: 0.45rem 0.8rem; margin-top: 0.8rem;
}}
.formula b {{ color: {primaryglow}; font-weight: 500; }}

.card {{
  background: rgba(24, 27, 37, 0.72); border: 1px solid {border};
  border-radius: 12px; padding: 1.05rem 1.2rem; height: 100%; min-height: 128px;
}}
.card.highlight {{
  border-color: rgba(140, 71, 209, 0.55); box-shadow: 0 0 24px rgba(140, 71, 209, 0.15);
}}
.kpilabel {{
  font-family: '{mono}', monospace; font-size: 0.68rem; letter-spacing: 0.1em;
  text-transform: uppercase; color: {mutedforeground};
}}
.kpivalue {{ font-size: 1.85rem; font-weight: 600; color: {foreground}; margin-top: 0.25rem; }}
.kpivalue.accent {{ color: {primaryglow}; }}
.kpinote {{ font-size: 0.82rem; color: {mutedforeground}; margin-top: 0.15rem; }}

.sectiontitle {{
  font-size: 1.25rem; font-weight: 600; color: {foreground}; margin: 0.4rem 0 0.2rem;
}}
.sectionnote {{ color: {mutedforeground}; font-size: 0.92rem; margin-bottom: 0.8rem; }}

.badge {{
  display: inline-block; font-family: '{mono}', monospace; font-size: 0.72rem;
  padding: 0.18rem 0.55rem; border-radius: 999px; border: 1px solid {border};
}}
.badge.go {{ color: {foreground}; background: rgba(140, 71, 209, 0.35); border-color: {primary}; }}
.badge.stop {{ color: {mutedforeground}; background: {muted}; }}

table.steps {{
  width: 100%; border-collapse: collapse; font-family: '{mono}', monospace; font-size: 0.85rem;
  color: {light}; margin: 0;
}}
table.steps td {{ border: none; padding: 0.28rem 0; }}
table.steps td:first-child {{ width: 1.4rem; color: {mutedforeground}; }}
table.steps td:last-child {{ text-align: right; }}
table.steps tr.total td {{
  border-top: 1px solid {border}; padding-top: 0.55rem; color: {foreground}; font-weight: 500;
}}

.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid {border}; }}
.stTabs [data-baseweb="tab"] {{ padding: 0.6rem 0; color: {mutedforeground}; }}
.stTabs [aria-selected="true"] {{ color: {foreground}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: {primary}; }}

[data-testid="stVerticalBlockBorderWrapper"] {{ border-color: {border}; border-radius: 12px; }}
a {{ color: {secondary}; }}
</style>
"""
