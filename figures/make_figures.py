#!/usr/bin/env python3
"""Generate Figure 2 of the paper (the prompt-injection slopegraph, fig4_goodhart.*).

Style follows ``make_fig1b.py``: Lato, white surface, ink rules, one accent colour, no
grid, direct labels instead of legend boxes. The figure is exactly 5.5 in wide and is
placed at 100% scale. It packs six panels into that width, so its panel sub-labels and
footer run at 5.8-6.2 pt; everything a reader must read there is a number or a bold title.

Outputs PNG (dpi 300), PDF and SVG. Read-only with respect to experimental data.

Number provenance (every value below is copied from a generated report):
  attempt rows and the gpt-5.4 base-twin raw row  analysis/p1t_phase5_attempt.md
  official raw, miss_func and miss_param          analysis/phase5_mechanical.md
  decision accuracy under each arm                analysis/p1w_arms_decision.md
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent

# --- fonts -----------------------------------------------------------------
# Lato, same registration as make_fig1b.py (falls back to DejaVu Sans without it).
for _f in Path("/usr/share/fonts/truetype/lato").glob("Lato-*.ttf"):
    font_manager.fontManager.addfont(str(_f))

# --- palette (identical to make_fig1b.py, sampled from fig1a_raw.png) -------
INK = "#181b1d"
GREY_DARK = "#908e8d"
WHITE = "#ffffff"
RED = "#b44139"
SLATE = "#46596c"

mpl.rcParams.update(
    {
        "font.family": "Lato",
        "font.sans-serif": ["Lato", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.edgecolor": INK,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

WIDTH = 5.5  # single-column body width, inches


def save_exact(fig: plt.Figure, stem: str) -> None:
    """Save without bbox_inches so the on-disk width is exactly `figsize`."""
    for ext in ("png", "pdf", "svg"):
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(HERE / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


# ===========================================================================
# Figure 2: the prompt-injection slopegraph
# ===========================================================================

# Numbers: analysis/p1t_phase5_attempt.md (attempt rows and the gpt-5.4 base-twin raw row,
# on the 223 certified pairs), analysis/phase5_mechanical.md (official raw, miss_func and
# miss_param), analysis/p1w_arms_decision.md (decision accuracy, 223 certified pairs). Each
# entry is (neutral, injected, McNemar p); p is kept for provenance and is quoted in the text.
# The Qwen3.5-9B-FC pro-action arm is absent: its handler ignored the injection (dose 0).
FIG4 = {
    "pro-action": {
        "attempt": {"miss_param": (11.4, 22.8, "5.7\\times10^{-6}"),
                    "miss_func": (18.4, 22.1, "0.14")},
        "raw": {"miss_param": (31.5, 45.0, "3.6\\times10^{-4}"),
                "miss_func": (33.5, 49.5, "9.1\\times10^{-6}"),
                "base": (39.2, 62.7, "2.1\\times10^{-7}")},
    },
    "pro-caution": {
        "attempt": {"miss_param": (21.7, 12.0, "4.0\\times10^{-5}"),
                    "miss_func": (33.7, 27.4, "0.0018")},
        "raw": {"miss_param": (48.0, 48.5, "1"),
                "miss_func": (54.0, 49.0, "0.053")},
    },
}

CAT_STYLE = {"miss_param": "solid", "miss_func": (0, (2.6, 1.4)), "base": (0, (0.8, 1.4)),
             "decision": "solid"}
CAT_LABEL = {"miss_param": "miss_param", "miss_func": "miss_func", "base": "base twins"}

# Paired decision accuracy on the 223 certified pairs under each arm: analysis/p1w_arms_decision.md.
# (neutral, injected, change with 95% cluster-bootstrap CI; the CI is quoted in the text, not drawn)
FIG4_DECISION = {
    "pro-action": (80.7, 81.8, -2.5, 4.8),   # change +1.1 [-2.5, +4.8]
    "pro-caution": (78.5, 83.0, 1.8, 7.1),   # change +4.5 [+1.8, +7.1]
}

ARMS4 = [
    ("pro-action", RED, "gpt-5.4\n“do not ask”"),
    ("pro-caution", SLATE, "gemma-4-31B-it\n“ask first”"),
]
ARM_COLOUR = {arm: c for arm, c, _ in ARMS4}
# (key, header, ylim, ticks); every group has its own scale, the three quantities differ
GROUPS4 = [
    ("attempt", "World-changing call, withheld turn", (7, 45), (10, 20, 30, 40)),
    ("raw", "Official score", (26, 70), (30, 40, 50, 60)),
    ("decision", "Decision accuracy, 223 pairs", (72, 89), (75, 80, 85)),
]


def _spread(vals, gap):
    """Label offsets: keep the order, push apart anything closer than `gap`, recentre."""
    order = sorted(vals, key=lambda c: vals[c])
    pos, prev = {}, None
    for c in order:
        y = vals[c] if prev is None else max(vals[c], prev + gap)
        pos[c] = y
        prev = y
    shift = (sum(pos.values()) - sum(vals.values())) / len(vals)
    return {c: pos[c] - shift - vals[c] for c in vals}


def _slope_panel(ax, data, title, *, colour, ylim, ticks, show_y, right_x=2.3):
    """One measure, neutral -> injected, one line per key. `colour` is a colour or a
    dict per key; the change sits in a column right of the injected value."""
    xs = (0.0, 1.0)
    cats = list(data)
    gap = 0.11 * (ylim[1] - ylim[0])
    left = _spread({c: data[c][0] for c in cats}, gap)
    right = _spread({c: data[c][1] for c in cats}, gap)
    for cat in cats:
        a, b, *_ = data[cat]
        col = colour[cat] if isinstance(colour, dict) else colour
        ax.plot(xs, (a, b), color=col, linewidth=1.5, linestyle=CAT_STYLE[cat],
                solid_capstyle="round", dash_capstyle="round", zorder=3)
        ax.scatter(xs, (a, b), s=13, facecolor=col, edgecolor=WHITE, linewidth=0.5, zorder=4)
        ax.text(-0.17, a + left[cat], f"{a:.1f}", ha="right", va="center", fontsize=6.6,
                fontweight="bold", color=INK, zorder=6)
        ax.text(1.17, b + right[cat], f"{b:.1f}", ha="left", va="center", fontsize=6.6,
                fontweight="bold", color=INK, zorder=6)

    ax.set_xlim(-1.3, right_x)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    tr = ax.get_xaxis_transform()  # x in data units, y in axes fraction
    for x, lab, ha in ((0.15, "neutral", "right"), (0.85, "injected", "left")):
        ax.text(x, -0.045, lab, transform=tr, ha=ha, va="top", fontsize=6.2, color=GREY_DARK)
    ax.set_yticks(list(ticks))
    if show_y:
        ax.tick_params(axis="y", length=3, width=1.0, labelsize=6.2, colors=INK, pad=2)
        ax.text(-1.22, ticks[-1] + 0.03 * (ylim[1] - ylim[0]), "%", ha="left", va="bottom", fontsize=6.2, color=GREY_DARK)
    else:
        ax.tick_params(axis="y", length=3, width=1.0, labelleft=False, colors=INK)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["left"].set_bounds(ticks[0], ticks[-1])
    ax.set_title(title, loc="left", fontsize=6.8, fontweight="bold",
                 color=colour if not isinstance(colour, dict) else INK, pad=4)


def fig4_goodhart() -> None:
    fig = plt.figure(figsize=(WIDTH, 1.80))
    ph, ybot = 0.58, 0.17
    W = WIDTH
    margin, pw, gap_in, gap_grp, pw3 = 0.20 / W, 0.82 / W, 0.05 / W, 0.10 / W, 0.95 / W
    x = 0.02 / W
    for key, header, ylim, ticks in GROUPS4:
        x += margin
        fig.text(x - 0.030, 0.960, header, ha="left", va="center", fontsize=7.4,
                 fontweight="bold", color=INK)
        if key == "decision":
            ax = fig.add_axes([x, ybot, pw3, ph])
            data = {arm: FIG4_DECISION[arm] for arm, _, _ in ARMS4}
            for arm in data:
                CAT_STYLE[arm] = "solid"
            _slope_panel(ax, data, "", colour=ARM_COLOUR, ylim=ylim, ticks=ticks,
                         show_y=True)
            for j, (arm, col, title) in enumerate(ARMS4):
                ax.text(0.0, 1.03 + (1 - j) * 0.13, title.replace("\n", " "),
                        transform=ax.transAxes, ha="left", va="bottom", fontsize=6.8,
                        fontweight="bold", color=col)
            x += pw3
        else:
            for i, (arm, colour, title) in enumerate(ARMS4):
                ax = fig.add_axes([x, ybot, pw, ph])
                _slope_panel(ax, FIG4[arm][key], title, colour=colour, ylim=ylim, ticks=ticks,
                             show_y=(i == 0))
                x += pw + (gap_in if i == 0 else 0)
        x += gap_grp

    # line-style key, bottom left, one row
    xk = 0.060
    for cat in ("miss_param", "miss_func", "base"):
        fig.add_artist(plt.Line2D([xk, xk + 0.034], [0.035, 0.035], color=GREY_DARK,
                                  linewidth=1.5, linestyle=CAT_STYLE[cat],
                                  transform=fig.transFigure, dash_capstyle="round"))
        fig.text(xk + 0.040, 0.035, CAT_LABEL[cat], ha="left", va="center", fontsize=6.2,
                 color=GREY_DARK)
        xk += 0.040 + 0.0082 * len(CAT_LABEL[cat]) + 0.022

    save_exact(fig, "fig4_goodhart")


def main() -> None:
    fig4_goodhart()
    print("Generated fig4_goodhart as PNG/PDF/SVG in", HERE)


if __name__ == "__main__":
    main()
