#!/usr/bin/env python3
"""Figure 1(b): the official score and the action decision on the same 223 twin pairs.

Numbers come from analysis/p1s_attempt_decision.md (definition C, 223 certified pairs) and
analysis/p1v_c_companions.md §2 (official raw on exactly those 223 miss items). Style follows the flat-vector cartoon in fig1a_raw.png, which is
panel (a); this script also writes the side-by-side composite fig1_combined.png.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

HERE = Path(__file__).resolve().parent
RAW = HERE / "fig1a_raw.png"

for _f in Path("/usr/share/fonts/truetype/lato").glob("Lato-*.ttf"):
    font_manager.fontManager.addfont(str(_f))

INK = "#181b1d"
GREY = "#c4c3c1"
GREY_DARK = "#908e8d"
WHITE = "#ffffff"
GREEN = "#398d48"
RED = "#b44139"

mpl.rcParams.update({
    "font.family": "Lato",
    "font.sans-serif": ["Lato", "DejaVu Sans"],
    "figure.facecolor": WHITE, "axes.facecolor": WHITE, "savefig.facecolor": WHITE,
    "text.color": INK, "svg.fonttype": "none", "pdf.fonttype": 42,
})

# analysis/p1s_attempt_decision.md (definition C, 223 certified pairs) + p1v_c_companions.md §2.
# name: (official raw on the 223 miss items, decision accuracy)
DATA = {
    "gpt-5.4":         (32.7, 80.7),
    "gemma-4-31B-it":  (50.7, 78.5),
    "Qwen3.8-Max":     (47.1, 78.3),
    "Qwen3.5-9B":      (52.5, 77.4),
    "Qwen3.6-27B":     (58.7, 77.1),
    "deepseek-v4-flash": (52.0, 75.8),
    "gemma-4-E4B-it":  (22.9, 67.7),
}
ACCENT = {"gpt-5.4": RED, "Qwen3.6-27B": GREEN}


def main() -> None:
    # Panel (b) is pasted next to the cartoon and lands at about 0.43 of its drawn size
    # on the page, so every size below is scaled by K to keep the smallest text near 7 pt.
    K = 1.65
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    xl, xr = 0.0, 1.0


    for name, (lo, hi) in DATA.items():
        c = ACCENT.get(name, GREY_DARK)
        acc = name in ACCENT
        ax.plot([xl, xr], [lo, hi], color=c, linewidth=(2.8 if acc else 1.5) * K,
                zorder=5 if acc else 3, solid_capstyle="round", alpha=1.0 if acc else 0.55)
        for x, v in ((xl, lo), (xr, hi)):
            ax.plot([x], [v], "o", color=c, markersize=(7.5 if acc else 5) * K,
                    markeredgecolor=WHITE, markeredgewidth=1.2 * K, zorder=6 if acc else 4)
        if acc:
            ax.text(xl - 0.06, lo, f"{lo:.1f}", ha="right", va="center", fontsize=10.5 * K,
                    color=c, fontweight="bold", zorder=7)
            ax.text(xr + 0.06, hi, f"{hi:.1f}", ha="left", va="center", fontsize=10.5 * K,
                    color=c, fontweight="bold", zorder=7)

    ax.text(xl, 96.5, "official score", ha="center", va="bottom", fontsize=11.5 * K,
            fontweight="bold", color=INK)
    ax.text(xl, 91.5, "on these 223 items", ha="center", va="bottom", fontsize=9.5 * K, color=GREY_DARK)
    ax.text(xr, 96.5, "action decision", ha="center", va="bottom", fontsize=11.5 * K,
            fontweight="bold", color=INK)
    ax.text(xr, 91.5, "on the same 223 pairs", ha="center", va="bottom", fontsize=9.5 * K, color=GREY_DARK)

    handles = [
        mpl.lines.Line2D([], [], color=RED, linewidth=2.8 * K, marker="o", markersize=7 * K,
                         markeredgecolor=WHITE, label="gpt-5.4"),
        mpl.lines.Line2D([], [], color=GREEN, linewidth=2.8 * K, marker="o", markersize=7 * K,
                         markeredgecolor=WHITE, label="Qwen3.6-27B"),
        mpl.lines.Line2D([], [], color=GREY_DARK, linewidth=1.5 * K, alpha=0.55, marker="o",
                         markersize=5 * K, markeredgecolor=WHITE, label="five other models"),
    ]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.03, 0.0), frameon=False,
              fontsize=10 * K, handlelength=1.8, labelspacing=0.5, labelcolor="linecolor")

    ax.set_xlim(-0.55, 1.58)
    ax.set_ylim(15, 101)
    ax.set_yticks([25, 50, 75], ["25%", "50%", "75%"])
    ax.tick_params(axis="y", labelsize=11 * K, colors=INK, width=2.0 * K, length=6 * K)
    ax.tick_params(axis="x", length=0, labelbottom=False)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["left"].set_linewidth(2.0 * K)
    ax.spines["left"].set_bounds(15, 85)

    fig.subplots_adjust(left=0.15, right=0.99, top=0.97, bottom=0.03)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(HERE / f"fig1b_reversal.{ext}", **({"dpi": 220} if ext == "png" else {}))
    plt.close(fig)

    a = Image.open(RAW).convert("RGB")
    b = Image.open(HERE / "fig1b_reversal.png").convert("RGB")
    b = b.resize((round(b.width * a.height / b.height), a.height), Image.LANCZOS)
    out = Image.new("RGB", (a.width + b.width + 40, a.height), "white")
    out.paste(a, (0, 0))
    out.paste(b, (a.width + 40, 0))
    out.save(HERE / "fig1_combined.png")


if __name__ == "__main__":
    main()
