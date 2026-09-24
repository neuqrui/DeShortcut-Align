#!/usr/bin/env python3
"""ICLR-style single-panel comparison: GRPO vs DeShortcut-Align on DSR / FRR."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

BASELINE = {"name": "GRPO", "dsr": 98.67, "frr": 98.67}
METHOD = {"name": "DeShortcut-Align", "dsr": 98.00, "frr": 32.89}

C_BASE = "#4C72B0"
C_OURS = "#DD8452"
C_DELTA = "#B03A3A"
C_GUIDE = "#5A5A5A"
C_GRID = "#ECECEC"
C_SPINE = "#2F2F2F"


def _setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.labelsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def plot(out_dir: Path) -> tuple[Path, Path]:
    _setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [r"DSR $\uparrow$", r"FRR $\downarrow$"]
    b_vals = np.array([BASELINE["dsr"], BASELINE["frr"]], dtype=float)
    m_vals = np.array([METHOD["dsr"], METHOD["frr"]], dtype=float)

    x = np.arange(len(metrics), dtype=float)
    width = 0.30
    offset = 0.23

    fig, ax = plt.subplots(figsize=(5.0, 3.35))
    fig.patch.set_facecolor("white")

    bars_b = ax.bar(
        x - offset,
        b_vals,
        width,
        label=BASELINE["name"],
        color=C_BASE,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    bars_m = ax.bar(
        x + offset,
        m_vals,
        width,
        label=METHOD["name"],
        color=C_OURS,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

    for bar in list(bars_b) + list(bars_m):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1.6,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=C_SPINE,
            clip_on=False,
            zorder=6,
        )

    # FRR drop: short guides in the gap; arrow tip lands on the 32.89 bar top
    x_grpo = x[1] - offset
    x_method = x[1] + offset
    y_hi, y_lo = BASELINE["frr"], METHOD["frr"]
    drop = y_hi - y_lo

    x_grpo_right = x_grpo + width / 2
    x_method_left = x_method - width / 2
    # tip slightly inside the orange bar so it clearly sits on the 32.89 head
    x_arrow = x_method_left + 0.03

    ax.plot(
        [x_grpo_right, x_arrow],
        [y_hi, y_hi],
        color=C_GUIDE,
        linestyle=(0, (1.5, 1.2)),
        linewidth=1.15,
        zorder=4,
        clip_on=False,
    )
    ax.plot(
        [x_arrow, x_arrow + 0.10],
        [y_lo, y_lo],
        color=C_GUIDE,
        linestyle=(0, (1.5, 1.2)),
        linewidth=1.15,
        zorder=4,
        clip_on=False,
    )
    ax.annotate(
        "",
        xy=(x_arrow, y_lo),
        xytext=(x_arrow, y_hi),
        arrowprops=dict(
            arrowstyle="-|>",
            color=C_GUIDE,
            lw=1.25,
            mutation_scale=13,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=5,
        clip_on=False,
    )
    ax.text(
        (x_grpo_right + x_arrow) / 2,
        (y_hi + y_lo) / 2,
        rf"$-{drop:.2f}$",
        ha="center",
        va="center",
        fontsize=10,
        color=C_DELTA,
        fontweight="bold",
        zorder=6,
        clip_on=False,
    )

    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 112)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_xlim(-0.55, 1.55)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.55, color=C_GRID, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(C_SPINE)
        ax.spines[spine].set_linewidth(0.9)
    ax.tick_params(colors=C_SPINE, length=3.5)

    leg = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=2,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.4,
        borderaxespad=0.0,
    )
    leg.get_texts()[1].set_fontweight("bold")

    png = out_dir / "wj150_grpo_vs_ours_iclr.png"
    pdf = out_dir / "wj150_grpo_vs_ours_iclr.pdf"
    fig.savefig(png, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return png, pdf


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "outputs" / "plot"
    png, pdf = plot(out)
    print(f"saved: {png}")
    print(f"saved: {pdf}")
