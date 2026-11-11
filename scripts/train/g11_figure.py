"""Plot held instruction effects and independently audited physical coverage."""

import json
import sys
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"hybrid": "#08766c", "gpt2": "#aa5429", "gated": "#536c92"}
NAMES = {"hybrid": "SLWM", "gpt2": "Dense GPT-2", "gated": "Gated recurrent"}


def draw(folder, dest):
    load = lambda n: json.loads((folder / n).read_text())
    p = load("primary.json")
    sc = load("scale.json")
    au = load("device-audit.json")
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 7.6), constrained_layout=True)

    def marks(ax, i, vals, c, dx, legend):
        y = 100 * np.asarray(vals)
        ax.plot([i + dx] * 2, [y.min(), y.max()], color=COLORS[c], alpha=0.75, lw=1.4)
        for j, v in enumerate(y):
            ax.scatter(
                i + dx,
                v,
                s=42,
                marker=("o", "s", "^")[j],
                edgecolors=COLORS[c],
                facecolors=COLORS[c] if j == 2 else "white",
                lw=1.7,
                label=NAMES[c] if legend and i == 0 and j == 0 else None,
            )

    for i, f in enumerate(("A", "B")):
        for c, dx in zip(COLORS, (-0.15, 0, 0.15)):
            rows = [
                p["comparisons"][f"g11-{f}-{c}-paired-s{s}"]["held"]["wrong"]
                for s in (1001, 1002, 1003)
            ]
            marks(axes[0, 0], i, [r["verb"]["mean"] for r in rows], c, dx, True)
            marks(axes[0, 1], i, [r["effect_mae"]["mean"] for r in rows], c, dx, False)
            rows = [
                sc["size_comparisons"][f"{f}-{c}-s{s}"]["large_minus_small"]
                for s in (911, 912)
            ]
            marks(axes[1, 0], i, [r["verb"]["mean"] for r in rows], c, dx, False)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.axhline(0, ls=":", lw=1, color="#444")
        ax.set_xlim(-0.45, 1.45)
        ax.set_xticks((0, 1), ("Coffee + computer", "Printer + camera"))
    axes[0, 0].set_title(
        "Actual vs earlier instruction · next verb", loc="left", fontsize=11
    )
    axes[0, 0].set_ylabel("Accuracy gain (percentage points)")
    axes[0, 0].legend(fontsize=8, frameon=False)
    axes[0, 1].set_title(
        "Actual vs earlier instruction · surface forecast", loc="left", fontsize=11
    )
    axes[0, 1].set_ylabel("Reduction in changed-fraction MAE (points)")
    axes[1, 0].set_title(
        "117M minus 13M · equal pretraining exposure", loc="left", fontsize=11
    )
    axes[1, 0].set_ylabel("Next-verb accuracy gain (points)")
    families = ("coffee", "computer", "printer", "camera")
    x = np.arange(4)
    count = lambda fam, code: sum(
        v["family"] == fam and (code is None or v["grade"] == code) for v in au["rows"]
    )
    axes[1, 1].bar(
        x, [count(f, None) for f in families], color="#d4dee2", label="Depth qualified"
    )
    axes[1, 1].bar(
        x,
        [count(f, "D") for f in families],
        color="#34786c",
        label="Clearly device-centered",
    )
    axes[1, 1].set_xticks(x, [f.title() for f in families])
    axes[1, 1].set_title(
        "Physical measurement coverage · blind review", loc="left", fontsize=11
    )
    axes[1, 1].set_ylabel("Held instruction events")
    axes[1, 1].legend(fontsize=8, frameon=False)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "G11 · spoken instructions on new device families",
        x=0.01,
        ha="left",
        fontsize=16,
    )
    fig.savefig(dest, dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw(Path(sys.argv[1]), Path(sys.argv[2]))
