"""Plot motion forecasts, the persistence control and the context-field intervention."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/mnt/slwm-data/SLWM/g14/release")


def main():
    d = json.loads((ROOT / "metrics/held-prediction.json").read_bytes())
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(12, 3.55), gridspec_kw={"width_ratios": [1, 1.2, 1]}
    )
    labels = ("Persistence", "Raw response", "Dense", "Spatial SLWM")
    keys = ("persistence-nominal", "raw", "gpt2", "transport")
    colors = ("#687881", "#927046", "#467397", "#187461")

    def show(ax, i, v, col):
        p = v["value"]
        ax.errorbar(
            p,
            i,
            xerr=[[max(0, p - v["lo"])], [max(0, v["hi"] - p)]],
            fmt="o",
            capsize=3,
            color=col,
            ms=6,
            lw=1.7,
        )

    for a, metric, title in zip(
        axes,
        ("all_blocks_mm", "moving_requested_block_mm"),
        ("All four blocks", "Requested block moving ≥5 mm"),
    ):
        for i, (key, col) in enumerate(zip(keys, colors)):
            show(a, i, d["scores"][key][metric], col)
        a.set_yticks(range(4), labels)
        a.set_xlim(left=0)
        a.set_title(title, loc="left", fontsize=11, weight="bold")
        a.set_xlabel("Displacement error (mm)")
    ax = axes[2]
    for i, (key, col) in enumerate(
        zip(("dense_full_over_mean", "spatial_full_over_mean"), colors[2:])
    ):
        show(ax, i, d["comparisons"][key]["moving_requested_block_mm"], col)
    ax.set_yticks((0, 1), ("Dense", "Spatial SLWM"))
    ax.set_xlim(left=0)
    ax.set_title("Contextual field, seed 1411", loc="left", fontsize=11, weight="bold")
    ax.set_xlabel("Error saved over fixed field (mm)")
    for ax in axes:
        ax.invert_yaxis()
        ax.margins(y=0.2)
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    fig.tight_layout(pad=1.4, w_pad=2.5)
    fig.savefig(ROOT / "physical-forecasts.png", dpi=175, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
