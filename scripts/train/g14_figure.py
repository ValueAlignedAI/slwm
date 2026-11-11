"""Render the registered, paired physical controls and their uncertainty as a scientific figure."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/mnt/slwm-data/SLWM/g14")


def main():
    file = ROOT / "release/metrics/physical-results.json"
    d = json.loads(file.read_bytes())
    groups = d["cohorts"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#233242",
            "text.color": "#233242",
        }
    )
    fig, axs = plt.subplots(
        1, 3, figsize=(13.1, 4.9), gridspec_kw={"width_ratios": [1.22, 1, 1]}
    )
    colors = {"held": "#286496", "fresh": "#bd591e"}
    lab = {"held": "8 held speakers", "fresh": "2 new speakers"}
    categories = [
        ("geometry_original_cue", "Geometry, original audio"),
        ("geometry", "Geometry, pretrained audio"),
        ("contact_speed", "Fixed fast contact push"),
        ("raw", "Raw predictor"),
        ("gpt2", "Dense GPT-2 style"),
        ("transport", "Spatial SLWM"),
    ]

    def dot(ax, pos, v, color, scale=100):
        ax.errorbar(
            v["value"] * scale,
            pos,
            xerr=np.maximum(
                0.0, np.array([[v["value"] - v["lo"]], [v["hi"] - v["value"]]])
            )
            * scale,
            fmt="o",
            markersize=4.5,
            color=color,
            elinewidth=1.5,
            capsize=2.4,
        )

    for role in ("held", "fresh"):
        c = groups[role]
        off = -0.12 if role == "held" else 0.12
        for i, (key, _) in enumerate(categories):
            dot(axs[0], i + off, c["by_policy_family"][key]["success"], colors[role])
        for ix, (k, name) in enumerate(
            (
                ("1411", "Seed 1411"),
                ("1412", "Seed 1412"),
                ("1413", "Seed 1413"),
                ("mean", "Mean of 3 seeds"),
            )
        ):
            v = (
                c["paired"]["transport_over_gpt2"]["success"]
                if k == "mean"
                else c["spatial_over_dense_by_seed"][k]["success"]
            )
            dot(axs[1], ix + off, v, colors[role])
        for ix, key in enumerate(
            (
                "transport_over_wrong_color",
                "transport_over_wrong_address",
                "retained_over_reset",
                "transport_full_over_mean_field",
            )
        ):
            dot(axs[2], ix + off, c["paired"][key]["success"], colors[role])
    for a in axs:
        a.grid(axis="x", alpha=0.18)
        a.set_axisbelow(True)
        a.invert_yaxis()
        a.tick_params(axis="y", length=0)
    axs[0].set_yticks(range(len(categories)), [k[1] for k in categories])
    axs[0].set_xlabel("Selective placements (%)")
    axs[0].set_xlim(0, 100)
    axs[1].set_yticks(
        range(4), ["Seed 1411", "Seed 1412", "Seed 1413", "Mean of 3 seeds"]
    )
    axs[1].axvline(0, color="#767d80", lw=0.9)
    axs[1].set_xlabel("SLWM minus dense (percentage points)")
    axs[2].set_yticks(
        range(4),
        [
            "Correct vs. wrong object",
            "Correct vs. wrong address",
            "Retained vs. reset goal",
            "Full vs. fixed learned field",
        ],
    )
    axs[2].axvline(0, color="#767d80", lw=0.9)
    axs[2].set_xlabel("Paired gain (percentage points)")
    for ax, name in zip(
        axs, ("Executed goals", "Matched processors", "Causal interventions")
    ):
        ax.set_title(name, loc="left", fontweight="bold", fontsize=12)
    for r in colors:
        axs[0].plot([], [], "o", color=colors[r], label=lab[r])
    fig.legend(
        *axs[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False
    )
    fig.subplots_adjust(left=0.15, right=0.987, top=0.90, bottom=0.21, wspace=0.91)
    fig.savefig(ROOT / "release/physical-comparison.png", dpi=175, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
