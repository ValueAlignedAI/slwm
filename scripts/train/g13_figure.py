"""Independent cohorts and strong controls for the G13 action result."""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main(folder):
    m = folder / "metrics"
    p = json.loads((m / "prospective.json").read_bytes())
    d = json.loads((m / "direct-action.json").read_bytes())
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    fig.suptitle(
        "Forecasting the sum of four future robot commands",
        fontsize=16,
        fontweight="bold",
    )
    colors = {"gpt2": "#67788c", "transport": "#187669"}
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.set_ylabel("Mean endpoint error (mm; lower is better)")
    left, right = axes
    seeds = (1301, 1302, 1303, 1311, 1312)
    for i, seed in enumerate(seeds):
        v = p["seed_results"][str(seed)]
        line = [v[k]["mm"]["value"] for k in ("gpt2", "transport")]
        left.plot([i - 0.13, i + 0.13], line, color="#c9cdd1", lw=1.5)
        for dx, k in ((-0.13, "gpt2"), (0.13, "transport")):
            left.scatter(i + dx, v[k]["mm"]["value"], c=colors[k], s=45, zorder=3)
    left.axhline(p["controls"]["raw"]["value"], color="#a25725", ls="--", lw=1.4)
    left.set_xticks(range(5), seeds)
    left.set_xlabel("Fixed training seed")
    left.set_ylim(109, 131)
    left.set_title(
        "Prospective processor test\n3,368 trajectories · 196 parent groups",
        fontsize=11,
    )
    for j, a in enumerate(("native", "field", "protected")):
        for offset, c in ((-0.12, "gpt2"), (0.12, "transport")):
            vals = []
            for i, seed in enumerate(seeds):
                r = d["runs"][f"{c}-linear-{seed}"]
                vals.append((r[a] if a == "native" else r["arms"][a]["mm"])["value"])
            right.scatter(
                j + offset + np.linspace(-0.035, 0.035, 5),
                vals,
                color=colors[c],
                s=18,
                alpha=0.45,
            )
            right.plot(
                [j + offset - 0.07, j + offset + 0.07],
                [np.mean(vals)] * 2,
                color=colors[c],
                lw=3,
            )
    right.axhline(d["raw"]["value"], color="#a25725", ls="--", lw=1.4)
    right.set_xticks(
        range(3), ["Native head", "Field-only\nreadout", "Prior + field\nreadout"]
    )
    right.set_title(
        "Independent readout test\n3,535 trajectories · 210 parent groups", fontsize=11
    )
    right.set_ylim(89, 117)
    fig.legend(
        handles=[
            Line2D(
                [], [], marker="o", ls="", color=colors["gpt2"], label="Dense GPT-2"
            ),
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                color=colors["transport"],
                label="Spatial SLWM",
            ),
            Line2D([], [], ls="--", color="#a25725", label="Raw-signal regression"),
        ],
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    fig.savefig(folder / "action-replications.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    stream = np.load(m / "rollout-traces.npz")
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.75), layout="constrained")
    for i, ax in enumerate(axes, 1):
        tag = f"robot-stream-{i}_transport-linear"
        for name, color, style in (
            ("observed", "#26333f", "-"),
            ("free", "#b76831", "--"),
            ("refresh", "#1c8076", "-"),
        ):
            a = stream[tag + "_" + name]
            plot_label = {
                "observed": "Measured",
                "free": "Free rollout",
                "refresh": "State refresh every 8 clocks",
            }[name]
            ax.plot(
                a[:, 0] * 1000,
                a[:, 1] * 1000,
                style,
                color=color,
                lw=1.15,
                label=plot_label,
                alpha=0.85,
            )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"Recording {i}")
        ax.grid(alpha=0.16)
        ax.set_xlabel("Robot x (mm)")
        if i == 1:
            ax.set_ylabel("Robot y (mm)")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Measured motion and accumulated forecast drift", fontsize=14, fontweight="bold"
    )
    fig.savefig(folder / "real-robot-replays.png", dpi=155, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    main(p.parse_args().folder)
