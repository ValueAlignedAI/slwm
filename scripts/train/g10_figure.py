"""Render the context, transient-command, grounding and outcome comparisons from released measurements."""

import json
import sys
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"hybrid": "#126658", "gpt2": "#a2522a", "gated": "#526a92"}
LABELS = {"hybrid": "SLWM", "gpt2": "Dense GPT-2", "gated": "Gated recurrent"}
SEEDS = (1001, 1002, 1003)


def draw(src, out):
    load = lambda f: json.loads((src / f).read_text())
    ctx = load("context.json")
    inst = load("instructions.json")
    real = load("grounding.json")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True)

    def marks(ax, x, series, core, dx):
        y = np.asarray(series) * 100
        ax.plot([x + dx, x + dx], [min(y), max(y)], lw=1.4, color=COLORS[core])
        for j, m in enumerate(("o", "s", "^")):
            ax.scatter(
                [x + dx],
                [y[j]],
                marker=m,
                facecolors="white" if j < 2 else COLORS[core],
                edgecolors=COLORS[core],
                s=30,
                linewidths=1.5,
                label=LABELS[core] if x == 0 and j == 0 else None,
            )

    for c, dx in zip(LABELS, (-0.16, 0, 0.16)):
        for i, mode in enumerate(("short", "variable", "long")):
            marks(
                axes[0, 0],
                i,
                [
                    ctx["rows"][f"context-{c}-{mode}-s{s}/5120"]["full"]["occupancy"]
                    for s in SEEDS
                ],
                c,
                dx,
            )
        for i, mode in enumerate(("continuous", "brief", "latched")):
            marks(
                axes[0, 1],
                i,
                [
                    inst["latch"][f"context-{c}-variable-s{s}"]["occupancy"][mode]
                    for s in SEEDS
                ],
                c,
                dx,
            )
    axes[0, 0].set_xticks(range(3), ["32 steps", "Mixed 32–512", "512 steps"])
    axes[0, 0].set_title("Training context · uninterrupted missions", loc="left")
    axes[0, 0].legend(loc="lower left", fontsize=8, frameon=False)
    axes[0, 0].set_ylabel("Time at goal (%)")
    axes[0, 1].set_xticks(range(3), ["Continuous", "Brief", "Brief + cache"])
    axes[0, 1].set_title("Instruction delivery · independent missions", loc="left")
    axes[0, 1].set_ylabel("Time at goal (%)")
    for ax in axes[0]:
        ax.set_ylim(0, 102)
    for c, dx in (("hybrid", -0.08), ("gpt2", 0.08)):
        for i, (role, kind) in enumerate(
            (
                ("familiar", "reconstruct"),
                ("familiar", "paired"),
                ("novel", "reconstruct"),
                ("novel", "paired"),
            )
        ):
            marks(
                axes[1, 0],
                i,
                [
                    real["rows"][f"align-{c}-{kind}-s{s}"]["panels"][role]["retrieval"][
                        "audio_to_vision"
                    ]["direct"]
                    for s in SEEDS
                ],
                c,
                dx,
            )
        for i, role in enumerate(("familiar", "novel")):
            marks(
                axes[1, 1],
                i,
                [
                    real["rows"][f"align-{c}-paired-s{s}"]["risk"][role]["metrics"][
                        "all"
                    ]["ap"]["mean"]
                    for s in SEEDS
                ],
                c,
                dx,
            )
    axes[1, 0].axhline(12.5, lw=1, ls=":", c="#444")
    axes[1, 0].set_xticks(
        range(4),
        [
            "Familiar\nreconstruct",
            "Familiar\npaired",
            "New tasks\nreconstruct",
            "New tasks\npaired",
        ],
    )
    axes[1, 0].set_title("Audio → video · eight times in one recording", loc="left")
    axes[1, 0].set_ylabel("Top-1 match (%)")
    axes[1, 0].set_ylim(bottom=0)
    for i, role in enumerate(("familiar", "novel")):
        ref = real["rows"]["align-hybrid-paired-s1001"]["risk"][role]
        for key, style, color in (("raw", "--", "#555"), ("prevalence", ":", "#333")):
            val = ref["metrics"][key]["ap"]["mean"] * 100
            axes[1, 1].plot(
                [i - 0.34, i + 0.34],
                [val, val],
                ls=style,
                c=color,
                lw=1.3,
                label=key.title() if i == 0 else None,
            )
    axes[1, 1].set_xticks(range(2), ["Familiar tasks", "New task families"])
    axes[1, 1].set_title("Next human action error · frozen field", loc="left")
    axes[1, 1].set_ylabel("Average precision (%)")
    axes[1, 1].set_ylim(bottom=0)
    axes[1, 1].legend(loc="lower center", fontsize=8, frameon=False)
    for ax in axes.ravel():
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.13)
    fig.suptitle(
        "G10 · what transfers across time, instructions and tasks?",
        x=0.01,
        ha="left",
        fontsize=15,
    )
    fig.savefig(out, dpi=155, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw(Path(sys.argv[1]), Path(sys.argv[2]))
