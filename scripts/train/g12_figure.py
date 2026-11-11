"""Plot the paired fresh-device instruction effects and absolute physical forecast error."""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main(folder):
    primary = json.loads((folder / "metrics/comparisons.json").read_bytes())
    early = json.loads((folder / "metrics/early.json").read_bytes())
    labels = ("Dense GPT-2", "Gated recurrent", "Oscillatory SLWM", "Associative SLWM")
    cores = ("gpt2", "gated", "oscillator", "associative")
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 6.8), layout="constrained")
    fig.suptitle(
        "Spoken instructions and visible device changes on fresh recordings",
        fontsize=16,
        fontweight="bold",
    )
    for col, (f, title) in enumerate(
        (("A", "Coffee machines and computers"), ("B", "Printers and cameras"))
    ):
        ax, phys = axes[0, col], axes[1, col]
        ax.set_title(title, fontsize=13, pad=8)
        ax.axvline(0, lw=1, color="#5a6770")
        for i, core in enumerate(cores):
            for offset, seed in zip((-0.19, 0, 0.19), (1201, 1202, 1203)):
                d = primary["runs"][f"{f}-{core}-paired-{seed}"]["by_role"]["confirm"]
                effect = d["speech"]["wrong"]["ordered"]
                y = 3 - i + offset
                ax.plot(
                    [100 * effect["lo"], 100 * effect["hi"]],
                    [y, y],
                    color="#3c6680",
                    lw=1.35,
                )
                ax.scatter(100 * effect["value"], y, s=24, color="#185574", zorder=4)
                phys.scatter(
                    d["input"]["correct"]["brier"]["value"],
                    y,
                    s=30,
                    facecolors="none",
                    edgecolors="#94381f",
                    linewidth=1.5,
                )
                later = early["runs"][f"{f}-{core}-dropout-{seed}"]["by_role"][
                    "confirm"
                ]["absolute"]["brier"]["value"]
                phys.scatter(later, y, s=23, color="#256e65", zorder=4)
        ax.set_xlabel(
            "Correct minus wrong speech, ordered accuracy (percentage points)",
            fontsize=9,
        )
        ax.set_xlim(-11, 17)
        ax.set_xticks([-10, -5, 0, 5, 10, 15])
        prior = primary["baselines"][f]["confirm"]["state_prior"]["brier"]["value"]
        phys.axvline(prior, color="#303640", ls="--", lw=1.2)
        phys.set_xlabel("Visible-change Brier score (lower is better)", fontsize=9)
        phys.set_xlim(0.13, 0.62)
        for a in (ax, phys):
            a.set_yticks([3, 2, 1, 0], labels if col == 0 else [])
            a.set_ylim(-0.5, 3.5)
            a.grid(axis="x", color="#dbe1e0", linewidth=0.6)
            a.spines[["top", "right", "left"]].set_visible(False)
            a.tick_params(axis="y", length=0, labelsize=10)
            a.set_axisbelow(True)
    axes[1, 0].legend(
        handles=[
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                mfc="none",
                mec="#94381f",
                label="Original schedule",
            ),
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                color="#256e65",
                label="Early joint + dropout",
            ),
            Line2D([], [], ls="--", color="#303640", label="Training prevalence"),
        ],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(1.06, -0.42),
        ncol=3,
        fontsize=9,
    )
    fig.savefig(folder / "g12-comparisons.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    main(p.parse_args().folder)
