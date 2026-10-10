"""
Learning curves from the deterministic evaluations logged during training
(runs/<variant>/seed<n>/eval_log.csv). Lines are the mean over seeds; the
shaded band spans the lowest to highest seed.

Examples:
  python plot_results.py                                       ->  learning_curves.png
  python plot_results.py --variants td3 td3_bonus td3_bonus_multi --out bonus_curves.png
"""
import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

# Fixed colour per agent, the same in every chart (validated categorical palette)
SERIES = {
    "itd3_multi":      ("ITD3 (multi-step)",      "#2a78d6"),
    "itd3":            ("ITD3 (one-step)",        "#eb6834"),
    "td3":             ("TD3",                    "#1baf7a"),
    "ppo":             ("PPO",                    "#eda100"),
    "td3_bonus":       ("TD3 + one-step bonus",   "#008300"),
    "td3_bonus_multi": ("TD3 + multi-step bonus", "#4a3aa7"),
}
MAIN = ["itd3_multi", "itd3", "td3", "ppo"]
SURFACE, INK, INK_SECONDARY, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"


def load_eval_logs(name):
    """Returns (steps, values) where values has shape (seeds, evaluations, 2): goal % and mean return."""
    logs = []
    for path in sorted(glob.glob(os.path.join("runs", name, "seed*", "eval_log.csv"))):
        with open(path, newline="") as f:
            rows = [(int(r["total_steps"]), 100 * float(r["goal"]), float(r["mean_return"]))
                    for r in csv.DictReader(f)]
        if rows:
            logs.append(rows)
    if not logs:
        return None
    n = min(len(rows) for rows in logs)            # unfinished runs may be shorter
    data = np.array([rows[:n] for rows in logs])   # (seeds, n, 3)
    return data[0, :, 0], data[:, :, 1:]


def spread_labels(ends, min_gap, top):
    """Nudge label heights apart so that no two are closer than min_gap, without going above top."""
    order = sorted(range(len(ends)), key=lambda i: ends[i])
    placed = list(ends)
    for prev, cur in zip(order, order[1:]):
        placed[cur] = max(placed[cur], placed[prev] + min_gap)
    overflow = max(0.0, placed[order[-1]] - top)
    return [y - overflow for y in placed]


def plot_learning_curves(variants, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE)
    panels = [("Goal reached (% of evaluation episodes)", (0, 100)), ("Mean evaluation return", None)]
    ends = [[], []]                                 # (final value, label) per panel

    for name in variants:
        label, color = SERIES[name]
        loaded = load_eval_logs(name)
        if loaded is None:
            continue
        steps, values = loaded
        seeds = f"{len(values)} seed{'s' if len(values) > 1 else ''}"
        for i, ax in enumerate(axes):
            mean = values[:, :, i].mean(0)
            ax.fill_between(steps, values[:, :, i].min(0), values[:, :, i].max(0),
                            color=color, alpha=0.10, linewidth=0)
            ax.plot(steps, mean, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
                    label=f"{label} ({seeds})")
            ax.plot(steps[-1], mean[-1], "o", color=color, markersize=8)
            ends[i].append((mean[-1], label, steps[-1]))

    for ax, (ylabel, ylim), panel_ends in zip(axes, panels, ends):
        ax.set_facecolor(SURFACE)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlabel("Timesteps", color=INK_SECONDARY)
        ax.set_ylabel(ylabel, color=INK_SECONDARY)
        ax.tick_params(colors=INK_MUTED)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)

        # Direct labels at the line ends, in text ink, nudged apart where lines converge
        if panel_ends:
            (low, high), (left, right) = ax.get_ylim(), ax.get_xlim()
            heights = spread_labels([value for value, _, _ in panel_ends], 0.08 * (high - low), high)
            for (_, label, x), y in zip(panel_ends, heights):
                ax.text(x + 0.02 * (right - left), y, label, color=INK_SECONDARY, fontsize=9,
                        va="center", ha="left", clip_on=False)

    # One legend row above both panels, clear of the lines and their end labels
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.93),
               ncol=len(variants), frameon=False, labelcolor=INK)
    fig.suptitle("Deterministic evaluation during training", color=INK)
    fig.tight_layout(rect=(0, 0, 0.96, 0.87))
    fig.subplots_adjust(wspace=0.4)

    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", choices=SERIES, default=MAIN)
    parser.add_argument("--out", default="learning_curves.png")
    args = parser.parse_args()
    plot_learning_curves(args.variants, args.out)
