from math import nan
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Number of converged parameters (out of 3000) for seeds 0..99.
# nan: stuck in evaluation, 0: diverged.
convergence = [
    2993,
    3000,
    3000,
    2992,
    0,
    0,
    3000,
    3000,
    3000,
    2997,
    0,
    0,
    2997,
    3000,
    3000,
    0,
    2878,
    2080,
    3000,
    2992,
    2995,
    2998,
    0,
    3000,
    3000,
    2965,
    3000,
    2981,
    2996,
    2990,
    3000,
    3000,
    2995,
    3000,
    2996,
    2997,
    2996,
    2995,
    2998,
    nan,
    2991,
    2997,
    2997,
    3000,
    2996,
    nan,
    3000,
    2992,
    2997,
    2997,
    3000,
    2982,
    2997,
    2996,
    2997,
    2996,
    2974,
    2996,
    0,
    0,
    2996,
    2997,
    2952,
    2997,
    3000,
    3000,
    2986,
    2997,
    2997,
    2997,
    2996,
    2997,
    2998,
    3000,
    2997,
    2974,
    2997,
    2989,
    2998,
    2995,
    2997,
    nan,
    3000,
    2991,
    2995,
    3000,
    2997,
    2997,
    0,
    3000,
    2995,
    0,
    2994,
    2995,
    3000,
    2999,
    2998,
    0,
    3000,
    2999,
]

N_PARAMS = 3000
GOOD_CONVERGENCE_THRESHOLD = 2990

COLORS = {
    "good": "#009639",
    "partial": "#F58220",
    "diverged": "#E30613",
    "stuck": "#93328E",
    "line_thr": "#8C8C8C",
}


def _set_publication_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 10,
            "mathtext.fontset": "stix",
            "axes.formatter.useoffset": False,
        }
    )


def _classify_runs(values):
    values = np.asarray(values, dtype=float)
    is_stuck = np.isnan(values)
    is_diverged = values == 0
    is_partial = (values > 0) & (values < GOOD_CONVERGENCE_THRESHOLD)
    is_good = values >= GOOD_CONVERGENCE_THRESHOLD
    is_full = values == N_PARAMS

    return {
        "values": values,
        "is_stuck": is_stuck,
        "is_diverged": is_diverged,
        "is_partial": is_partial,
        "is_good": is_good,
        "is_full": is_full,
        "valid_positive": values[(~is_stuck) & (values > 0)],
    }


def make_publication_plot(values, save_dir=None, show_plot=True):
    _set_publication_style()
    stats = _classify_runs(values)
    seed_idx = np.arange(len(values))

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(3.35, 1.9),
        gridspec_kw={"height_ratios": [6, 1], "hspace": 0},
    )
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.20, top=0.97)

    # --- Top panel: converged seeds ---
    sc_good = ax_top.scatter(
        seed_idx[stats["is_good"]],
        stats["values"][stats["is_good"]],
        s=12,
        marker="o",
        color=COLORS["good"],
        alpha=0.85,
        label=f"Good ($N_p \\geq {GOOD_CONVERGENCE_THRESHOLD}$)",
        zorder=3,
    )
    sc_partial = ax_top.scatter(
        seed_idx[stats["is_partial"]],
        stats["values"][stats["is_partial"]],
        s=14,
        marker="^",
        color=COLORS["partial"],
        alpha=0.90,
        label=f"Partial ($0 < N_p < {GOOD_CONVERGENCE_THRESHOLD}$)",
        zorder=3,
    )
    ax_top.set_ylabel("Converged parameters")
    ax_top.set_ylim(1975, 3050)
    ax_top.set_xlim(-1, len(values))
    ax_top.tick_params(bottom=False, labelbottom=False)

    # --- Bottom panel: diverged and stuck ---
    sc_div = ax_bot.scatter(
        seed_idx[stats["is_diverged"]],
        np.full(np.sum(stats["is_diverged"]), 0.5),
        s=14,
        marker="x",
        color=COLORS["diverged"],
        alpha=0.90,
        label="Diverged",
        zorder=3,
        linewidths=1.0,
    )
    sc_stuck = ax_bot.scatter(
        seed_idx[stats["is_stuck"]],
        np.full(np.sum(stats["is_stuck"]), 0.5),
        s=14,
        marker="s",
        color=COLORS["stuck"],
        alpha=0.90,
        label="Stuck",
        zorder=3,
    )
    ax_bot.set_ylim(0, 1)
    ax_bot.set_xlim(-1, len(values))
    ax_bot.set_yticks([0.5])
    ax_bot.set_yticklabels(["Div./\nStuck"], fontsize=6.5)
    ax_bot.tick_params(axis="y", length=0)
    ax_bot.set_xlabel("Seed")

    # --- Legend (top panel, all handles) ---
    handles = [sc_good, sc_partial, sc_div, sc_stuck]
    ax_top.legend(
        handles=handles,
        loc="lower right",
        fontsize=7,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.4,
        labelspacing=0.3,
    )
    # fig.tight_layout()

    if save_dir is None:
        save_dir = Path(__file__).resolve().parent / "out"
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    png_path = save_dir / "FigS7_convergence_100_seeds.png"
    fig.savefig(png_path, dpi=600)

    print(f"Saved: {png_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    make_publication_plot(convergence, show_plot=False)
