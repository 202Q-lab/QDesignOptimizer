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


def _set_publication_style():
    """Mirror typography and plotting style used in plot_convergence.py."""
    plt.rcParams["axes.formatter.useoffset"] = False
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 35,
            "mathtext.fontset": "stix",
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

    fig, ax = plt.subplots(1, 1, figsize=(3.5 * 3.35, 3.5 * 3.35))
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.15, top=0.97)

    # Seed-wise performance map
    ax.scatter(
        seed_idx[stats["is_good"]],
        stats["values"][stats["is_good"]],
        s=80,
        marker="o",
        alpha=0.85,
        label=f"Good ($\\geq${GOOD_CONVERGENCE_THRESHOLD})",
    )
    ax.scatter(
        seed_idx[stats["is_partial"]],
        stats["values"][stats["is_partial"]],
        s=95,
        marker="^",
        alpha=0.90,
        label="Partial",
    )
    ax.scatter(
        seed_idx[stats["is_diverged"]],
        np.full(np.sum(stats["is_diverged"]), 2000),
        s=95,
        marker="x",
        alpha=0.90,
        label="Diverged",
    )
    ax.scatter(
        seed_idx[stats["is_stuck"]],
        np.full(np.sum(stats["is_stuck"]), 2000),
        s=95,
        marker="s",
        alpha=0.90,
        label="Stuck",
    )
    ax.axhline(N_PARAMS, linestyle="--", linewidth=2.5, alpha=0.7)
    ax.axhline(GOOD_CONVERGENCE_THRESHOLD, linestyle=":", linewidth=2.5, alpha=0.7)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Converged parameters")
    ax.set_ylim(2000, 3010)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", fontsize=20, frameon=False)

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
    make_publication_plot(convergence, show_plot=True)
