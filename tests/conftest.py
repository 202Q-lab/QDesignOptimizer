"""Shared test configuration."""

import matplotlib

# Never open plot windows during tests (plot_progress calls plt.show()).
matplotlib.use("Agg")
