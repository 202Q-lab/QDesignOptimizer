from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import root

from qdesignoptimizer.utils.names_parameters import mode, param


def sample_uniform_from_union(
    intervals: List[Tuple[float, float]], size, rng: np.random.Generator
):
    """
    Sample uniformly from a union of disjoint closed intervals.

    This function generates random samples that are uniformly distributed across
    the combined space of multiple disjoint intervals. Each interval contributes
    samples proportional to its length, ensuring overall uniform distribution.

    Parameters
    ----------
    intervals : List[Tuple[float, float]]
        List of disjoint closed intervals as (low, high) tuples.
        Example: [(-2.5, -0.6), (0.6, 2.5)] represents [-2.5, -0.6] ∪ [0.6, 2.5]
    size : int or tuple of ints
        Output shape. If int, return 1-D array of length size.
        If tuple, return array with that shape.
    rng : np.random.Generator
        NumPy random number generator instance for reproducible sampling.

    Returns
    -------
    np.ndarray
        Array of samples uniformly distributed over the union of intervals.
        Shape matches the `size` parameter.

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> # Sample from [-2, -1] ∪ [1, 2]
    >>> samples = sample_uniform_from_union([(-2, -1), (1, 2)], size=1000, rng=rng)
    >>> # Approximately 50% of samples will be in each interval

    Notes
    -----
    This creates a uniform distribution over the union by:
    1. Computing interval weights proportional to their lengths
    2. Randomly selecting which interval to sample from
    3. Uniformly sampling within the selected interval

    The intervals should be disjoint (non-overlapping) for correct uniform behavior.
    """
    # Extract interval boundaries and compute lengths
    lows = np.array([lo for lo, hi in intervals], dtype=float)
    highs = np.array([hi for lo, hi in intervals], dtype=float)
    interval_lengths = highs - lows

    # Probability of selecting each interval (proportional to length for uniform distribution)
    interval_probs = interval_lengths / interval_lengths.sum()

    # Step 1: Choose which interval to sample from (weighted by length)
    selected_intervals = rng.choice(len(intervals), size=size, p=interval_probs)

    # Step 2: Sample uniformly within each selected interval
    uniform_samples = rng.random(size)  # uniform in [0, 1)

    # Step 3: Scale and shift to the selected intervals
    return (
        lows[selected_intervals]
        + uniform_samples * interval_lengths[selected_intervals]
    )


def exponent_approx_to_1_over_round(
    x: np.ndarray, exponent_approx_to_1_over: int
) -> np.ndarray:
    """Rounding map R(x) = round(2x)/2 elementwise."""
    return np.round(exponent_approx_to_1_over * x) / exponent_approx_to_1_over


@dataclass
class ScaledSystem:
    """
    Scaled-up problem definition.

    Indexing:
      - i in [0..n_clusters-1], j in [0..m_per_cluster-1]
      - a = (i, j) is a composite parameter index.
      - For each a we pick three weak-coupling targets b_a, c_a, d_a in other clusters.
        Their exponents are gamma[a, t] for t=0,1,2 respectively.

    Shapes:
      - alpha, beta: (n, m, m)  with zero diagonal (no self term in products)
      - gamma:       (n, m, 3)  exponents for the three cross-cluster couplings
      - b_i, b_j:    (n, m, 3)  integer indices for the three composite targets
      - x0, y_target_values:(n, m)
    """

    n_clusters: int
    m_per_cluster: int
    epsilon: float
    exponent_approx_to_1_over: int
    seed: int
    sample_range_alpha_ij_eq_k: list
    sample_range_alpha_ij_neq_k: list
    sample_range_beta: list
    sample_range_gamma: list

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        n, m = self.n_clusters, self.m_per_cluster

        self.alpha = np.zeros((n, m, m))
        for i in range(n):
            for j in range(m):
                for k in range(m):

                    if j == k:
                        alpha = sample_uniform_from_union(
                            self.sample_range_alpha_ij_eq_k, size=1, rng=rng
                        )
                    else:
                        alpha = sample_uniform_from_union(
                            self.sample_range_alpha_ij_neq_k, size=1, rng=rng
                        )
                    self.alpha[i, j, k] = alpha
                    # if k != j + 1:
                    #     self.alpha[i, j, k] = temp

        # Zero super-diagonal (j, j+1) for alpha
        # idx = np.arange(m-1)  # indices 0 to m-2
        # self.alpha[:, idx, idx+1] = 0.0  # alpha[i, j, j+1] = 0 for j < m-1
        self.alpha_approx = exponent_approx_to_1_over_round(
            self.alpha, self.exponent_approx_to_1_over
        )

        # U_beta  = [-0.6, -0.3] ∪ [ 0.3,  0.6]
        self.beta = sample_uniform_from_union(
            self.sample_range_beta, size=(n, m, m), rng=rng
        )
        # Zero the self-coupling diagonal so that ∏_k includes a harmless factor 1 at k=j
        idx = np.arange(m)
        self.beta[:, idx, idx] = 0.0  # diagonal: beta[i, j, j] = 0
        self.beta_approx = exponent_approx_to_1_over_round(
            self.beta, self.exponent_approx_to_1_over
        )

        self.gamma = {}
        for i in range(n):
            for j in range(m):
                if (i, j) not in self.gamma:
                    self.gamma[(i, j)] = {}

                # Pick three distinct (k,l) from other clusters
                params = []
                while len(params) < 3:
                    k = rng.integers(0, n)
                    l = rng.integers(0, m)
                    if k != i and (k, l) not in self.gamma[(i, j)]:
                        params.append((k, l))
                        if (i, j) not in self.gamma:
                            self.gamma[(i, j)] = {}
                        self.gamma[(i, j)][(k, l)] = sample_uniform_from_union(
                            self.sample_range_gamma, size=1, rng=rng
                        )

        # --- Initial design variables and target parameters ---
        # U_i = [0.5, 1.0]§
        self._x = rng.uniform(0.5, 1.0, size=(n, m))
        self._y = rng.uniform(0.5, 1.0, size=(n, m))
        self.y_target_value = rng.uniform(0.5, 1.0, size=(n, m))
        self.flattened_y_target = self.create_flattened_y_target()

    def flatten(self, vec, prefix, suffix) -> np.ndarray:
        """Return y as a flattened array of shape (n_clusters * m_per_cluster,)"""
        return {
            f"{prefix}{i},{j}{suffix}": vec[i, j]
            for i in range(self.n_clusters)
            for j in range(self.m_per_cluster)
        }

    def get_flattened_y(self) -> np.ndarray:
        return self.flatten(self._y, "", "_")

    def create_flattened_y_target(self) -> np.ndarray:
        return self.flatten(self._y, "", "_")

    def get_flattened_y_target(self) -> np.ndarray:
        return self.flattened_y_target

    def get_flattened_x(self) -> np.ndarray:
        return self.flatten(self._x, "dv_", "")

    def set_updated_design_vars(self, updated_design_vars: Dict[str, float]):
        """
        Update self._x from a flattened dictionary with keys f"dv_{i},{j}".

        Parameters
        ----------
        updated_design_vars : Dict[str, float]
            Dictionary with keys like "dv_0_0", "dv_0_1", etc. and float values
        """
        for i in range(self.n_clusters):
            for j in range(self.m_per_cluster):
                key = f"dv_{i},{j}"
                if key in updated_design_vars:
                    self._x[i, j] = updated_design_vars[key]
                else:
                    print(f"Warning: Key {key} not found in updated_design_vars")

    def _g_ij(self, i, j, y_values=None) -> float:
        """Calculate g(i,j) = ∏_k x_{i,k}^{alpha_{i,j,k}} * ∏_k y_{i,k}^{beta_{i,j,k}}"""
        if y_values is None:
            y_values = self._y
        return np.prod(np.power(self._x[i, :], self.alpha[i, j, :])) * np.prod(
            np.power(y_values[i, :], self.beta[i, j, :])
        )

    def _h_ij(self, i, j, y_values) -> callable:
        """Calculate h(a) = 1 + epsilon * ∏_{t=0..2} y_{b_t(a)}^{gamma_{b_t(a)}}"""
        y_from_other_clusters = [y_values[k, l] for (k, l) in self.gamma[(i, j)].keys()]
        gammas = [self.gamma[(i, j)][(k, l)] for (k, l) in self.gamma[(i, j)].keys()]
        term = np.prod(
            [y_t**gamma for y_t, gamma in zip(y_from_other_clusters, gammas)]
        )
        return 1.0 + self.epsilon * term

    def _solve_yij_equal_gij(self, tol=1e-12):
        """
        Solve for y values using a nonlinear solver for the strongly coupled system within each cluster.
        For each cluster i, solve the system: y[i,j] = g_ij(i,j) for all j in that cluster.

        Parameters
        ----------
        tol : float
            Tolerance for convergence
        """

        for cluster_i in range(self.n_clusters):
            # Define the residual function for this cluster: F(y_i) = y_i - g_i(y_i) = 0
            def cluster_residual(y_cluster):
                # Temporarily update this cluster's values in self._y
                old_cluster = self._y[cluster_i, :].copy()
                self._y[cluster_i, :] = y_cluster

                residuals = np.zeros(self.m_per_cluster)
                for param_j in range(self.m_per_cluster):
                    # Calculate g_ij using current self._y
                    g_val = self._g_ij(cluster_i, param_j)
                    residuals[param_j] = (y_cluster[param_j] - g_val) ** 2

                # Restore original cluster values
                self._y[cluster_i, :] = old_cluster
                return residuals

            # Initial guess for this cluster
            y0_cluster = self._y[cluster_i, :].copy()

            # Solve the nonlinear system using modified Powell hybrid method
            result = root(cluster_residual, y0_cluster, method="hybr", tol=tol)

            if not result.success:
                print(
                    f"Warning: Solver failed for cluster {cluster_i}. Message: {result.message}"
                )

            # Store the solution for this cluster
            self._y[cluster_i, :] = result.x

    def _solve_yij_equal_gij_hij_perturbatively(self, tol=1e-12):
        residual = tol + 1.0
        y_pert = self._y.copy()
        while residual > tol:
            for i, j in np.ndindex(self.n_clusters, self.m_per_cluster):
                y_pert[i, j] = self._g_ij(i, j, y_pert) * self._h_ij(i, j, y_pert)
            residual = np.sum((self._y - y_pert) ** 2)
            self._y = y_pert.copy()

    def gather_info_for_y_given_x(self, tol=1e-12):
        self._solve_yij_equal_gij(tol=tol)  # as initial guess for perturbative solution
        self._solve_yij_equal_gij_hij_perturbatively(tol=tol)


def get_prop_to(i: int, j: int, sys: ScaledSystem) -> callable:
    """Return a function p,v -> prop_to(i,j) = ∏_k v_{i,k}^{alpha_{i,j,k}} * ∏_k p_{i,k}^{beta_{i,j,k}}"""
    func = lambda p, v: np.prod(
        [
            np.power(v[f"dv_{i},{k}"], sys.alpha_approx[i, j, k])
            for k in range(sys.m_per_cluster)
        ]
    ) * np.prod(
        [
            np.power(p[param(mode(f"{i},{k}"), "")], sys.beta_approx[i, j, k])
            for k in range(sys.m_per_cluster)
        ]
    )
    return func
