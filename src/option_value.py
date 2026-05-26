"""
Real-options valuation of a hydropower reservoir.

The reservoir is treated as an American option on production. Each day:

    * If we produce, we earn  Q × spot_price  (cash flow today) and the
      reservoir loses Q MWh.
    * If we hold, no cash flow, no draw-down.
    * In *both* cases, daily inflow I MWh is added to the reservoir.
    * The reservoir level is clipped at [0, K]. Inflow that would push it
      above K is "spilled" (lost forever — implicit penalty via the lost
      future cash flow).

The state is two-dimensional: reservoir level S and the log-price residual
X from the Schwartz 1-factor model (see :mod:`src.price_model`). We solve
the Bellman equation by **backward induction on a discrete (S, X) grid**,
with linear interpolation along the S axis when an action puts us between
grid points (so the reservoir spec doesn't have to be grid-aligned).

The interesting output is the **water value**

    V_∂(S, X, t) = ∂V/∂S ≈ (V(S+dS, X, t) − V(S, X, t)) / dS         [EUR/MWh]

i.e. the marginal value of one extra MWh in the reservoir at state (S, X, t).
That's the quantity a portfolio manager actually uses when deciding whether
to bid the next MWh into the day-ahead auction.

Notes on simplifications (v1):
    * No discounting (r = 0). Simple to extend by passing
      ``discount_rate_per_year`` to :class:`ReservoirSpec`.
    * Calibration is under the historical measure (P), not risk-neutral (Q).
      Step would be to recalibrate on forward curves.
    * No ramping or minimum-stop constraints; daily produce/hold is binary.
    * No salvage value at the horizon: V(T, S, X) = 0. Choose a horizon
      that ends in a low-price season (e.g. start = 1 Jan → end = 31 Dec)
      so the bias is small.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from .price_model import SchwartzModel


# -----------------------------------------------------------------------------
# Reservoir / plant specification
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ReservoirSpec:
    """
    Physical specification of the reservoir-plant system.

    Attributes
    ----------
    capacity_mwh : float
        Maximum reservoir content, K, in MWh of stored energy.
    daily_production_mwh : float
        Q — energy produced in one day at full power.
        For a 100 MW plant: Q = 100 × 24 = 2 400 MWh.
    daily_inflow_mwh : float
        I — assumed deterministic daily inflow in MWh of energy equivalent.
    discount_rate_per_year : float, default 0.0
        Continuously-compounded annual discount rate r. We apply
        exp(-r/365) per daily step.
    """

    capacity_mwh: float
    daily_production_mwh: float
    daily_inflow_mwh: float
    discount_rate_per_year: float = 0.0


# -----------------------------------------------------------------------------
# Numerical helpers
# -----------------------------------------------------------------------------

def x_transition_matrix(model: SchwartzModel, x_grid: np.ndarray) -> np.ndarray:
    """
    Discrete transition matrix for the residual X on ``x_grid``.

    ``P[i, j] = P(X_{t+1} ≈ x_grid[j] | X_t = x_grid[i])`` using the AR(1)
    conditional distribution X_{t+1} | X_t = x  ~  N(φ x, σ_ε²) and
    midpoint bins between grid points (plus open-ended tails outside).

    Returns
    -------
    np.ndarray, shape (n_X, n_X)
        Row-stochastic transition matrix.
    """
    # Edges between adjacent grid points; tails extend to ±∞.
    midpoints = 0.5 * (x_grid[:-1] + x_grid[1:])
    edges = np.concatenate(([-np.inf], midpoints, [np.inf]))

    # Conditional means for each starting grid point.
    mu = model.phi * x_grid                          # shape (n_X,)

    # Standardized edges:  z[i, k] = (edges[k] - mu[i]) / sigma_eps
    z = (edges[None, :] - mu[:, None]) / model.sigma_eps
    cdf = norm.cdf(z)

    # Cell mass = CDF(right edge) − CDF(left edge).
    P = cdf[:, 1:] - cdf[:, :-1]

    # Normalise rows (safety against floating-point drift).
    P /= P.sum(axis=1, keepdims=True)
    return P


def _interp_along_s(values_2d: np.ndarray, s_grid: np.ndarray, s_query: np.ndarray) -> np.ndarray:
    """
    Linear interpolation along the S axis of a (n_S, n_X) array.

    For each query position ``s_query[i]`` (one per row of the input),
    interpolate across ``s_grid`` for *every* X column simultaneously
    using vectorised index arithmetic.

    Parameters
    ----------
    values_2d : np.ndarray, shape (n_S, n_X)
        The function being interpolated, V(S, X).
    s_grid : np.ndarray, shape (n_S,)
        Monotonically increasing grid.
    s_query : np.ndarray, shape (n_S,)
        Where to evaluate V along S, one per starting position.

    Returns
    -------
    np.ndarray, shape (n_S, n_X)
        V at the query positions, for each X.
    """
    n_S = len(s_grid)
    # searchsorted gives the right-bracket index; subtract 1 for left index.
    idx = np.clip(np.searchsorted(s_grid, s_query) - 1, 0, n_S - 2)
    s_left = s_grid[idx]
    s_right = s_grid[idx + 1]
    # Interpolation weight in [0, 1].
    w = ((s_query - s_left) / (s_right - s_left)).clip(0.0, 1.0)
    # Pull the bracketing rows out and blend.
    v_left = values_2d[idx, :]                       # shape (n_S, n_X)
    v_right = values_2d[idx + 1, :]
    return v_left * (1.0 - w[:, None]) + v_right * w[:, None]


# -----------------------------------------------------------------------------
# Core: backward induction
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ValueGrid:
    """
    Result of one backward-induction solve.

    Attributes
    ----------
    V : np.ndarray, shape (T+1, n_S, n_X)
        V[t, s, x] = optimal expected NPV from day t until horizon T,
        starting from S = S_grid[s], X = X_grid[x].
    policy : np.ndarray, shape (T, n_S, n_X), dtype int8
        Optimal action: 0 = hold, 1 = produce.
    S_grid : np.ndarray, shape (n_S,)
        Reservoir-level grid (MWh).
    X_grid : np.ndarray, shape (n_X,)
        Residual grid (log-units).
    doy_array : np.ndarray, shape (T+1,)
        Day-of-year at each time step (1 = Jan 1).
    spec : ReservoirSpec
    model : SchwartzModel
    """

    V: np.ndarray
    policy: np.ndarray
    S_grid: np.ndarray
    X_grid: np.ndarray
    doy_array: np.ndarray
    spec: ReservoirSpec
    model: SchwartzModel

    def water_value(self, t: int, x_index: int | None = None) -> np.ndarray:
        """
        Forward-difference water value ∂V/∂S at time t, conditional on
        residual X = X_grid[x_index] (default: x_index for X ≈ 0).

        Returns shape (n_S - 1,) in EUR/MWh.
        """
        if x_index is None:
            x_index = int(np.argmin(np.abs(self.X_grid)))
        V_t = self.V[t, :, x_index]
        dS = self.S_grid[1] - self.S_grid[0]
        return (V_t[1:] - V_t[:-1]) / dS

    def water_value_midpoints(self) -> np.ndarray:
        """Mid-grid S coordinates that align with :meth:`water_value` output."""
        return 0.5 * (self.S_grid[:-1] + self.S_grid[1:])


def solve_water_value(
    model: SchwartzModel,
    spec: ReservoirSpec,
    horizon_days: int = 365,
    start_doy: int = 1,
    n_S: int = 121,
    n_X: int = 41,
    x_range_in_std: float = 4.0,
) -> ValueGrid:
    """
    Backward-induction Bellman solver.

    Parameters
    ----------
    model : SchwartzModel
        Calibrated price model.
    spec : ReservoirSpec
        Physical reservoir/plant spec.
    horizon_days : int, default 365
        Number of daily decision steps. Terminal value V(T, ·, ·) = 0.
    start_doy : int, default 1
        Day-of-year for t = 0 (1 = Jan 1).
    n_S : int, default 121
        Number of grid points for reservoir level. Finer = more accurate,
        slower. 121 with capacity 120 GWh gives ~1 GWh resolution.
    n_X : int, default 41
        Number of grid points for residual X. 41 covering ±4σ_stat is
        typically more than enough.
    x_range_in_std : float, default 4.0
        How many stationary standard deviations of X to span.

    Returns
    -------
    ValueGrid
    """
    K = spec.capacity_mwh
    Q = spec.daily_production_mwh
    I = spec.daily_inflow_mwh
    r_daily = spec.discount_rate_per_year / 365.0
    discount = float(np.exp(-r_daily))

    # ----- Grids -------------------------------------------------------------
    S_grid = np.linspace(0.0, K, n_S)
    sigma_X_stat = model.sigma / np.sqrt(2.0 * model.kappa)  # stationary std of X
    x_max = x_range_in_std * sigma_X_stat
    X_grid = np.linspace(-x_max, x_max, n_X)

    # ----- Transition matrix for X ------------------------------------------
    P_trans = x_transition_matrix(model, X_grid)         # (n_X, n_X)

    # ----- Calendar (day-of-year for each time step) ------------------------
    doy_array = ((start_doy - 1 + np.arange(horizon_days + 1)) % 365) + 1

    # ----- Next-S after each action (continuous units of MWh) ---------------
    # Produce: lose Q, then receive I, then clip.
    S_next_produce = np.clip(S_grid - Q + I, 0.0, K)
    # Hold: receive I, then clip (spill if exceeds K).
    S_next_hold = np.clip(S_grid + I, 0.0, K)
    # Feasibility: can produce only if reservoir has at least Q in it today.
    can_produce = S_grid >= Q                              # shape (n_S,), bool

    # ----- Allocate value function and policy --------------------------------
    V = np.zeros((horizon_days + 1, n_S, n_X), dtype=np.float64)
    policy = np.zeros((horizon_days, n_S, n_X), dtype=np.int8)

    # ----- Backward induction main loop --------------------------------------
    for t in range(horizon_days - 1, -1, -1):
        # E_V_next[s, i]  =  Σ_j P_trans[i, j] V[t+1, s, j]   for current X = x_i.
        # That's a matmul: V[t+1] @ P_trans.T   (shape (n_S, n_X)).
        E_V_next = V[t + 1] @ P_trans.T

        # Interpolate this conditional expectation at the post-action S values.
        E_after_produce = _interp_along_s(E_V_next, S_grid, S_next_produce)
        E_after_hold    = _interp_along_s(E_V_next, S_grid, S_next_hold)

        # Prices at all X for today's day-of-year.
        f_t = float(model.seasonal(doy_array[t])[0])
        prices = np.exp(f_t + X_grid)                     # shape (n_X,)

        # Action values.
        val_produce = Q * prices[None, :] + discount * E_after_produce  # (n_S, n_X)
        val_hold    = discount * E_after_hold

        # Mask out infeasible-produce rows (S < Q) so they never win.
        val_produce[~can_produce, :] = -np.inf

        # Optimal: take the better of the two actions, store both value and choice.
        stacked = np.stack([val_hold, val_produce], axis=0)            # (2, n_S, n_X)
        V[t] = stacked.max(axis=0)
        policy[t] = stacked.argmax(axis=0).astype(np.int8)

    return ValueGrid(
        V=V,
        policy=policy,
        S_grid=S_grid,
        X_grid=X_grid,
        doy_array=doy_array,
        spec=spec,
        model=model,
    )


# -----------------------------------------------------------------------------
# Forward simulation under the optimal policy (for visualisation)
# -----------------------------------------------------------------------------

def simulate_optimal_run(
    grid: ValueGrid,
    initial_fill: float,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Simulate one realisation of the price path and play the optimal policy.

    Parameters
    ----------
    grid : ValueGrid
        Solved value/policy grid.
    initial_fill : float
        Starting reservoir level S_0 in MWh. Must lie in [0, K].
    seed : int, optional
        Random seed for the price-path noise.

    Returns
    -------
    pd.DataFrame, indexed by step (day), with columns:
        doy            : day-of-year
        X              : residual
        price_eur_mwh  : spot price that day
        S_start        : reservoir level at start of day (MWh)
        action         : 0 = hold, 1 = produce
        cash_flow_eur  : Q × price on producing days, 0 otherwise
        S_end          : reservoir level after action + inflow (MWh)
    """
    spec = grid.spec
    model = grid.model
    rng = np.random.default_rng(seed)
    T = len(grid.policy)
    K = spec.capacity_mwh
    Q = spec.daily_production_mwh
    I = spec.daily_inflow_mwh

    if not (0.0 <= initial_fill <= K):
        raise ValueError(f"initial_fill={initial_fill} outside [0, {K}]")

    # Pre-generate residual path
    X_path = np.zeros(T + 1)
    X_path[0] = 0.0  # start at the unconditional mean
    for i in range(1, T + 1):
        X_path[i] = model.phi * X_path[i - 1] + rng.normal(0.0, model.sigma_eps)

    records = []
    S = float(initial_fill)
    for t in range(T):
        x = X_path[t]
        doy = int(grid.doy_array[t])

        # Look up policy at (S, X) using nearest neighbour on each axis.
        # (Linear-interpolated policy doesn't make sense — it's categorical.
        # For value lookups we use interpolation; for actions we snap.)
        s_idx = int(np.clip(np.searchsorted(grid.S_grid, S), 0, len(grid.S_grid) - 1))
        x_idx = int(np.argmin(np.abs(grid.X_grid - x)))
        action = int(grid.policy[t, s_idx, x_idx])

        # Override if infeasible (defensive — solver already masks this out).
        if action == 1 and S < Q:
            action = 0

        price = float(np.exp(model.seasonal(doy)[0] + x))
        cash = Q * price if action == 1 else 0.0
        S_next = float(np.clip(S - (Q if action == 1 else 0.0) + I, 0.0, K))

        records.append({
            "doy": doy,
            "X": x,
            "price_eur_mwh": price,
            "S_start": S,
            "action": action,
            "cash_flow_eur": cash,
            "S_end": S_next,
        })
        S = S_next

    return pd.DataFrame(records)
