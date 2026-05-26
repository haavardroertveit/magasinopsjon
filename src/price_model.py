"""
Schwartz 1-factor price model for daily NO2 spot.

The model
---------
    log P_t = f(t) + X_t
    f(t)    = β_0 + β_1 sin(ω t) + β_2 cos(ω t),    ω = 2π / 365.25
    dX_t    = -κ X_t dt + σ dW_t                    (Ornstein-Uhlenbeck)

Discrete-time AR(1) at Δt = 1 day:
    X_{t+1} = φ X_t + ε_t,    ε_t ~ N(0, σ_ε²)
    φ       = exp(-κ Δt)

Calibration recipe (three short OLS steps)
------------------------------------------
    1. β from OLS of log P_t on [1, sin(ω t), cos(ω t)]  where t = day-of-year.
    2. φ from OLS of residual X_t on its lag, no intercept (residual is
       zero-mean by construction after step 1).
    3. σ_ε is the std of the AR(1) innovation. Continuous-time σ then follows
       from the exact OU variance formula:
           Var(X_{t+Δt} | X_t) = σ² (1 − e^{−2κΔt}) / (2κ)
       so σ = σ_ε · √(2κ / (1 − e^{−2κΔt})).

The module exposes:
    SchwartzModel  — immutable container of calibrated parameters + utilities
    fit_seasonal   — step 1 (alone)
    fit_ar1        — step 2 (alone)
    to_continuous  — step 3 conversion
    fit_schwartz   — end-to-end: returns a SchwartzModel

References
----------
    Schwartz, E. (1997). The Stochastic Behavior of Commodity Prices. JoF.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


# Period of the seasonal cycle in days. We use the tropical year (365.25)
# rather than 365 so that the phase doesn't slowly drift across leap years.
TROPICAL_YEAR_DAYS: float = 365.25
OMEGA: float = 2 * np.pi / TROPICAL_YEAR_DAYS


# -----------------------------------------------------------------------------
# Calibrated-model container
# -----------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class SchwartzModel:
    """
    Calibrated Schwartz 1-factor model. Immutable (``frozen=True``) so it can be
    passed around between functions and notebooks without anyone modifying it
    in-place. ``eq=False`` because the ``beta`` field is a numpy array, which
    doesn't play nicely with dataclass-generated equality.

    Attributes
    ----------
    beta : np.ndarray, shape (3,)
        Seasonal coefficients [β_0, β_1, β_2] on [1, sin(ωt), cos(ωt)].
    phi : float
        Discrete AR(1) coefficient on the (zero-mean) residual.
    sigma_eps : float
        Standard deviation of one-step (daily) innovation ε.
    kappa : float
        Continuous-time mean-reversion speed (per day).
    sigma : float
        Continuous-time diffusion coefficient (per √day).
    dt : float
        Time-step that ``phi`` and ``sigma_eps`` correspond to. Always 1.0 in
        v1 (daily data), but parameterized so we can extend later.
    seasonal_r2 : float
        R² of the seasonal OLS fit on log-price (in-sample).
    n_obs : int
        Number of observations the model was calibrated on. Useful for
        comparing post-crisis vs. full-sample fits later.
    """

    beta: np.ndarray
    phi: float
    sigma_eps: float
    kappa: float
    sigma: float
    dt: float = 1.0
    seasonal_r2: float = 0.0
    n_obs: int = 0

    # ----- Derived quantities -------------------------------------------------

    @property
    def half_life_days(self) -> float:
        """Time for the residual to revert halfway to zero: ln 2 / κ."""
        return float(np.log(2) / self.kappa)

    @property
    def unconditional_log_var(self) -> float:
        """Stationary variance of X_t under the OU process: σ² / (2κ)."""
        return float(self.sigma ** 2 / (2 * self.kappa))

    @property
    def unconditional_mean_eur_mwh(self) -> float:
        """
        Stationary mean of P_t (not log P_t!), averaged over the year.

        For log P = f(t) + X with X ~ N(0, σ²/(2κ)) stationary, the marginal
        price has a log-normal-with-seasonal-drift distribution. The annual
        mean uses the lognormal formula E[exp(Y)] = exp(μ + ½σ²):
            E[P] = exp(β_0 + ½ · σ²/(2κ))     (since sin/cos integrate to 0).
        """
        return float(np.exp(self.beta[0] + 0.5 * self.unconditional_log_var))

    # ----- Functional evaluations --------------------------------------------

    def seasonal(self, day_of_year: np.ndarray | float) -> np.ndarray:
        """
        Evaluate the deterministic seasonal component f(t) at given day(s)
        of the year (1 = Jan 1).
        """
        t = np.atleast_1d(day_of_year).astype(float)
        return self.beta[0] + self.beta[1] * np.sin(OMEGA * t) + self.beta[2] * np.cos(OMEGA * t)

    def simulate(
        self,
        dates: pd.DatetimeIndex,
        X0: float = 0.0,
        n_paths: int = 1000,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Simulate Monte Carlo price paths over ``dates``.

        The first row corresponds to ``dates[0]`` with residual = X0;
        subsequent rows evolve via the AR(1) recursion. Seasonal component
        is added deterministically.

        Parameters
        ----------
        dates : pd.DatetimeIndex
            Daily timestamps. Length determines simulation horizon.
        X0 : float
            Starting value of the residual (log-deviation from seasonal mean).
        n_paths : int
            Monte Carlo sample size.
        seed : int, optional
            For reproducibility.

        Returns
        -------
        np.ndarray, shape (len(dates), n_paths)
            Simulated price levels in EUR/MWh.
        """
        rng = np.random.default_rng(seed)
        n = len(dates)

        # Pre-allocate residual matrix and iterate the AR(1) recursion.
        # We do this as a python loop over time (n is small, ~365) and
        # vectorize across paths.
        X = np.empty((n, n_paths))
        X[0, :] = X0
        for i in range(1, n):
            eps = rng.normal(0.0, self.sigma_eps, size=n_paths)
            X[i, :] = self.phi * X[i - 1, :] + eps

        # Add seasonal component, broadcasting across paths.
        doy = dates.dayofyear.values
        f_vals = self.seasonal(doy)[:, None]  # shape (n, 1)

        log_prices = f_vals + X
        return np.exp(log_prices)

    # ----- Persistence --------------------------------------------------------

    def to_dict(self) -> dict:
        """Plain-Python view of the parameters, JSON-serialisable."""
        return {
            "beta": self.beta.tolist(),
            "phi": float(self.phi),
            "sigma_eps": float(self.sigma_eps),
            "kappa": float(self.kappa),
            "sigma": float(self.sigma),
            "dt": float(self.dt),
            "seasonal_r2": float(self.seasonal_r2),
            "n_obs": int(self.n_obs),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SchwartzModel":
        """Inverse of :meth:`to_dict`."""
        return cls(
            beta=np.array(d["beta"], dtype=float),
            phi=d["phi"],
            sigma_eps=d["sigma_eps"],
            kappa=d["kappa"],
            sigma=d["sigma"],
            dt=d.get("dt", 1.0),
            seasonal_r2=d.get("seasonal_r2", 0.0),
            n_obs=d.get("n_obs", 0),
        )

    def save(self, path: str | Path) -> None:
        """Write parameters to a JSON file (human-readable, version-control-able)."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "SchwartzModel":
        """Read parameters from a JSON file written by :meth:`save`."""
        return cls.from_dict(json.loads(Path(path).read_text()))


# -----------------------------------------------------------------------------
# Individual calibration steps (also useful in isolation for the notebook)
# -----------------------------------------------------------------------------

def fit_seasonal(
    dates: pd.DatetimeIndex,
    log_price: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Step 1: OLS-fit the seasonal component f(t).

    Parameters
    ----------
    dates : pd.DatetimeIndex
    log_price : np.ndarray
        log(price) at each date. Must be finite (no NaN/Inf).

    Returns
    -------
    beta : np.ndarray, shape (3,)
        Coefficients on [1, sin(ωt), cos(ωt)].
    fitted : np.ndarray
        f(t_i) for each input date.
    residual : np.ndarray
        ``log_price - fitted``.
    r2 : float
        Coefficient of determination 1 - var(residual)/var(log_price).
    """
    t = dates.dayofyear.values.astype(float)
    X = np.column_stack([
        np.ones_like(t),
        np.sin(OMEGA * t),
        np.cos(OMEGA * t),
    ])
    beta, *_ = np.linalg.lstsq(X, log_price, rcond=None)
    fitted = X @ beta
    residual = log_price - fitted
    r2 = 1.0 - residual.var() / log_price.var()
    return beta, fitted, residual, float(r2)


def fit_ar1(residual: np.ndarray) -> tuple[float, float]:
    """
    Step 2: Fit AR(1) without intercept on a zero-mean residual.

        X_{t+1} = φ X_t + ε,   ε ~ N(0, σ_ε²)

    Closed-form OLS without intercept:  φ = Σ X_t X_{t+1} / Σ X_t².
    """
    x_prev = residual[:-1]
    x_curr = residual[1:]
    phi = float(np.sum(x_prev * x_curr) / np.sum(x_prev ** 2))
    eps = x_curr - phi * x_prev
    # ddof=1 — sample std (we estimated φ, so we lose one degree of freedom).
    sigma_eps = float(eps.std(ddof=1))
    return phi, sigma_eps


def to_continuous(phi: float, sigma_eps: float, dt: float = 1.0) -> tuple[float, float]:
    """
    Step 3: convert discrete AR(1) parameters to continuous-time OU parameters.

    Uses the *exact* OU variance formula so the conversion is accurate even
    when κΔt is not small.
    """
    if not (0.0 < phi < 1.0):
        # If φ falls outside (0, 1) the residual is not mean-reverting in the
        # OU sense — most likely a calibration data issue. Surface it loudly.
        raise ValueError(
            f"phi={phi:.4f} is outside (0, 1). Residual is not mean-reverting; "
            "check that the seasonal fit removed the trend and that the input "
            "is daily and contiguous."
        )
    kappa = -np.log(phi) / dt
    sigma = sigma_eps * np.sqrt(2 * kappa / (1.0 - np.exp(-2 * kappa * dt)))
    return float(kappa), float(sigma)


def fit_schwartz(
    dates: pd.DatetimeIndex,
    prices: np.ndarray,
    clip_floor: float = 1.0,
) -> SchwartzModel:
    """
    End-to-end calibration: returns a :class:`SchwartzModel`.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Daily timestamps (contiguous, no gaps).
    prices : np.ndarray
        Daily EUR/MWh prices, same length as ``dates``.
    clip_floor : float
        Lower bound applied to ``prices`` before log-transform, to handle
        the few zero/negative observations in the NO2 sample. The choice
        affects the calibration only marginally; we discuss it in the
        notebook.
    """
    log_p = np.log(np.maximum(prices, clip_floor))
    beta, _fitted, residual, r2 = fit_seasonal(dates, log_p)
    phi, sigma_eps = fit_ar1(residual)
    kappa, sigma = to_continuous(phi, sigma_eps, dt=1.0)
    return SchwartzModel(
        beta=beta,
        phi=phi,
        sigma_eps=sigma_eps,
        kappa=kappa,
        sigma=sigma,
        dt=1.0,
        seasonal_r2=r2,
        n_obs=len(prices),
    )
