"""
Basis Statistical Analysis

Computes mean-reversion and distributional statistics for basis series.
Uses statsmodels for ADF test and OLS half-life estimation.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def compute_basis_stats(basis_bps: pd.Series, interval: str) -> dict:
    """
    Compute comprehensive statistics for a basis series.

    Args:
        basis_bps: Series of basis values in basis points
        interval: Data interval string (e.g., '1m', '1h', '1d') for rate conversions

    Returns:
        Dict with all computed statistics
    """
    clean = basis_bps.dropna()
    if len(clean) < 30:
        return {"error": "Insufficient data (need >= 30 observations)"}

    bars_per_day = _bars_per_day(interval)

    stats = {}

    # --- Basic distribution ---
    stats["mean_bps"] = float(clean.mean())
    stats["std_bps"] = float(clean.std())
    stats["min_bps"] = float(clean.min())
    stats["max_bps"] = float(clean.max())
    stats["median_bps"] = float(clean.median())
    stats["skewness"] = float(clean.skew())
    stats["kurtosis"] = float(clean.kurtosis())  # excess kurtosis

    # --- Current state ---
    stats["current_bps"] = float(clean.iloc[-1])
    stats["zscore"] = float((clean.iloc[-1] - clean.mean()) / clean.std()) if clean.std() > 0 else 0.0

    # --- Directional ---
    stats["pct_positive"] = float((clean > 0).mean() * 100)
    stats["pct_negative"] = float((clean < 0).mean() * 100)

    # --- Mean reversion ---
    stats.update(_adf_test(clean))
    stats.update(_half_life(clean, bars_per_day))
    stats.update(_hurst_exponent(clean))
    stats.update(_mean_crossing_rate(clean, bars_per_day))

    return stats


def _adf_test(series: pd.Series) -> dict:
    """Augmented Dickey-Fuller test for stationarity."""
    try:
        result = adfuller(series, maxlag=None, autolag="AIC")
        return {
            "adf_statistic": float(result[0]),
            "adf_pvalue": float(result[1]),
            "adf_stationary": result[1] < 0.05,
        }
    except Exception:
        return {"adf_statistic": None, "adf_pvalue": None, "adf_stationary": None}


def _half_life(series: pd.Series, bars_per_day: float) -> dict:
    """
    Estimate mean-reversion half-life via OLS on the Ornstein-Uhlenbeck process.

    Model: ΔS_t = β · S_{t-1} + ε
    Half-life = -ln(2) / β  (in bars), then converted to days.
    """
    try:
        lagged = series.shift(1)
        delta = series - lagged
        # Drop NaN from shift
        lagged = lagged.iloc[1:].values
        delta = delta.iloc[1:].values

        lagged_with_const = add_constant(lagged)
        model = OLS(delta, lagged_with_const).fit()
        beta = model.params[1]

        if beta >= 0:
            # Not mean-reverting
            return {"half_life_bars": None, "half_life_days": None}

        hl_bars = -np.log(2) / beta
        hl_days = hl_bars / bars_per_day if bars_per_day > 0 else None

        return {
            "half_life_bars": float(hl_bars),
            "half_life_days": float(hl_days) if hl_days is not None else None,
        }
    except Exception:
        return {"half_life_bars": None, "half_life_days": None}


def _hurst_exponent(series: pd.Series, max_lags: int = 100) -> dict:
    """
    Estimate Hurst exponent via rescaled range (R/S) analysis.

    H < 0.5: mean-reverting
    H = 0.5: random walk
    H > 0.5: trending
    """
    try:
        ts = series.values
        n = len(ts)
        max_k = min(max_lags, n // 4)
        if max_k < 4:
            return {"hurst_exponent": None, "hurst_regime": None}

        lags = range(4, max_k + 1)
        rs_values = []

        for lag in lags:
            rs_lag = []
            for start in range(0, n - lag, lag):
                chunk = ts[start:start + lag]
                mean_chunk = chunk.mean()
                cumdev = np.cumsum(chunk - mean_chunk)
                r = cumdev.max() - cumdev.min()
                s = chunk.std(ddof=1)
                if s > 0:
                    rs_lag.append(r / s)
            if rs_lag:
                rs_values.append((np.log(lag), np.log(np.mean(rs_lag))))

        if len(rs_values) < 3:
            return {"hurst_exponent": None, "hurst_regime": None}

        log_lags, log_rs = zip(*rs_values)
        coeffs = np.polyfit(log_lags, log_rs, 1)
        h = float(coeffs[0])

        if h < 0.45:
            regime = "mean-reverting"
        elif h > 0.55:
            regime = "trending"
        else:
            regime = "random walk"

        return {"hurst_exponent": h, "hurst_regime": regime}
    except Exception:
        return {"hurst_exponent": None, "hurst_regime": None}


def _mean_crossing_rate(series: pd.Series, bars_per_day: float) -> dict:
    """Count how often the basis crosses its mean, expressed as crossings per day."""
    try:
        mean_val = series.mean()
        above = series > mean_val
        crossings = (above != above.shift(1)).sum() - 1  # subtract initial NaN transition
        crossings = max(0, crossings)
        crossings_per_day = float(crossings / len(series) * bars_per_day) if bars_per_day > 0 else 0
        return {
            "mean_crossings_total": int(crossings),
            "mean_crossings_per_day": crossings_per_day,
        }
    except Exception:
        return {"mean_crossings_total": None, "mean_crossings_per_day": None}


def _bars_per_day(interval: str) -> float:
    """Convert interval string to approximate bars per day."""
    mapping = {
        "1m": 1440,
        "3m": 480,
        "5m": 288,
        "15m": 96,
        "30m": 48,
        "1h": 24,
        "2h": 12,
        "4h": 6,
        "6h": 4,
        "8h": 3,
        "12h": 2,
        "1d": 1,
    }
    return mapping.get(interval, 24)
