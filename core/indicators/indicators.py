"""
Technical indicators module using talipp.

talipp is an incremental technical analysis library — indicators update in O(1)
when new data arrives, making it ideal for both batch backtesting and live streaming.

Usage:
    from core.indicators import compute_indicators, INDICATOR_PRESETS
    
    # Add specific indicators
    df = compute_indicators(df, [
        ("sma", {"length": 20}),
        ("rsi", {"length": 14}),
        ("macd", {"fast": 12, "slow": 26, "signal": 9}),
    ])
    
    # Or use a preset
    df = compute_indicators(df, INDICATOR_PRESETS["trend"])
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from talipp.ohlcv import OHLCV
from talipp.indicators import (
    SMA, EMA, WMA, DEMA, TEMA, HMA, KAMA, SMMA, ZLEMA, T3, McGinleyDynamic, ALMA,
    MACD, ADX, Aroon, Ichimoku, SuperTrend, ParabolicSAR, DPO, TRIX,
    RSI, Stoch, StochRSI, CCI, Williams, ROC, BOP, UO, TSI, AO,
    BB, ATR, NATR, KeltnerChannels, DonchianChannels, CHOP,
    OBV, SOBV, VWAP, AccuDist, ForceIndex, EMV, KVO,
)


# ── Indicator metadata registry ──
# Each entry describes an indicator for the UI: category, display type
# (overlay = drawn on price chart, panel = separate sub-chart), default params,
# and a human-readable label.

INDICATOR_REGISTRY: dict[str, dict[str, Any]] = {
    # ── Trend / Moving Averages ──
    "sma":      {"label": "Simple Moving Average",      "category": "trend",      "display": "overlay", "params": {"length": {"default": 20, "type": "int", "min": 2, "max": 500}}},
    "ema":      {"label": "Exponential Moving Average",  "category": "trend",      "display": "overlay", "params": {"length": {"default": 20, "type": "int", "min": 2, "max": 500}}},
    "wma":      {"label": "Weighted Moving Average",     "category": "trend",      "display": "overlay", "params": {"length": {"default": 20, "type": "int", "min": 2, "max": 500}}},
    "dema":     {"label": "Double EMA",                  "category": "trend",      "display": "overlay", "params": {"length": {"default": 20, "type": "int", "min": 2, "max": 500}}},
    "tema":     {"label": "Triple EMA",                  "category": "trend",      "display": "overlay", "params": {"length": {"default": 20, "type": "int", "min": 2, "max": 500}}},
    "hma":      {"label": "Hull Moving Average",         "category": "trend",      "display": "overlay", "params": {"length": {"default": 20, "type": "int", "min": 2, "max": 500}}},
    "kama":     {"label": "Kaufman Adaptive MA",         "category": "trend",      "display": "overlay", "params": {"length": {"default": 10, "type": "int", "min": 2, "max": 200}, "fast": {"default": 2, "type": "int", "min": 1, "max": 50}, "slow": {"default": 30, "type": "int", "min": 5, "max": 200}}},
    "smma":     {"label": "Smoothed Moving Average",     "category": "trend",      "display": "overlay", "params": {"length": {"default": 20, "type": "int", "min": 2, "max": 500}}},
    "zlema":    {"label": "Zero-Lag EMA",                "category": "trend",      "display": "overlay", "params": {"length": {"default": 20, "type": "int", "min": 2, "max": 500}}},
    "t3":       {"label": "T3 Moving Average",           "category": "trend",      "display": "overlay", "params": {"length": {"default": 5, "type": "int", "min": 2, "max": 200}, "factor": {"default": 0.7, "type": "float", "min": 0.0, "max": 1.0}}},
    "alma":     {"label": "Arnaud Legoux MA",            "category": "trend",      "display": "overlay", "params": {"length": {"default": 9, "type": "int", "min": 2, "max": 200}, "offset": {"default": 0.85, "type": "float", "min": 0.0, "max": 1.0}, "sigma": {"default": 6.0, "type": "float", "min": 0.1, "max": 20.0}}},
    "mcginley": {"label": "McGinley Dynamic",            "category": "trend",      "display": "overlay", "params": {"length": {"default": 14, "type": "int", "min": 2, "max": 200}}},
    "macd":     {"label": "MACD",                        "category": "trend",      "display": "panel",   "params": {"fast": {"default": 12, "type": "int", "min": 2, "max": 100}, "slow": {"default": 26, "type": "int", "min": 5, "max": 200}, "signal": {"default": 9, "type": "int", "min": 2, "max": 50}}},
    "adx":      {"label": "Average Directional Index",   "category": "trend",      "display": "panel",   "params": {"length": {"default": 14, "type": "int", "min": 2, "max": 100}}},
    "aroon":    {"label": "Aroon",                       "category": "trend",      "display": "panel",   "params": {"length": {"default": 25, "type": "int", "min": 2, "max": 200}}},
    "supertrend": {"label": "SuperTrend",                "category": "trend",      "display": "overlay", "params": {"atr_length": {"default": 10, "type": "int", "min": 2, "max": 100}, "mult": {"default": 3, "type": "int", "min": 1, "max": 10}}},
    "psar":     {"label": "Parabolic SAR",               "category": "trend",      "display": "overlay", "params": {"af": {"default": 0.02, "type": "float", "min": 0.001, "max": 0.2}, "af_step": {"default": 0.02, "type": "float", "min": 0.001, "max": 0.1}, "max_af": {"default": 0.2, "type": "float", "min": 0.05, "max": 0.5}}},
    "dpo":      {"label": "Detrended Price Oscillator",  "category": "trend",      "display": "panel",   "params": {"length": {"default": 20, "type": "int", "min": 2, "max": 200}}},
    "trix":     {"label": "TRIX",                        "category": "trend",      "display": "panel",   "params": {"length": {"default": 15, "type": "int", "min": 2, "max": 100}}},
    # ── Momentum ──
    "rsi":      {"label": "Relative Strength Index",     "category": "momentum",   "display": "panel",   "params": {"length": {"default": 14, "type": "int", "min": 2, "max": 100}}},
    "stoch":    {"label": "Stochastic Oscillator",       "category": "momentum",   "display": "panel",   "params": {"k": {"default": 14, "type": "int", "min": 2, "max": 100}, "d": {"default": 3, "type": "int", "min": 1, "max": 50}}},
    "stochrsi": {"label": "Stochastic RSI",              "category": "momentum",   "display": "panel",   "params": {"rsi_length": {"default": 14, "type": "int", "min": 2, "max": 100}, "stoch_length": {"default": 14, "type": "int", "min": 2, "max": 100}, "k": {"default": 3, "type": "int", "min": 1, "max": 50}, "d": {"default": 3, "type": "int", "min": 1, "max": 50}}},
    "cci":      {"label": "Commodity Channel Index",     "category": "momentum",   "display": "panel",   "params": {"length": {"default": 20, "type": "int", "min": 2, "max": 200}}},
    "willr":    {"label": "Williams %R",                 "category": "momentum",   "display": "panel",   "params": {"length": {"default": 14, "type": "int", "min": 2, "max": 100}}},
    "roc":      {"label": "Rate of Change",              "category": "momentum",   "display": "panel",   "params": {"length": {"default": 10, "type": "int", "min": 1, "max": 200}}},
    "bop":      {"label": "Balance of Power",            "category": "momentum",   "display": "panel",   "params": {}},
    "uo":       {"label": "Ultimate Oscillator",         "category": "momentum",   "display": "panel",   "params": {"fast": {"default": 7, "type": "int", "min": 1, "max": 50}, "mid": {"default": 14, "type": "int", "min": 2, "max": 100}, "slow": {"default": 28, "type": "int", "min": 5, "max": 200}}},
    "tsi":      {"label": "True Strength Index",         "category": "momentum",   "display": "panel",   "params": {"fast": {"default": 13, "type": "int", "min": 2, "max": 50}, "slow": {"default": 25, "type": "int", "min": 5, "max": 100}}},
    "ao":       {"label": "Awesome Oscillator",          "category": "momentum",   "display": "panel",   "params": {}},
    # ── Volatility ──
    "bbands":   {"label": "Bollinger Bands",             "category": "volatility", "display": "overlay", "params": {"length": {"default": 20, "type": "int", "min": 2, "max": 200}, "std": {"default": 2.0, "type": "float", "min": 0.5, "max": 5.0}}},
    "atr":      {"label": "Average True Range",          "category": "volatility", "display": "panel",   "params": {"length": {"default": 14, "type": "int", "min": 2, "max": 100}}},
    "natr":     {"label": "Normalised ATR",              "category": "volatility", "display": "panel",   "params": {"length": {"default": 14, "type": "int", "min": 2, "max": 100}}},
    "kc":       {"label": "Keltner Channels",            "category": "volatility", "display": "overlay", "params": {"ma_length": {"default": 20, "type": "int", "min": 2, "max": 200}, "atr_length": {"default": 10, "type": "int", "min": 2, "max": 100}, "mult": {"default": 1.5, "type": "float", "min": 0.5, "max": 5.0}}},
    "donchian": {"label": "Donchian Channels",           "category": "volatility", "display": "overlay", "params": {"length": {"default": 20, "type": "int", "min": 2, "max": 200}}},
    "chop":     {"label": "Choppiness Index",            "category": "volatility", "display": "panel",   "params": {"length": {"default": 14, "type": "int", "min": 2, "max": 100}}},
    # ── Volume ──
    "obv":      {"label": "On-Balance Volume",           "category": "volume",     "display": "panel",   "params": {}},
    "sobv":     {"label": "Smoothed OBV",                "category": "volume",     "display": "panel",   "params": {"length": {"default": 5, "type": "int", "min": 2, "max": 50}}},
    "vwap":     {"label": "VWAP",                        "category": "volume",     "display": "overlay", "params": {}},
    "ad":       {"label": "Accumulation/Distribution",   "category": "volume",     "display": "panel",   "params": {}},
    "force":    {"label": "Force Index",                 "category": "volume",     "display": "panel",   "params": {"length": {"default": 13, "type": "int", "min": 2, "max": 100}}},
    "emv":      {"label": "Ease of Movement",            "category": "volume",     "display": "panel",   "params": {"length": {"default": 14, "type": "int", "min": 2, "max": 100}}},
}

# Common indicator presets
INDICATOR_PRESETS = {
    "minimal": [
        ("sma", {"length": 20}),
        ("sma", {"length": 50}),
        ("rsi", {"length": 14}),
    ],
    "trend": [
        ("sma", {"length": 10}),
        ("sma", {"length": 20}),
        ("sma", {"length": 50}),
        ("sma", {"length": 200}),
        ("ema", {"length": 12}),
        ("ema", {"length": 26}),
        ("macd", {"fast": 12, "slow": 26, "signal": 9}),
        ("adx", {"length": 14}),
    ],
    "momentum": [
        ("rsi", {"length": 14}),
        ("stoch", {"k": 14, "d": 3}),
        ("cci", {"length": 20}),
        ("willr", {"length": 14}),
        ("roc", {"length": 10}),
    ],
    "volatility": [
        ("bbands", {"length": 20, "std": 2}),
        ("atr", {"length": 14}),
        ("natr", {"length": 14}),
        ("kc", {"ma_length": 20, "atr_length": 10, "mult": 1.5}),
    ],
    "volume": [
        ("obv", {}),
        ("vwap", {}),
    ],
    "full": [
        ("sma", {"length": 10}),
        ("sma", {"length": 20}),
        ("sma", {"length": 50}),
        ("sma", {"length": 200}),
        ("ema", {"length": 12}),
        ("ema", {"length": 26}),
        ("macd", {"fast": 12, "slow": 26, "signal": 9}),
        ("adx", {"length": 14}),
        ("rsi", {"length": 14}),
        ("stoch", {"k": 14, "d": 3}),
        ("cci", {"length": 20}),
        ("bbands", {"length": 20, "std": 2}),
        ("atr", {"length": 14}),
    ],
}


# Warmup bars needed per indicator type.
INDICATOR_WARMUP: dict[str, callable] = {
    # Trend / MA
    "sma":       lambda p: p.get("length", 20),
    "ema":       lambda p: p.get("length", 20),
    "wma":       lambda p: p.get("length", 20),
    "dema":      lambda p: 2 * p.get("length", 20),
    "tema":      lambda p: 3 * p.get("length", 20),
    "hma":       lambda p: p.get("length", 20),
    "kama":      lambda p: p.get("length", 10) + p.get("slow", 30),
    "smma":      lambda p: p.get("length", 20),
    "zlema":     lambda p: p.get("length", 20),
    "t3":        lambda p: 6 * p.get("length", 5),
    "alma":      lambda p: p.get("length", 9),
    "mcginley":  lambda p: p.get("length", 14),
    "macd":      lambda p: p.get("slow", 26) + p.get("signal", 9),
    "adx":       lambda p: 2 * p.get("length", 14),
    "aroon":     lambda p: p.get("length", 25),
    "supertrend": lambda p: p.get("atr_length", 10),
    "psar":      lambda p: 2,
    "dpo":       lambda p: p.get("length", 20),
    "trix":      lambda p: 3 * p.get("length", 15),
    # Momentum
    "rsi":       lambda p: p.get("length", 14) + 1,
    "stoch":     lambda p: p.get("k", 14) + p.get("d", 3),
    "stochrsi":  lambda p: p.get("rsi_length", 14) + p.get("stoch_length", 14),
    "cci":       lambda p: p.get("length", 20),
    "willr":     lambda p: p.get("length", 14),
    "roc":       lambda p: p.get("length", 10),
    "bop":       lambda p: 0,
    "uo":        lambda p: p.get("slow", 28),
    "tsi":       lambda p: p.get("slow", 25) + p.get("fast", 13),
    "ao":        lambda p: 34,
    # Volatility
    "bbands":    lambda p: p.get("length", 20),
    "atr":       lambda p: p.get("length", 14),
    "natr":      lambda p: p.get("length", 14),
    "kc":        lambda p: max(p.get("ma_length", 20), p.get("atr_length", 10)),
    "donchian":  lambda p: p.get("length", 20),
    "chop":      lambda p: p.get("length", 14),
    # Volume
    "obv":       lambda p: 0,
    "sobv":      lambda p: p.get("length", 5),
    "vwap":      lambda p: 0,
    "ad":        lambda p: 0,
    "force":     lambda p: p.get("length", 13),
    "emv":       lambda p: p.get("length", 14),
}


# ── Helpers ──

def _to_ohlcv(df: pd.DataFrame) -> list[OHLCV]:
    """Convert DataFrame rows to talipp OHLCV objects."""
    return [
        OHLCV(r.open, r.high, r.low, r.close, r.volume)
        for r in df[["open", "high", "low", "close", "volume"]].itertuples(index=False)
    ]


def _extract(indicator, length: int) -> list[float | None]:
    """Extract plain float values from a talipp indicator result list."""
    out: list[float | None] = []
    for v in indicator:
        if v is None:
            out.append(None)
        elif isinstance(v, (int, float)):
            out.append(float(v))
        else:
            out.append(None)
    # Pad to match df length
    while len(out) < length:
        out.insert(0, None)
    return out[:length]


def _to_series(values: list, index: pd.Index) -> pd.Series:
    """Convert a list of values (possibly None) to a pandas Series."""
    arr = np.array([float(v) if v is not None else np.nan for v in values], dtype=np.float64)
    if len(arr) < len(index):
        arr = np.concatenate([np.full(len(index) - len(arr), np.nan), arr])
    return pd.Series(arr[:len(index)], index=index)


# ── Public API ──

def get_warmup_bars(indicators: list[tuple[str, dict]]) -> int:
    """
    Calculate the maximum warmup bars needed across a list of indicators.
    
    Args:
        indicators: List of (indicator_name, params) tuples
        
    Returns:
        Maximum number of bars needed before all indicators produce valid values.
        Includes a 10% safety margin, rounded up.
    """
    if not indicators:
        return 0
    
    max_warmup = 0
    for name, params in indicators:
        fn = INDICATOR_WARMUP.get(name.lower())
        if fn:
            max_warmup = max(max_warmup, fn(params))
        else:
            vals = [v for v in params.values() if isinstance(v, (int, float))]
            max_warmup = max(max_warmup, max(vals) * 2 if vals else 50)
    
    return int(max_warmup * 1.1) + 1


def compute_indicators(
    df: pd.DataFrame,
    indicators: list[tuple[str, dict]],
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Compute technical indicators and add as columns.
    
    Args:
        df: OHLCV DataFrame with columns: open, high, low, close, volume
        indicators: List of (indicator_name, params) tuples
        inplace: If True, modify df in place; otherwise return copy
        
    Returns:
        DataFrame with indicator columns added
        
    Example:
        df = compute_indicators(df, [
            ("sma", {"length": 20}),
            ("rsi", {"length": 14}),
            ("bbands", {"length": 20, "std": 2}),
        ])
    """
    if not inplace:
        df = df.copy()
    
    for indicator_name, params in indicators:
        try:
            _add_indicator(df, indicator_name, params)
        except Exception as e:
            print(f"Warning: Failed to compute {indicator_name}: {e}")
    
    return df


def _add_indicator(df: pd.DataFrame, name: str, params: dict) -> None:
    """Add a single indicator to the DataFrame using talipp."""
    n = name.lower()
    idx = df.index
    closes = df["close"].tolist()
    ohlcv = None  # lazily built

    def get_ohlcv():
        nonlocal ohlcv
        if ohlcv is None:
            ohlcv = _to_ohlcv(df)
        return ohlcv

    def s(vals):
        return _to_series(list(vals), idx)

    # ── Trend / Moving Averages ──
    if n == "sma":
        length = params.get("length", 20)
        df[f"SMA_{length}"] = s(SMA(period=length, input_values=closes))

    elif n == "ema":
        length = params.get("length", 20)
        df[f"EMA_{length}"] = s(EMA(period=length, input_values=closes))

    elif n == "wma":
        length = params.get("length", 20)
        df[f"WMA_{length}"] = s(WMA(period=length, input_values=closes))

    elif n == "dema":
        length = params.get("length", 20)
        df[f"DEMA_{length}"] = s(DEMA(period=length, input_values=closes))

    elif n == "tema":
        length = params.get("length", 20)
        df[f"TEMA_{length}"] = s(TEMA(period=length, input_values=closes))

    elif n == "hma":
        length = params.get("length", 20)
        df[f"HMA_{length}"] = s(HMA(period=length, input_values=closes))

    elif n == "kama":
        length = params.get("length", 10)
        fast = params.get("fast", 2)
        slow = params.get("slow", 30)
        df[f"KAMA_{length}"] = s(KAMA(period=length, fast_ema_constant_period=fast, slow_ema_constant_period=slow, input_values=closes))

    elif n == "smma":
        length = params.get("length", 20)
        df[f"SMMA_{length}"] = s(SMMA(period=length, input_values=closes))

    elif n == "zlema":
        length = params.get("length", 20)
        df[f"ZLEMA_{length}"] = s(ZLEMA(period=length, input_values=closes))

    elif n == "t3":
        length = params.get("length", 5)
        factor = params.get("factor", 0.7)
        df[f"T3_{length}"] = s(T3(period=length, factor=factor, input_values=closes))

    elif n == "alma":
        length = params.get("length", 9)
        offset = params.get("offset", 0.85)
        sigma = params.get("sigma", 6.0)
        df[f"ALMA_{length}"] = s(ALMA(period=length, offset=offset, sigma=sigma, input_values=closes))

    elif n == "mcginley":
        length = params.get("length", 14)
        df[f"McGinley_{length}"] = s(McGinleyDynamic(period=length, input_values=closes))

    elif n == "macd":
        fast = params.get("fast", 12)
        slow = params.get("slow", 26)
        signal = params.get("signal", 9)
        ind = MACD(fast_period=fast, slow_period=slow, signal_period=signal, input_values=closes)
        tag = f"{fast}_{slow}_{signal}"
        df[f"MACD_{tag}"] = _to_series([v.macd if v else None for v in ind], idx)
        df[f"MACDs_{tag}"] = _to_series([v.signal if v else None for v in ind], idx)
        df[f"MACDh_{tag}"] = _to_series([v.histogram if v else None for v in ind], idx)

    elif n == "adx":
        length = params.get("length", 14)
        ind = ADX(di_period=length, adx_period=length, input_values=get_ohlcv())
        df[f"ADX_{length}"] = _to_series([v.adx if v else None for v in ind], idx)
        df[f"DMP_{length}"] = _to_series([v.plus_di if v else None for v in ind], idx)
        df[f"DMN_{length}"] = _to_series([v.minus_di if v else None for v in ind], idx)

    elif n == "aroon":
        length = params.get("length", 25)
        ind = Aroon(period=length, input_values=get_ohlcv())
        df[f"AroonUp_{length}"] = _to_series([v.up if v else None for v in ind], idx)
        df[f"AroonDown_{length}"] = _to_series([v.down if v else None for v in ind], idx)

    elif n == "supertrend":
        atr_length = params.get("atr_length", 10)
        mult = params.get("mult", 3)
        ind = SuperTrend(atr_period=atr_length, mult=mult, input_values=get_ohlcv())
        df[f"SuperTrend_{atr_length}_{mult}"] = _to_series([v.value if v else None for v in ind], idx)

    elif n == "psar":
        af = params.get("af", 0.02)
        af_step = params.get("af_step", 0.02)
        max_af = params.get("max_af", 0.2)
        ind = ParabolicSAR(init_accel_factor=af, accel_factor_inc=af_step, max_accel_factor=max_af, input_values=get_ohlcv())
        df["PSAR"] = _to_series([v.value if v else None for v in ind], idx)

    elif n == "dpo":
        length = params.get("length", 20)
        df[f"DPO_{length}"] = s(DPO(period=length, input_values=closes))

    elif n == "trix":
        length = params.get("length", 15)
        df[f"TRIX_{length}"] = s(TRIX(period=length, input_values=closes))

    # ── Momentum ──
    elif n == "rsi":
        length = params.get("length", 14)
        df[f"RSI_{length}"] = s(RSI(period=length, input_values=closes))

    elif n == "stoch":
        k = params.get("k", 14)
        d = params.get("d", 3)
        ind = Stoch(period=k, smoothing_period=d, input_values=get_ohlcv())
        df[f"STOCHk_{k}_{d}"] = _to_series([v.k if v else None for v in ind], idx)
        df[f"STOCHd_{k}_{d}"] = _to_series([v.d if v else None for v in ind], idx)

    elif n == "stochrsi":
        rsi_l = params.get("rsi_length", 14)
        stoch_l = params.get("stoch_length", 14)
        k_smooth = params.get("k", 3)
        d_smooth = params.get("d", 3)
        ind = StochRSI(rsi_period=rsi_l, stoch_period=stoch_l, k_smoothing_period=k_smooth, d_smoothing_period=d_smooth, input_values=closes)
        df[f"StochRSIk_{rsi_l}"] = _to_series([v.k if v else None for v in ind], idx)
        df[f"StochRSId_{rsi_l}"] = _to_series([v.d if v else None for v in ind], idx)

    elif n == "cci":
        length = params.get("length", 20)
        df[f"CCI_{length}"] = s(CCI(period=length, input_values=get_ohlcv()))

    elif n == "willr":
        length = params.get("length", 14)
        df[f"WILLR_{length}"] = s(Williams(period=length, input_values=get_ohlcv()))

    elif n == "roc":
        length = params.get("length", 10)
        df[f"ROC_{length}"] = s(ROC(period=length, input_values=closes))

    elif n == "bop":
        df["BOP"] = s(BOP(input_values=get_ohlcv()))

    elif n == "uo":
        fast = params.get("fast", 7)
        mid = params.get("mid", 14)
        slow = params.get("slow", 28)
        df[f"UO_{fast}_{mid}_{slow}"] = s(UO(fast_period=fast, mid_period=mid, slow_period=slow, input_values=get_ohlcv()))

    elif n == "tsi":
        fast = params.get("fast", 13)
        slow = params.get("slow", 25)
        df[f"TSI_{fast}_{slow}"] = s(TSI(fast_period=fast, slow_period=slow, input_values=closes))

    elif n == "ao":
        df["AO"] = s(AO(fast_period=5, slow_period=34, input_values=get_ohlcv()))

    # ── Volatility ──
    elif n == "bbands":
        length = params.get("length", 20)
        std = params.get("std", 2.0)
        ind = BB(period=length, std_dev_mult=float(std), input_values=closes)
        tag = f"{length}_{std}"
        df[f"BBL_{tag}"] = _to_series([v.lb if v else None for v in ind], idx)
        df[f"BBM_{tag}"] = _to_series([v.cb if v else None for v in ind], idx)
        df[f"BBU_{tag}"] = _to_series([v.ub if v else None for v in ind], idx)

    elif n == "atr":
        length = params.get("length", 14)
        df[f"ATR_{length}"] = s(ATR(period=length, input_values=get_ohlcv()))

    elif n == "natr":
        length = params.get("length", 14)
        df[f"NATR_{length}"] = s(NATR(period=length, input_values=get_ohlcv()))

    elif n == "kc":
        ma_length = params.get("ma_length", 20)
        atr_length = params.get("atr_length", 10)
        mult = params.get("mult", 1.5)
        ind = KeltnerChannels(ma_period=ma_length, atr_period=atr_length, atr_mult_up=float(mult), atr_mult_down=float(mult), input_values=get_ohlcv())
        tag = f"{ma_length}"
        df[f"KCL_{tag}"] = _to_series([v.lb if v else None for v in ind], idx)
        df[f"KCM_{tag}"] = _to_series([v.cb if v else None for v in ind], idx)
        df[f"KCU_{tag}"] = _to_series([v.ub if v else None for v in ind], idx)

    elif n == "donchian":
        length = params.get("length", 20)
        ind = DonchianChannels(period=length, input_values=get_ohlcv())
        df[f"DCL_{length}"] = _to_series([v.lb if v else None for v in ind], idx)
        df[f"DCM_{length}"] = _to_series([v.cb if v else None for v in ind], idx)
        df[f"DCU_{length}"] = _to_series([v.ub if v else None for v in ind], idx)

    elif n == "chop":
        length = params.get("length", 14)
        df[f"CHOP_{length}"] = s(CHOP(period=length, input_values=get_ohlcv()))

    # ── Volume ──
    elif n == "obv":
        df["OBV"] = s(OBV(input_values=get_ohlcv()))

    elif n == "sobv":
        length = params.get("length", 5)
        df[f"SOBV_{length}"] = s(SOBV(period=length, input_values=get_ohlcv()))

    elif n == "vwap":
        df["VWAP"] = s(VWAP(input_values=get_ohlcv()))

    elif n == "ad":
        df["AD"] = s(AccuDist(input_values=get_ohlcv()))

    elif n == "force":
        length = params.get("length", 13)
        df[f"Force_{length}"] = s(ForceIndex(period=length, input_values=get_ohlcv()))

    elif n == "emv":
        length = params.get("length", 14)
        df[f"EMV_{length}"] = s(EMV(period=length, volume_div=10000, input_values=get_ohlcv()))

    else:
        raise ValueError(f"Unknown indicator: {name}")


def list_available_indicators() -> list[str]:
    """List all available indicator names."""
    return sorted(INDICATOR_REGISTRY.keys())


def get_indicator_metadata() -> dict[str, dict[str, Any]]:
    """Return the full indicator registry with metadata for each indicator."""
    return INDICATOR_REGISTRY


def get_indicator_columns(df: pd.DataFrame) -> list[str]:
    """Get list of indicator columns (non-OHLCV columns)."""
    ohlcv_cols = {"open", "high", "low", "close", "volume", "quote_volume", "count", 
                  "taker_buy_volume", "taker_buy_quote_volume"}
    return [col for col in df.columns if col.lower() not in ohlcv_cols]


def describe_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Get summary statistics for indicator columns."""
    indicator_cols = get_indicator_columns(df)
    if not indicator_cols:
        return pd.DataFrame()
    return df[indicator_cols].describe()
