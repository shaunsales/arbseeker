"""
Technical indicators module using pandas-ta.

Pre-computes indicators before backtesting for efficient array-based access.

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

from typing import Optional
import pandas as pd
import pandas_ta as ta


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
        ("kc", {"length": 20}),
    ],
    "volume": [
        ("obv", {}),
        ("vwap", {}),
        ("mfi", {"length": 14}),
    ],
    "full": [
        # Trend
        ("sma", {"length": 10}),
        ("sma", {"length": 20}),
        ("sma", {"length": 50}),
        ("sma", {"length": 200}),
        ("ema", {"length": 12}),
        ("ema", {"length": 26}),
        ("macd", {"fast": 12, "slow": 26, "signal": 9}),
        ("adx", {"length": 14}),
        # Momentum
        ("rsi", {"length": 14}),
        ("stoch", {"k": 14, "d": 3}),
        ("cci", {"length": 20}),
        # Volatility
        ("bbands", {"length": 20, "std": 2}),
        ("atr", {"length": 14}),
    ],
}


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
    """Add a single indicator to the DataFrame."""
    
    # Use pandas-ta's strategy approach for cleaner code
    # But we'll call individual functions for more control
    
    name_lower = name.lower()
    
    # Trend indicators
    if name_lower == "sma":
        length = params.get("length", 20)
        df[f"SMA_{length}"] = ta.sma(df["close"], length=length)
        
    elif name_lower == "ema":
        length = params.get("length", 20)
        df[f"EMA_{length}"] = ta.ema(df["close"], length=length)
        
    elif name_lower == "wma":
        length = params.get("length", 20)
        df[f"WMA_{length}"] = ta.wma(df["close"], length=length)
        
    elif name_lower == "macd":
        fast = params.get("fast", 12)
        slow = params.get("slow", 26)
        signal = params.get("signal", 9)
        macd = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
        if macd is not None:
            df[f"MACD_{fast}_{slow}_{signal}"] = macd.iloc[:, 0]
            df[f"MACDh_{fast}_{slow}_{signal}"] = macd.iloc[:, 1]
            df[f"MACDs_{fast}_{slow}_{signal}"] = macd.iloc[:, 2]
            
    elif name_lower == "adx":
        length = params.get("length", 14)
        adx = ta.adx(df["high"], df["low"], df["close"], length=length)
        if adx is not None:
            df[f"ADX_{length}"] = adx.iloc[:, 0]
            df[f"DMP_{length}"] = adx.iloc[:, 1]
            df[f"DMN_{length}"] = adx.iloc[:, 2]
            
    # Momentum indicators
    elif name_lower == "rsi":
        length = params.get("length", 14)
        df[f"RSI_{length}"] = ta.rsi(df["close"], length=length)
        
    elif name_lower == "stoch":
        k = params.get("k", 14)
        d = params.get("d", 3)
        stoch = ta.stoch(df["high"], df["low"], df["close"], k=k, d=d)
        if stoch is not None:
            df[f"STOCHk_{k}_{d}"] = stoch.iloc[:, 0]
            df[f"STOCHd_{k}_{d}"] = stoch.iloc[:, 1]
            
    elif name_lower == "cci":
        length = params.get("length", 20)
        df[f"CCI_{length}"] = ta.cci(df["high"], df["low"], df["close"], length=length)
        
    elif name_lower == "willr":
        length = params.get("length", 14)
        df[f"WILLR_{length}"] = ta.willr(df["high"], df["low"], df["close"], length=length)
        
    elif name_lower == "roc":
        length = params.get("length", 10)
        df[f"ROC_{length}"] = ta.roc(df["close"], length=length)
        
    elif name_lower == "mom":
        length = params.get("length", 10)
        df[f"MOM_{length}"] = ta.mom(df["close"], length=length)
        
    # Volatility indicators
    elif name_lower == "bbands":
        length = params.get("length", 20)
        std = params.get("std", 2)
        bbands = ta.bbands(df["close"], length=length, std=std)
        if bbands is not None:
            df[f"BBL_{length}_{std}"] = bbands.iloc[:, 0]
            df[f"BBM_{length}_{std}"] = bbands.iloc[:, 1]
            df[f"BBU_{length}_{std}"] = bbands.iloc[:, 2]
            df[f"BBB_{length}_{std}"] = bbands.iloc[:, 3]  # Bandwidth
            df[f"BBP_{length}_{std}"] = bbands.iloc[:, 4]  # Percent
            
    elif name_lower == "atr":
        length = params.get("length", 14)
        df[f"ATR_{length}"] = ta.atr(df["high"], df["low"], df["close"], length=length)
        
    elif name_lower == "natr":
        length = params.get("length", 14)
        df[f"NATR_{length}"] = ta.natr(df["high"], df["low"], df["close"], length=length)
        
    elif name_lower == "kc":
        length = params.get("length", 20)
        kc = ta.kc(df["high"], df["low"], df["close"], length=length)
        if kc is not None:
            df[f"KCL_{length}"] = kc.iloc[:, 0]
            df[f"KCM_{length}"] = kc.iloc[:, 1]
            df[f"KCU_{length}"] = kc.iloc[:, 2]
            
    # Volume indicators
    elif name_lower == "obv":
        df["OBV"] = ta.obv(df["close"], df["volume"])
        
    elif name_lower == "vwap":
        df["VWAP"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
        
    elif name_lower == "mfi":
        length = params.get("length", 14)
        df[f"MFI_{length}"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=length)
        
    elif name_lower == "ad":
        df["AD"] = ta.ad(df["high"], df["low"], df["close"], df["volume"])
        
    # Support/Resistance
    elif name_lower == "pivot":
        pivot = ta.pivot(df["high"], df["low"], df["close"])
        if pivot is not None:
            for col in pivot.columns:
                df[col] = pivot[col]
                
    else:
        # Try using pandas-ta's generic approach
        try:
            result = df.ta(kind=name, **params, append=False)
            if result is not None:
                if isinstance(result, pd.DataFrame):
                    for col in result.columns:
                        df[col] = result[col]
                else:
                    df[f"{name.upper()}"] = result
        except Exception as e:
            raise ValueError(f"Unknown indicator: {name}") from e


def list_available_indicators() -> list[str]:
    """List all available indicator names."""
    return [
        # Trend
        "sma", "ema", "wma", "macd", "adx",
        # Momentum
        "rsi", "stoch", "cci", "willr", "roc", "mom",
        # Volatility
        "bbands", "atr", "natr", "kc",
        # Volume
        "obv", "vwap", "mfi", "ad",
        # Support/Resistance
        "pivot",
    ]


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
