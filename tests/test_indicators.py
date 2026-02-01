"""
Tests for the technical indicators module.

Run with: pytest tests/test_indicators.py -v
"""

import pytest
import pandas as pd
import numpy as np

from core.indicators import (
    compute_indicators,
    list_available_indicators,
    get_indicator_columns,
    INDICATOR_PRESETS,
)


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 100
    
    # Generate random walk prices
    returns = np.random.randn(n) * 0.02
    close = 100 * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        "open": close * (1 + np.random.randn(n) * 0.005),
        "high": close * (1 + np.abs(np.random.randn(n)) * 0.01),
        "low": close * (1 - np.abs(np.random.randn(n)) * 0.01),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })
    
    df.index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return df


class TestComputeIndicators:
    """Test indicator computation."""
    
    def test_sma(self, sample_ohlcv):
        """Test SMA calculation."""
        df = compute_indicators(sample_ohlcv, [("sma", {"length": 20})])
        
        assert "SMA_20" in df.columns
        assert df["SMA_20"].iloc[:19].isna().all()  # First 19 values should be NaN
        assert not df["SMA_20"].iloc[19:].isna().any()  # Rest should have values
        
        # Verify calculation
        expected = sample_ohlcv["close"].rolling(20).mean()
        pd.testing.assert_series_equal(df["SMA_20"], expected, check_names=False)
    
    def test_rsi(self, sample_ohlcv):
        """Test RSI calculation."""
        df = compute_indicators(sample_ohlcv, [("rsi", {"length": 14})])
        
        assert "RSI_14" in df.columns
        # RSI should be between 0 and 100
        valid_rsi = df["RSI_14"].dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
    
    def test_macd(self, sample_ohlcv):
        """Test MACD calculation."""
        df = compute_indicators(sample_ohlcv, [("macd", {"fast": 12, "slow": 26, "signal": 9})])
        
        assert "MACD_12_26_9" in df.columns
        assert "MACDh_12_26_9" in df.columns  # Histogram
        assert "MACDs_12_26_9" in df.columns  # Signal line
    
    def test_bbands(self, sample_ohlcv):
        """Test Bollinger Bands calculation."""
        df = compute_indicators(sample_ohlcv, [("bbands", {"length": 20, "std": 2})])
        
        assert "BBL_20_2" in df.columns  # Lower band
        assert "BBM_20_2" in df.columns  # Middle band
        assert "BBU_20_2" in df.columns  # Upper band
        
        # Upper > Middle > Lower
        valid_idx = df["BBM_20_2"].notna()
        assert (df.loc[valid_idx, "BBU_20_2"] >= df.loc[valid_idx, "BBM_20_2"]).all()
        assert (df.loc[valid_idx, "BBM_20_2"] >= df.loc[valid_idx, "BBL_20_2"]).all()
    
    def test_atr(self, sample_ohlcv):
        """Test ATR calculation."""
        df = compute_indicators(sample_ohlcv, [("atr", {"length": 14})])
        
        assert "ATR_14" in df.columns
        # ATR should be positive
        assert (df["ATR_14"].dropna() > 0).all()
    
    def test_multiple_indicators(self, sample_ohlcv):
        """Test computing multiple indicators at once."""
        df = compute_indicators(sample_ohlcv, [
            ("sma", {"length": 10}),
            ("sma", {"length": 20}),
            ("rsi", {"length": 14}),
            ("macd", {"fast": 12, "slow": 26, "signal": 9}),
        ])
        
        expected_cols = ["SMA_10", "SMA_20", "RSI_14", "MACD_12_26_9"]
        for col in expected_cols:
            assert col in df.columns


class TestIndicatorPresets:
    """Test indicator preset configurations."""
    
    def test_minimal_preset(self, sample_ohlcv):
        """Test minimal preset."""
        df = compute_indicators(sample_ohlcv, INDICATOR_PRESETS["minimal"])
        
        assert "SMA_20" in df.columns
        assert "SMA_50" in df.columns
        assert "RSI_14" in df.columns
    
    def test_trend_preset(self, sample_ohlcv):
        """Test trend preset."""
        df = compute_indicators(sample_ohlcv, INDICATOR_PRESETS["trend"])
        
        assert "MACD_12_26_9" in df.columns
        assert "ADX_14" in df.columns
    
    def test_all_presets_work(self, sample_ohlcv):
        """Test that all presets compute without error."""
        for preset_name, indicators in INDICATOR_PRESETS.items():
            df = compute_indicators(sample_ohlcv, indicators)
            indicator_cols = get_indicator_columns(df)
            assert len(indicator_cols) > 0, f"Preset {preset_name} produced no indicators"


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_list_available_indicators(self):
        """Test listing available indicators."""
        indicators = list_available_indicators()
        
        assert "sma" in indicators
        assert "rsi" in indicators
        assert "macd" in indicators
        assert "bbands" in indicators
    
    def test_get_indicator_columns(self, sample_ohlcv):
        """Test getting indicator columns."""
        df = compute_indicators(sample_ohlcv, [("sma", {"length": 20})])
        
        indicator_cols = get_indicator_columns(df)
        assert "SMA_20" in indicator_cols
        assert "close" not in indicator_cols
        assert "volume" not in indicator_cols


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
