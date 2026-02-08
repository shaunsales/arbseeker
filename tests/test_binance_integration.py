"""
Integration tests for Binance Vision data download.

These tests make real network requests to Binance Vision.
Run with: pytest tests/test_binance_integration.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import shutil

from core.data.binance import (
    download_month,
    get_monthly_url,
    KLINE_COLUMNS,
    INTERVALS,
)
from core.data.storage import save_monthly, load_ohlcv, list_available_periods, get_data_path
from core.data.validator import validate_ohlcv, fill_gaps, get_data_summary


class TestBinanceDataFormat:
    """Test that downloaded data matches expected Binance format."""
    
    def test_download_single_month_btcusdt_1h(self):
        """
        Download 1 month of BTCUSDT 1h data and verify format.
        
        Expected Binance Vision columns:
        open_time, open, high, low, close, volume, close_time, 
        quote_volume, count, taker_buy_volume, taker_buy_quote_volume, ignore
        """
        # Download January 2024
        df = download_month(
            symbol="BTCUSDT",
            interval="1h",
            year=2024,
            month=1,
            market="futures",
        )
        
        assert df is not None, "Failed to download data"
        assert len(df) > 0, "Downloaded data is empty"
        
        # Verify we have a DatetimeIndex
        assert isinstance(df.index, pd.DatetimeIndex), "Index should be DatetimeIndex"
        assert df.index.tz is not None, "Index should be timezone-aware (UTC)"
        
        # Verify expected columns exist
        expected_columns = [
            "open", "high", "low", "close", "volume", 
            "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"
        ]
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"
        
        # Verify data types
        assert df["open"].dtype == np.float64, "open should be float64"
        assert df["high"].dtype == np.float64, "high should be float64"
        assert df["low"].dtype == np.float64, "low should be float64"
        assert df["close"].dtype == np.float64, "close should be float64"
        assert df["volume"].dtype == np.float64, "volume should be float64"
        assert df["quote_volume"].dtype == np.float64, "quote_volume should be float64"
        assert df["count"].dtype in [np.int64, np.int32], "count should be integer"
        assert df["taker_buy_volume"].dtype == np.float64, "taker_buy_volume should be float64"
        assert df["taker_buy_quote_volume"].dtype == np.float64, "taker_buy_quote_volume should be float64"
        
        # Verify date range (January 2024)
        assert df.index.min().year == 2024, "Start year should be 2024"
        assert df.index.min().month == 1, "Start month should be January"
        assert df.index.max().year == 2024, "End year should be 2024"
        assert df.index.max().month == 1, "End month should be January"
        
        # Verify expected bar count (31 days * 24 hours = 744 bars for January)
        expected_bars = 31 * 24
        assert len(df) == expected_bars, f"Expected {expected_bars} bars for January, got {len(df)}"
        
        # Verify OHLC relationships (high >= low, etc.)
        assert (df["high"] >= df["low"]).all(), "High should be >= Low"
        assert (df["high"] >= df["open"]).all(), "High should be >= Open"
        assert (df["high"] >= df["close"]).all(), "High should be >= Close"
        assert (df["low"] <= df["open"]).all(), "Low should be <= Open"
        assert (df["low"] <= df["close"]).all(), "Low should be <= Close"
        
        # Verify positive values
        assert (df["open"] > 0).all(), "Open prices should be positive"
        assert (df["volume"] >= 0).all(), "Volume should be non-negative"
        assert (df["count"] >= 0).all(), "Trade count should be non-negative"
        
        # Verify timestamps are sequential (1 hour apart)
        time_diffs = df.index.to_series().diff().dropna()
        expected_diff = pd.Timedelta(hours=1)
        assert (time_diffs == expected_diff).all(), "Timestamps should be 1 hour apart"
        
        print(f"\n✅ Downloaded {len(df)} bars for BTCUSDT 1h January 2024")
        print(f"   Date range: {df.index.min()} to {df.index.max()}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Price range: ${df['low'].min():,.2f} - ${df['high'].max():,.2f}")
        print(f"   Total volume: {df['volume'].sum():,.2f} BTC")
        print(f"   Total trades: {df['count'].sum():,}")
    
    def test_url_format(self):
        """Verify URL generation matches Binance Vision format."""
        url = get_monthly_url("BTCUSDT", "1h", 2024, 1, "futures")
        expected = "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01.zip"
        assert url == expected, f"URL mismatch: {url} != {expected}"
        
        # Spot market
        url_spot = get_monthly_url("BTCUSDT", "1h", 2024, 1, "spot")
        expected_spot = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01.zip"
        assert url_spot == expected_spot, f"Spot URL mismatch"
    
    def test_column_names_match_binance(self):
        """Verify our column names match official Binance format."""
        # Official Binance Vision column names (from CSV header)
        binance_columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "count", "taker_buy_volume",
            "taker_buy_quote_volume", "ignore"
        ]
        assert KLINE_COLUMNS == binance_columns, f"Column mismatch: {KLINE_COLUMNS} != {binance_columns}"


class TestDataValidation:
    """Test data validation utilities."""
    
    def test_validate_downloaded_data(self):
        """Validate downloaded data for gaps and quality."""
        df = download_month("BTCUSDT", "1h", 2024, 1, "futures")
        assert df is not None
        
        report = validate_ohlcv(df, "1h")
        
        print(f"\n{report}")
        
        # January should have 100% coverage (no gaps expected)
        assert report.coverage_pct >= 99.0, f"Coverage too low: {report.coverage_pct}%"
        assert report.null_count == 0, f"Found {report.null_count} null values"
        assert report.is_valid, "Data should pass validation"
    
    def test_data_summary(self):
        """Test data summary generation."""
        df = download_month("BTCUSDT", "1h", 2024, 1, "futures")
        assert df is not None
        
        summary = get_data_summary(df)
        
        assert "bars" in summary
        assert "start" in summary
        assert "end" in summary
        assert "open_price" in summary
        assert "close_price" in summary
        assert "high_price" in summary
        assert "low_price" in summary
        assert "total_volume" in summary
        
        assert summary["bars"] == 744, f"Expected 744 bars, got {summary['bars']}"
        assert summary["open_price"] > 0
        assert summary["total_volume"] > 0
        
        print(f"\nData Summary: {summary}")


class TestStorageRoundtrip:
    """Test save and load functionality."""
    
    def test_save_and_load_parquet(self):
        """Test saving to parquet and loading back."""
        # Download data
        df = download_month("BTCUSDT", "1h", 2024, 1, "futures")
        assert df is not None
        
        # Use temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Monkey-patch DATA_DIR temporarily
            import core.data.storage as storage_module
            original_data_dir = storage_module.DATA_DIR
            storage_module.DATA_DIR = Path(tmpdir)
            
            try:
                # Save
                path = save_monthly(df, "binance", "futures", "BTCUSDT", "1h", 2024, 1)
                assert path.exists(), f"File not created: {path}"
                
                # Verify path structure
                expected_path = Path(tmpdir) / "binance" / "futures" / "BTCUSDT" / "1h" / "2024-01.parquet"
                assert path == expected_path, f"Path mismatch: {path} != {expected_path}"
                
                # Load back
                df_loaded = load_ohlcv("binance", "futures", "BTCUSDT", "1h", periods=["2024-01"])
                
                # Verify data matches
                assert len(df_loaded) == len(df), "Row count mismatch"
                assert list(df_loaded.columns) == list(df.columns), "Column mismatch"
                
                # Verify values match
                pd.testing.assert_frame_equal(df, df_loaded)
                
                # Verify list_available_periods
                periods = list_available_periods("binance", "futures", "BTCUSDT", "1h")
                assert periods == ["2024-01"], f"Expected ['2024-01'], got {periods}"
                
                print(f"\n✅ Roundtrip test passed")
                print(f"   Saved to: {path}")
                print(f"   File size: {path.stat().st_size / 1024:.1f} KB")
                
            finally:
                storage_module.DATA_DIR = original_data_dir


class TestDifferentIntervals:
    """Test downloading different intervals."""
    
    @pytest.mark.parametrize("interval,expected_bars", [
        ("1h", 744),   # 31 days * 24 hours
        ("4h", 186),   # 31 days * 6 bars/day
        ("1d", 31),    # 31 days
    ])
    def test_interval_bar_counts(self, interval, expected_bars):
        """Verify bar counts for different intervals."""
        df = download_month("BTCUSDT", interval, 2024, 1, "futures")
        
        if df is None:
            pytest.skip(f"No data available for {interval}")
        
        assert len(df) == expected_bars, f"Expected {expected_bars} bars for {interval}, got {len(df)}"
        print(f"\n✅ {interval}: {len(df)} bars (expected {expected_bars})")


if __name__ == "__main__":
    # Run quick smoke test
    print("=" * 60)
    print("Binance Vision Integration Test")
    print("=" * 60)
    
    test = TestBinanceDataFormat()
    test.test_download_single_month_btcusdt_1h()
    test.test_url_format()
    test.test_column_names_match_binance()
    
    test2 = TestDataValidation()
    test2.test_validate_downloaded_data()
    test2.test_data_summary()
    
    test3 = TestStorageRoundtrip()
    test3.test_save_and_load_parquet()
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
