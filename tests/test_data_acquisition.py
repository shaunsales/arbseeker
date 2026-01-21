#!/usr/bin/env python3
"""
Unit tests for Stage 1: Data Acquisition

Tests cover:
- Data cleaning and normalization functions
- Schema validation
- Parquet save/load operations
- Edge cases (empty data, missing columns)
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from importlib import import_module

# Import the module under test
data_acq = import_module("1_data_acquisition")


# ============================================
# Fixtures
# ============================================

@pytest.fixture
def sample_yahoo_raw():
    """Sample raw Yahoo Finance data."""
    dates = pd.date_range(
        start="2024-01-15 14:30:00",
        periods=100,
        freq="1min",
        tz="UTC"
    )
    return pd.DataFrame({
        "Open": np.random.uniform(100, 110, 100),
        "High": np.random.uniform(110, 115, 100),
        "Low": np.random.uniform(95, 100, 100),
        "Close": np.random.uniform(100, 110, 100),
        "Volume": np.random.randint(1000, 10000, 100),
    }, index=dates)


@pytest.fixture
def sample_aster_raw():
    """Sample raw Aster klines data."""
    dates = pd.date_range(
        start="2024-01-15 14:30:00",
        periods=100,
        freq="1min",
        tz="UTC"
    )
    return pd.DataFrame({
        "Open": np.random.uniform(100, 110, 100).astype(str),
        "High": np.random.uniform(110, 115, 100).astype(str),
        "Low": np.random.uniform(95, 100, 100).astype(str),
        "Close": np.random.uniform(100, 110, 100).astype(str),
        "Volume": np.random.randint(1000, 10000, 100).astype(str),
    }, index=dates)


@pytest.fixture
def sample_hyperliquid_raw():
    """Sample raw Hyperliquid candles data."""
    dates = pd.date_range(
        start="2024-01-15 14:30:00",
        periods=100,
        freq="1min",
        tz="UTC"
    )
    return pd.DataFrame({
        "Open": np.random.uniform(100, 110, 100),
        "High": np.random.uniform(110, 115, 100),
        "Low": np.random.uniform(95, 100, 100),
        "Close": np.random.uniform(100, 110, 100),
        "Volume": np.random.uniform(100, 1000, 100),
    }, index=dates)


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary data directory for tests."""
    cleaned_dir = tmp_path / "data" / "cleaned"
    cleaned_dir.mkdir(parents=True)
    return cleaned_dir


# ============================================
# Unit Tests: clean_ohlcv_data
# ============================================

class TestCleanOhlcvData:
    """Tests for the clean_ohlcv_data function."""
    
    def test_cleans_yahoo_data(self, sample_yahoo_raw):
        """Should clean Yahoo data and add metadata."""
        result = data_acq.clean_ohlcv_data(sample_yahoo_raw, "yahoo", "TSLA")
        
        assert len(result) == 100
        assert list(result.columns) == data_acq.SCHEMA_COLUMNS
        assert (result["source"] == "yahoo").all()
        assert (result["symbol"] == "TSLA").all()
        assert "mid" in result.columns
    
    def test_cleans_aster_data_with_string_prices(self, sample_aster_raw):
        """Should handle string price values from Aster."""
        result = data_acq.clean_ohlcv_data(sample_aster_raw, "aster", "TSLAUSDT")
        
        assert len(result) == 100
        assert result["open"].dtype == np.float64
        assert (result["source"] == "aster").all()
    
    def test_calculates_mid_price(self, sample_yahoo_raw):
        """Should calculate mid = (high + low) / 2."""
        result = data_acq.clean_ohlcv_data(sample_yahoo_raw, "yahoo", "TEST")
        
        expected_mid = (sample_yahoo_raw["High"] + sample_yahoo_raw["Low"]) / 2
        np.testing.assert_array_almost_equal(result["mid"].values, expected_mid.values)
    
    def test_handles_empty_dataframe(self):
        """Should return empty DataFrame with correct schema for empty input."""
        empty_df = pd.DataFrame()
        result = data_acq.clean_ohlcv_data(empty_df, "yahoo", "TEST")
        
        assert result.empty
        assert list(result.columns) == data_acq.SCHEMA_COLUMNS
    
    def test_handles_missing_required_columns(self):
        """Should return empty DataFrame if required columns missing."""
        df = pd.DataFrame({"Open": [1, 2, 3], "Close": [1, 2, 3]})
        result = data_acq.clean_ohlcv_data(df, "yahoo", "TEST")
        
        assert result.empty
    
    def test_handles_missing_volume(self, sample_yahoo_raw):
        """Should handle missing volume column."""
        df = sample_yahoo_raw.drop(columns=["Volume"])
        result = data_acq.clean_ohlcv_data(df, "yahoo", "TEST")
        
        assert "volume" in result.columns
        assert (result["volume"] == 0.0).all()
    
    def test_drops_rows_with_nan_prices(self, sample_yahoo_raw):
        """Should drop rows with NaN in price columns."""
        df = sample_yahoo_raw.copy()
        df.iloc[0, 0] = np.nan  # Set first Open to NaN
        df.iloc[5, 3] = np.nan  # Set sixth Close to NaN
        
        result = data_acq.clean_ohlcv_data(df, "yahoo", "TEST")
        
        assert len(result) == 98
    
    def test_index_named_timestamp(self, sample_yahoo_raw):
        """Should ensure index is named 'timestamp'."""
        result = data_acq.clean_ohlcv_data(sample_yahoo_raw, "yahoo", "TEST")
        
        assert result.index.name == "timestamp"


# ============================================
# Unit Tests: Parquet Operations
# ============================================

class TestParquetOperations:
    """Tests for save_to_parquet and load_from_parquet functions."""
    
    def test_save_and_load_roundtrip(self, sample_yahoo_raw, tmp_data_dir, monkeypatch):
        """Should save and load data without loss."""
        # Patch the DATA_CLEANED_DIR
        monkeypatch.setattr(data_acq, "DATA_CLEANED_DIR", tmp_data_dir)
        
        cleaned = data_acq.clean_ohlcv_data(sample_yahoo_raw, "yahoo", "TSLA")
        saved_path = data_acq.save_to_parquet(cleaned, "TSLA", "yahoo")
        
        assert saved_path is not None
        assert saved_path.exists()
        
        loaded = data_acq.load_from_parquet("TSLA", "yahoo")
        
        pd.testing.assert_frame_equal(cleaned, loaded)
    
    def test_save_empty_returns_none(self, tmp_data_dir, monkeypatch):
        """Should return None when saving empty DataFrame."""
        monkeypatch.setattr(data_acq, "DATA_CLEANED_DIR", tmp_data_dir)
        
        empty_df = pd.DataFrame()
        result = data_acq.save_to_parquet(empty_df, "TSLA", "yahoo")
        
        assert result is None
    
    def test_load_nonexistent_returns_empty(self, tmp_data_dir, monkeypatch):
        """Should return empty DataFrame for non-existent file."""
        monkeypatch.setattr(data_acq, "DATA_CLEANED_DIR", tmp_data_dir)
        
        result = data_acq.load_from_parquet("NONEXISTENT", "yahoo")
        
        assert result.empty
    
    def test_file_naming_convention(self, sample_yahoo_raw, tmp_data_dir, monkeypatch):
        """Should follow {asset}_{source}.parquet naming."""
        monkeypatch.setattr(data_acq, "DATA_CLEANED_DIR", tmp_data_dir)
        
        cleaned = data_acq.clean_ohlcv_data(sample_yahoo_raw, "yahoo", "TSLA")
        saved_path = data_acq.save_to_parquet(cleaned, "TSLA", "yahoo")
        
        assert saved_path.name == "tsla_yahoo.parquet"


# ============================================
# Unit Tests: Configuration
# ============================================

class TestConfiguration:
    """Tests for module configuration."""
    
    def test_assets_list_format(self):
        """ASSETS should have correct format: (name, yahoo_symbol, aster_symbol, market_open_hour)."""
        from importlib import import_module
        data_acq = import_module("1_data_acquisition")
        
        for asset in data_acq.ASSETS:
            assert len(asset) == 4
            name, yahoo_sym, aster_sym, market_hour = asset
            assert isinstance(name, str)
            assert isinstance(yahoo_sym, str)
            assert isinstance(aster_sym, str)
            assert isinstance(market_hour, int)
    
    def test_schema_columns_complete(self):
        """SCHEMA_COLUMNS should contain all required columns."""
        required = ["open", "high", "low", "close", "mid", "volume", "source", "symbol"]
        assert data_acq.SCHEMA_COLUMNS == required


# ============================================
# Integration Tests (with mocked APIs)
# ============================================

class TestAcquireAssetData:
    """Integration tests for acquire_asset_data function."""
    
    def test_acquires_data_from_both_sources(
        self, sample_yahoo_raw, tmp_data_dir, monkeypatch
    ):
        """Should acquire data from Yahoo and Aster."""
        from datetime import datetime, timezone
        
        monkeypatch.setattr(data_acq, "DATA_CLEANED_DIR", tmp_data_dir)
        monkeypatch.setattr(data_acq, "fetch_yahoo_data_paginated", lambda *args, **kwargs: sample_yahoo_raw)
        monkeypatch.setattr(data_acq, "fetch_aster_klines_paginated", lambda *args, **kwargs: sample_yahoo_raw)
        
        start = datetime(2024, 1, 15, tzinfo=timezone.utc)
        end = datetime(2024, 1, 16, tzinfo=timezone.utc)
        
        results = data_acq.acquire_asset_data(
            "TEST", "TEST", "TESTUSDT", start, end
        )
        
        assert results["yahoo"] is not None
        assert results["aster"] is not None
    
    def test_handles_empty_yahoo_data(self, tmp_data_dir, monkeypatch, capsys):
        """Should handle case when Yahoo returns no data."""
        from datetime import datetime, timezone
        
        monkeypatch.setattr(data_acq, "DATA_CLEANED_DIR", tmp_data_dir)
        monkeypatch.setattr(data_acq, "fetch_yahoo_data_paginated", lambda *args, **kwargs: pd.DataFrame())
        monkeypatch.setattr(data_acq, "fetch_aster_klines_paginated", lambda *args, **kwargs: pd.DataFrame())
        
        start = datetime(2024, 1, 15, tzinfo=timezone.utc)
        end = datetime(2024, 1, 16, tzinfo=timezone.utc)
        
        results = data_acq.acquire_asset_data(
            "TEST", "TEST", "TESTUSDT", start, end
        )
        
        captured = capsys.readouterr()
        assert "No data" in captured.out
