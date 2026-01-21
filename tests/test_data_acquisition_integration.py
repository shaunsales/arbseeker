#!/usr/bin/env python3
"""
Integration tests for Stage 1: Data Acquisition Pipeline

Tests end-to-end data acquisition, cleaning, and alignment for a 72-hour period.
These tests make real API calls to Yahoo Finance and Aster.

Run with: pytest tests/test_data_acquisition_integration.py -v -s
"""

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from importlib import import_module

data_acq = import_module("1_data_acquisition")


# ============================================
# Test Configuration
# ============================================

def get_recent_trading_window():
    """
    Get a 72-hour window that includes recent trading days.
    Avoids weekends for stock data (TSLA).
    """
    now = datetime.now(timezone.utc)
    
    # Find the most recent weekday (Mon-Fri)
    # Go back to ensure we have complete trading days
    end = now.replace(hour=21, minute=0, second=0, microsecond=0)
    
    # If today is weekend, go back to Friday
    while end.weekday() >= 5:  # 5=Sat, 6=Sun
        end -= timedelta(days=1)
    
    # Go back one more day to ensure data is available
    end -= timedelta(days=1)
    
    # Start 72 hours before, but adjust to include trading days
    start = end - timedelta(hours=72)
    
    return start, end


TEST_START_DATE, TEST_END_DATE = get_recent_trading_window()

# Test data directory (separate from production)
TEST_DATA_DIR = Path(__file__).parent / "fixtures" / "integration_test_data"


# ============================================
# Fixtures
# ============================================

@pytest.fixture(scope="module")
def test_data_dir(tmp_path_factory):
    """Create temporary directory for test data."""
    return tmp_path_factory.mktemp("integration_test_data")


@pytest.fixture(scope="module")
def tsla_data(test_data_dir, monkeypatch_module):
    """Fetch TSLA data for integration test (72 hours)."""
    monkeypatch_module.setattr(data_acq, "DATA_CLEANED_DIR", test_data_dir)
    
    print(f"\n  Fetching TSLA data: {TEST_START_DATE} to {TEST_END_DATE}")
    
    # Fetch Yahoo
    yahoo_df = data_acq.fetch_yahoo_data_paginated("TSLA", TEST_START_DATE, TEST_END_DATE)
    if not yahoo_df.empty:
        yahoo_cleaned = data_acq.clean_ohlcv_data(yahoo_df, "yahoo", "TSLA")
        data_acq.save_to_parquet(yahoo_cleaned, "TSLA", "yahoo")
    
    # Fetch Aster
    aster_df = data_acq.fetch_aster_klines_paginated("TSLAUSDT", TEST_START_DATE, TEST_END_DATE)
    if not aster_df.empty:
        aster_cleaned = data_acq.clean_ohlcv_data(aster_df, "aster", "TSLAUSDT")
        data_acq.save_to_parquet(aster_cleaned, "TSLA", "aster")
    
    # Load and align
    yahoo_loaded = data_acq.load_from_parquet("TSLA", "yahoo")
    aster_loaded = data_acq.load_from_parquet("TSLA", "aster")
    
    yahoo_aligned, aster_aligned = data_acq.align_timestamps(yahoo_loaded, aster_loaded)
    
    return {
        "yahoo_raw": yahoo_df,
        "aster_raw": aster_df,
        "yahoo_aligned": yahoo_aligned,
        "aster_aligned": aster_aligned,
    }


@pytest.fixture(scope="module")
def gold_data(test_data_dir, monkeypatch_module):
    """Fetch GOLD data for integration test (72 hours)."""
    monkeypatch_module.setattr(data_acq, "DATA_CLEANED_DIR", test_data_dir)
    
    print(f"\n  Fetching GOLD data: {TEST_START_DATE} to {TEST_END_DATE}")
    
    # Fetch Yahoo
    yahoo_df = data_acq.fetch_yahoo_data_paginated("GC=F", TEST_START_DATE, TEST_END_DATE)
    if not yahoo_df.empty:
        yahoo_cleaned = data_acq.clean_ohlcv_data(yahoo_df, "yahoo", "GC=F")
        data_acq.save_to_parquet(yahoo_cleaned, "GOLD", "yahoo")
    
    # Fetch Aster
    aster_df = data_acq.fetch_aster_klines_paginated("XAUUSDT", TEST_START_DATE, TEST_END_DATE)
    if not aster_df.empty:
        aster_cleaned = data_acq.clean_ohlcv_data(aster_df, "aster", "XAUUSDT")
        data_acq.save_to_parquet(aster_cleaned, "GOLD", "aster")
    
    # Load and align
    yahoo_loaded = data_acq.load_from_parquet("GOLD", "yahoo")
    aster_loaded = data_acq.load_from_parquet("GOLD", "aster")
    
    yahoo_aligned, aster_aligned = data_acq.align_timestamps(yahoo_loaded, aster_loaded)
    
    return {
        "yahoo_raw": yahoo_df,
        "aster_raw": aster_df,
        "yahoo_aligned": yahoo_aligned,
        "aster_aligned": aster_aligned,
    }


@pytest.fixture(scope="module")
def monkeypatch_module():
    """Module-scoped monkeypatch fixture."""
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


# ============================================
# Integration Tests: Yahoo Finance
# ============================================

class TestYahooFetchPaginated:
    """Integration tests for Yahoo Finance paginated fetch."""
    
    def test_fetches_tsla_data(self, tsla_data):
        """Should fetch TSLA minute data from Yahoo."""
        df = tsla_data["yahoo_raw"]
        
        assert not df.empty, "Yahoo should return TSLA data"
        assert len(df) > 0, "Should have records"
        print(f"    Yahoo TSLA: {len(df)} records")
    
    def test_fetches_gold_data(self, gold_data):
        """Should fetch Gold futures minute data from Yahoo."""
        df = gold_data["yahoo_raw"]
        
        assert not df.empty, "Yahoo should return Gold data"
        assert len(df) > 0, "Should have records"
        print(f"    Yahoo GOLD: {len(df)} records")
    
    def test_yahoo_timestamps_are_utc(self, tsla_data):
        """Yahoo timestamps should be UTC normalized."""
        df = tsla_data["yahoo_raw"]
        
        if not df.empty:
            assert df.index.tz is not None, "Index should have timezone"
            assert str(df.index.tz) == "UTC", "Timezone should be UTC"


# ============================================
# Integration Tests: Aster
# ============================================

class TestAsterFetchPaginated:
    """Integration tests for Aster paginated fetch."""
    
    def test_fetches_tsla_data(self, tsla_data):
        """Should fetch TSLAUSDT minute data from Aster."""
        df = tsla_data["aster_raw"]
        
        assert not df.empty, "Aster should return TSLA data"
        assert len(df) > 0, "Should have records"
        print(f"    Aster TSLA: {len(df)} records")
    
    def test_fetches_gold_data(self, gold_data):
        """Should fetch XAUUSDT minute data from Aster."""
        df = gold_data["aster_raw"]
        
        assert not df.empty, "Aster should return Gold data"
        assert len(df) > 0, "Should have records"
        print(f"    Aster GOLD: {len(df)} records")
    
    def test_aster_timestamps_are_utc(self, tsla_data):
        """Aster timestamps should be UTC normalized."""
        df = tsla_data["aster_raw"]
        
        if not df.empty:
            assert df.index.tz is not None, "Index should have timezone"
            assert str(df.index.tz) == "UTC", "Timezone should be UTC"
    
    def test_aster_pagination_fetches_full_range(self, tsla_data):
        """Aster pagination should fetch data covering the full 72-hour range."""
        df = tsla_data["aster_raw"]
        
        if not df.empty:
            # Should have close to 72 hours of data (4320 minutes)
            # Allow some tolerance for API limitations
            expected_min = 4000  # At least ~67 hours
            assert len(df) >= expected_min, f"Expected at least {expected_min} records, got {len(df)}"


# ============================================
# Integration Tests: Data Alignment
# ============================================

class TestDataAlignment:
    """Integration tests for timestamp alignment between sources."""
    
    def test_tsla_aligned_counts_match(self, tsla_data):
        """Aligned TSLA data should have equal record counts."""
        yahoo = tsla_data["yahoo_aligned"]
        aster = tsla_data["aster_aligned"]
        
        assert len(yahoo) == len(aster), f"Record counts should match: Yahoo={len(yahoo)}, Aster={len(aster)}"
        print(f"    TSLA aligned: {len(yahoo)} bars")
    
    def test_gold_aligned_counts_match(self, gold_data):
        """Aligned GOLD data should have equal record counts."""
        yahoo = gold_data["yahoo_aligned"]
        aster = gold_data["aster_aligned"]
        
        assert len(yahoo) == len(aster), f"Record counts should match: Yahoo={len(yahoo)}, Aster={len(aster)}"
        print(f"    GOLD aligned: {len(yahoo)} bars")
    
    def test_tsla_timestamps_identical(self, tsla_data):
        """Aligned TSLA timestamps should be identical between sources."""
        yahoo = tsla_data["yahoo_aligned"]
        aster = tsla_data["aster_aligned"]
        
        if not yahoo.empty and not aster.empty:
            assert yahoo.index.equals(aster.index), "Timestamps should be identical"
    
    def test_gold_timestamps_identical(self, gold_data):
        """Aligned GOLD timestamps should be identical between sources."""
        yahoo = gold_data["yahoo_aligned"]
        aster = gold_data["aster_aligned"]
        
        if not yahoo.empty and not aster.empty:
            assert yahoo.index.equals(aster.index), "Timestamps should be identical"
    
    def test_aligned_data_has_valid_prices(self, tsla_data):
        """Aligned data should have valid price values."""
        yahoo = tsla_data["yahoo_aligned"]
        aster = tsla_data["aster_aligned"]
        
        if not yahoo.empty:
            # No NaN in price columns
            assert not yahoo["close"].isna().any(), "Yahoo should have no NaN closes"
            assert not aster["close"].isna().any(), "Aster should have no NaN closes"
            
            # Prices should be positive
            assert (yahoo["close"] > 0).all(), "Yahoo prices should be positive"
            assert (aster["close"] > 0).all(), "Aster prices should be positive"
    
    def test_aligned_data_within_date_range(self, tsla_data):
        """Aligned data should be within the requested date range."""
        yahoo = tsla_data["yahoo_aligned"]
        
        if not yahoo.empty:
            assert yahoo.index.min() >= TEST_START_DATE, "Data should start after START_DATE"
            assert yahoo.index.max() <= TEST_END_DATE, "Data should end before END_DATE"


# ============================================
# Integration Tests: Data Schema
# ============================================

class TestCleanedDataSchema:
    """Integration tests for cleaned data schema compliance."""
    
    def test_yahoo_has_required_columns(self, tsla_data):
        """Cleaned Yahoo data should have all required columns."""
        df = tsla_data["yahoo_aligned"]
        
        if not df.empty:
            for col in data_acq.SCHEMA_COLUMNS:
                assert col in df.columns, f"Missing column: {col}"
    
    def test_aster_has_required_columns(self, tsla_data):
        """Cleaned Aster data should have all required columns."""
        df = tsla_data["aster_aligned"]
        
        if not df.empty:
            for col in data_acq.SCHEMA_COLUMNS:
                assert col in df.columns, f"Missing column: {col}"
    
    def test_mid_price_calculated_correctly(self, tsla_data):
        """Mid price should equal (high + low) / 2."""
        df = tsla_data["yahoo_aligned"]
        
        if not df.empty:
            expected_mid = (df["high"] + df["low"]) / 2
            pd.testing.assert_series_equal(
                df["mid"], expected_mid,
                check_names=False,
                rtol=1e-10
            )
    
    def test_source_metadata_correct(self, tsla_data):
        """Source metadata should be correctly set."""
        yahoo = tsla_data["yahoo_aligned"]
        aster = tsla_data["aster_aligned"]
        
        if not yahoo.empty:
            assert (yahoo["source"] == "yahoo").all(), "Yahoo source should be 'yahoo'"
        if not aster.empty:
            assert (aster["source"] == "aster").all(), "Aster source should be 'aster'"


# ============================================
# Integration Tests: Parquet Persistence
# ============================================

class TestParquetPersistence:
    """Integration tests for parquet save/load operations."""
    
    def test_save_load_roundtrip_preserves_data(self, tsla_data, test_data_dir, monkeypatch_module):
        """Data should survive parquet save/load roundtrip."""
        monkeypatch_module.setattr(data_acq, "DATA_CLEANED_DIR", test_data_dir)
        
        original = tsla_data["yahoo_aligned"]
        
        if not original.empty:
            # Save
            data_acq.save_to_parquet(original, "TSLA_TEST", "yahoo")
            
            # Load
            loaded = data_acq.load_from_parquet("TSLA_TEST", "yahoo")
            
            # Compare
            pd.testing.assert_frame_equal(original, loaded)
    
    def test_parquet_files_created(self, test_data_dir, monkeypatch_module):
        """Parquet files should be created in the data directory."""
        monkeypatch_module.setattr(data_acq, "DATA_CLEANED_DIR", test_data_dir)
        
        parquet_files = list(test_data_dir.glob("*.parquet"))
        assert len(parquet_files) > 0, "Should have created parquet files"
        print(f"    Created {len(parquet_files)} parquet files")
