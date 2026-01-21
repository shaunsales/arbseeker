#!/usr/bin/env python3
"""
Unit tests for Stage 2: Basis Analysis

Tests basis calculation functions, market hours detection, and statistics.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from importlib import import_module

# Import the module
basis_analysis = import_module("2_basis_analysis")


# ============================================
# Fixtures
# ============================================

@pytest.fixture
def sample_tradfi_df():
    """Create sample TradFi price data."""
    timestamps = pd.date_range(
        start="2026-01-15 14:30:00",
        periods=10,
        freq="1min",
        tz="UTC"
    )
    
    return pd.DataFrame({
        "open": [100.0] * 10,
        "high": [101.0] * 10,
        "low": [99.0] * 10,
        "close": [100.5] * 10,
        "mid": [100.0] * 10,
        "volume": [1000.0] * 10,
        "source": ["yahoo"] * 10,
        "symbol": ["TEST"] * 10,
    }, index=timestamps)


@pytest.fixture
def sample_defi_df():
    """Create sample DeFi price data with slight premium."""
    timestamps = pd.date_range(
        start="2026-01-15 14:30:00",
        periods=10,
        freq="1min",
        tz="UTC"
    )
    
    # DeFi prices slightly higher (10 bps premium on average)
    return pd.DataFrame({
        "open": [100.1] * 10,
        "high": [101.1] * 10,
        "low": [99.1] * 10,
        "close": [100.6] * 10,
        "mid": [100.1] * 10,  # 0.1% = 10 bps premium
        "volume": [500.0] * 10,
        "source": ["aster"] * 10,
        "symbol": ["TESTUSDT"] * 10,
    }, index=timestamps)


# ============================================
# Test: Basis Calculation
# ============================================

class TestCalculateBasis:
    """Tests for calculate_basis function."""
    
    def test_positive_basis(self):
        """DeFi > TradFi should give positive basis."""
        tradfi = 100.0
        defi = 100.1  # 0.1% higher
        
        basis_abs, basis_bps = basis_analysis.calculate_basis(tradfi, defi)
        
        assert basis_abs == pytest.approx(0.1, rel=0.01)
        assert basis_bps == pytest.approx(10.0, rel=0.01)  # 10 bps
    
    def test_negative_basis(self):
        """DeFi < TradFi should give negative basis."""
        tradfi = 100.0
        defi = 99.9  # 0.1% lower
        
        basis_abs, basis_bps = basis_analysis.calculate_basis(tradfi, defi)
        
        assert basis_abs == pytest.approx(-0.1, rel=0.01)
        assert basis_bps == pytest.approx(-10.0, rel=0.01)
    
    def test_zero_basis(self):
        """Equal prices should give zero basis."""
        tradfi = 100.0
        defi = 100.0
        
        basis_abs, basis_bps = basis_analysis.calculate_basis(tradfi, defi)
        
        assert basis_abs == 0.0
        assert basis_bps == 0.0
    
    def test_handles_zero_tradfi(self):
        """Should handle zero TradFi price without error."""
        tradfi = 0.0
        defi = 100.0
        
        basis_abs, basis_bps = basis_analysis.calculate_basis(tradfi, defi)
        
        assert basis_abs == 100.0
        assert basis_bps == 0.0  # Avoid division by zero
    
    def test_large_basis(self):
        """Test with larger price difference."""
        tradfi = 100.0
        defi = 101.0  # 1% higher
        
        basis_abs, basis_bps = basis_analysis.calculate_basis(tradfi, defi)
        
        assert basis_abs == pytest.approx(1.0, rel=0.01)
        assert basis_bps == pytest.approx(100.0, rel=0.01)  # 100 bps = 1%


# ============================================
# Test: Market Hours Detection
# ============================================

class TestIsMarketOpen:
    """Tests for is_market_open function."""
    
    def test_market_open_during_nyse_hours(self):
        """Should return True during NYSE hours (14:30-21:00 UTC)."""
        # 15:00 UTC on Wednesday = 10:00 AM ET
        ts = pd.Timestamp("2026-01-15 15:00:00", tz="UTC")
        assert basis_analysis.is_market_open(ts) is True
    
    def test_market_closed_before_open(self):
        """Should return False before NYSE opens."""
        # 14:00 UTC = 9:00 AM ET (30 min before open)
        ts = pd.Timestamp("2026-01-15 14:00:00", tz="UTC")
        assert basis_analysis.is_market_open(ts) is False
    
    def test_market_closed_after_close(self):
        """Should return False after NYSE closes."""
        # 21:30 UTC = 4:30 PM ET (30 min after close)
        ts = pd.Timestamp("2026-01-15 21:30:00", tz="UTC")
        assert basis_analysis.is_market_open(ts) is False
    
    def test_market_closed_on_weekend(self):
        """Should return False on weekends."""
        # Saturday during what would be trading hours
        ts = pd.Timestamp("2026-01-18 15:00:00", tz="UTC")  # Saturday
        assert basis_analysis.is_market_open(ts) is False
    
    def test_market_open_at_exact_open(self):
        """Should return True at exact market open time."""
        ts = pd.Timestamp("2026-01-15 14:30:00", tz="UTC")
        assert basis_analysis.is_market_open(ts) is True
    
    def test_market_closed_at_exact_close(self):
        """Should return False at exact market close time."""
        ts = pd.Timestamp("2026-01-15 21:00:00", tz="UTC")
        assert basis_analysis.is_market_open(ts) is False


# ============================================
# Test: Basis Calculation for Asset
# ============================================

class TestCalculateBasisForAsset:
    """Tests for calculate_basis_for_asset function."""
    
    def test_calculates_basis_for_aligned_data(self, sample_tradfi_df, sample_defi_df):
        """Should calculate basis for aligned price data."""
        basis_df = basis_analysis.calculate_basis_for_asset(
            sample_tradfi_df, sample_defi_df, "TEST"
        )
        
        assert not basis_df.empty
        assert len(basis_df) == 10
        assert "tradfi_mid" in basis_df.columns
        assert "defi_mid" in basis_df.columns
        assert "basis_absolute" in basis_df.columns
        assert "basis_bps" in basis_df.columns
        assert "market_open" in basis_df.columns
    
    def test_basis_values_correct(self, sample_tradfi_df, sample_defi_df):
        """Should calculate correct basis values."""
        basis_df = basis_analysis.calculate_basis_for_asset(
            sample_tradfi_df, sample_defi_df, "TEST"
        )
        
        # Mid prices: TradFi=100.0, DeFi=100.1 -> basis=0.1, 10 bps
        assert basis_df["basis_absolute"].iloc[0] == pytest.approx(0.1, rel=0.01)
        assert basis_df["basis_bps"].iloc[0] == pytest.approx(10.0, rel=0.01)
    
    def test_handles_empty_tradfi(self, sample_defi_df):
        """Should return empty DataFrame when TradFi data is empty."""
        empty_df = pd.DataFrame()
        
        basis_df = basis_analysis.calculate_basis_for_asset(
            empty_df, sample_defi_df, "TEST"
        )
        
        assert basis_df.empty
    
    def test_handles_empty_defi(self, sample_tradfi_df):
        """Should return empty DataFrame when DeFi data is empty."""
        empty_df = pd.DataFrame()
        
        basis_df = basis_analysis.calculate_basis_for_asset(
            sample_tradfi_df, empty_df, "TEST"
        )
        
        assert basis_df.empty


# ============================================
# Test: Statistics Calculation
# ============================================

class TestCalculateBasisStatistics:
    """Tests for calculate_basis_statistics function."""
    
    def test_calculates_statistics(self, sample_tradfi_df, sample_defi_df):
        """Should calculate all required statistics."""
        basis_df = basis_analysis.calculate_basis_for_asset(
            sample_tradfi_df, sample_defi_df, "TEST"
        )
        
        stats = basis_analysis.calculate_basis_statistics(basis_df)
        
        assert "count" in stats
        assert "mean_bps" in stats
        assert "std_bps" in stats
        assert "min_bps" in stats
        assert "max_bps" in stats
        assert "median_bps" in stats
        assert "pct_gt_20bps" in stats
    
    def test_handles_empty_dataframe(self):
        """Should return empty dict for empty DataFrame."""
        empty_df = pd.DataFrame()
        
        stats = basis_analysis.calculate_basis_statistics(empty_df)
        
        assert stats == {}
    
    def test_market_hours_stats_included(self, sample_tradfi_df, sample_defi_df):
        """Should include market hours statistics."""
        basis_df = basis_analysis.calculate_basis_for_asset(
            sample_tradfi_df, sample_defi_df, "TEST"
        )
        
        stats = basis_analysis.calculate_basis_statistics(basis_df)
        
        # Sample data is during market hours (14:30 UTC)
        assert "market_hours_mean_bps" in stats
        assert "market_hours_count" in stats


# ============================================
# Test: Output Functions
# ============================================

class TestSaveBasisToCsv:
    """Tests for save_basis_to_csv function."""
    
    def test_saves_csv_file(self, sample_tradfi_df, sample_defi_df, tmp_path, monkeypatch):
        """Should save basis data to CSV file."""
        # Redirect output to temp directory
        monkeypatch.setattr(basis_analysis, "OUTPUT_DIR", tmp_path)
        
        basis_df = basis_analysis.calculate_basis_for_asset(
            sample_tradfi_df, sample_defi_df, "TEST"
        )
        
        csv_path = basis_analysis.save_basis_to_csv(basis_df, "TEST")
        
        assert csv_path.exists()
        assert csv_path.name == "test_basis.csv"
    
    def test_csv_contains_all_columns(self, sample_tradfi_df, sample_defi_df, tmp_path, monkeypatch):
        """CSV should contain all required columns."""
        monkeypatch.setattr(basis_analysis, "OUTPUT_DIR", tmp_path)
        
        basis_df = basis_analysis.calculate_basis_for_asset(
            sample_tradfi_df, sample_defi_df, "TEST"
        )
        
        csv_path = basis_analysis.save_basis_to_csv(basis_df, "TEST")
        
        # Read back and verify
        loaded = pd.read_csv(csv_path)
        
        assert "tradfi_mid" in loaded.columns
        assert "defi_mid" in loaded.columns
        assert "basis_absolute" in loaded.columns
        assert "basis_bps" in loaded.columns
        assert "market_open" in loaded.columns
