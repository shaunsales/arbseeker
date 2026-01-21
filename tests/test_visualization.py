#!/usr/bin/env python3
"""
Unit tests for Stage 3: Visualization

Tests chart generation and statistics calculation functions.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from importlib import import_module

# Import the module
visualization = import_module("3_visualization")


# ============================================
# Fixtures
# ============================================

@pytest.fixture
def sample_basis_df():
    """Create sample basis data for testing."""
    timestamps = pd.date_range(
        start="2026-01-15 14:30:00",
        periods=100,
        freq="1min",
        tz="UTC"
    )
    
    np.random.seed(42)
    tradfi_mid = 100.0 + np.random.randn(100) * 0.5
    defi_mid = tradfi_mid + np.random.randn(100) * 0.1  # Small premium
    
    df = pd.DataFrame({
        "tradfi_mid": tradfi_mid,
        "defi_mid": defi_mid,
        "basis_absolute": defi_mid - tradfi_mid,
        "basis_bps": ((defi_mid - tradfi_mid) / tradfi_mid) * 10000,
        "market_open": [True] * 50 + [False] * 50,  # Half market hours
    }, index=timestamps)
    
    return df


# ============================================
# Test: Statistics Calculation
# ============================================

class TestCalculateStatistics:
    """Tests for calculate_statistics function."""
    
    def test_calculates_all_metrics(self, sample_basis_df):
        """Should calculate all required statistics."""
        stats = visualization.calculate_statistics(sample_basis_df)
        
        assert "count" in stats
        assert "mean_bps" in stats
        assert "std_bps" in stats
        assert "min_bps" in stats
        assert "max_bps" in stats
        assert "pct_gt_20bps" in stats
    
    def test_count_matches_dataframe(self, sample_basis_df):
        """Count should match DataFrame length."""
        stats = visualization.calculate_statistics(sample_basis_df)
        assert stats["count"] == len(sample_basis_df)
    
    def test_includes_market_hours_stats(self, sample_basis_df):
        """Should include market hours vs off-hours statistics."""
        stats = visualization.calculate_statistics(sample_basis_df)
        
        assert "market_hours_mean_bps" in stats
        assert "off_hours_mean_bps" in stats
    
    def test_handles_empty_dataframe(self):
        """Should return empty dict for empty DataFrame."""
        empty_df = pd.DataFrame()
        stats = visualization.calculate_statistics(empty_df)
        assert stats == {}


# ============================================
# Test: Chart Generation
# ============================================

class TestChartGeneration:
    """Tests for chart generation functions."""
    
    def test_price_comparison_chart_creates_file(self, sample_basis_df, tmp_path, monkeypatch):
        """Should create price comparison PNG file."""
        monkeypatch.setattr(visualization, "CHARTS_DIR", tmp_path)
        
        output_path = visualization.create_price_comparison_chart(
            "TEST", "Test TradFi", "Test DeFi", sample_basis_df
        )
        
        assert output_path.exists()
        assert output_path.suffix == ".png"
        assert "price_comparison" in output_path.name
    
    def test_basis_timeseries_chart_creates_file(self, sample_basis_df, tmp_path, monkeypatch):
        """Should create basis timeseries PNG file."""
        monkeypatch.setattr(visualization, "CHARTS_DIR", tmp_path)
        
        output_path = visualization.create_basis_timeseries_chart("TEST", sample_basis_df)
        
        assert output_path.exists()
        assert output_path.suffix == ".png"
        assert "basis_timeseries" in output_path.name
    
    def test_basis_distribution_chart_creates_file(self, sample_basis_df, tmp_path, monkeypatch):
        """Should create basis distribution PNG file."""
        monkeypatch.setattr(visualization, "CHARTS_DIR", tmp_path)
        
        output_path = visualization.create_basis_distribution_chart("TEST", sample_basis_df)
        
        assert output_path.exists()
        assert output_path.suffix == ".png"
        assert "basis_distribution" in output_path.name
    
    def test_summary_table_creates_file(self, sample_basis_df, tmp_path, monkeypatch):
        """Should create summary table PNG file."""
        monkeypatch.setattr(visualization, "CHARTS_DIR", tmp_path)
        
        stats = visualization.calculate_statistics(sample_basis_df)
        all_stats = {"TEST": stats}
        
        output_path = visualization.create_summary_table(all_stats)
        
        assert output_path.exists()
        assert output_path.suffix == ".png"
        assert "summary_table" in output_path.name


# ============================================
# Test: Data Loading
# ============================================

class TestDataLoading:
    """Tests for data loading functions."""
    
    def test_load_basis_data_returns_empty_for_missing_file(self, tmp_path, monkeypatch):
        """Should return empty DataFrame for missing file."""
        monkeypatch.setattr(visualization, "BASIS_DIR", tmp_path)
        
        df = visualization.load_basis_data("NONEXISTENT")
        
        assert df.empty
    
    def test_load_price_data_returns_empty_for_missing_file(self, tmp_path, monkeypatch):
        """Should return empty DataFrame for missing file."""
        monkeypatch.setattr(visualization, "DATA_CLEANED_DIR", tmp_path)
        
        df = visualization.load_price_data("NONEXISTENT", "yahoo")
        
        assert df.empty
