#!/usr/bin/env python3
"""
Simple basis validation script - no authentication required.
Downloads historical data from Yahoo Finance and Hyperliquid/Aster.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Official SDKs
from hyperliquid.info import Info
from hyperliquid.utils import constants
from aster.rest_api import Client as AsterClient

# ============================================
# TradFi Data (Yahoo Finance - No Auth Required)
# ============================================

def get_yahoo_data(symbol: str, days: int = 7) -> pd.DataFrame:
    """
    Fetch historical data from Yahoo Finance.
    
    Symbols:
    - TSLA: Tesla stock
    - GC=F: Gold futures
    
    Note: Yahoo limits 1m data to 7 days max.
    """
    ticker = yf.Ticker(symbol)
    end = datetime.now()
    start = end - timedelta(days=days)
    
    df = ticker.history(start=start, end=end, interval="1m")
    df.index = df.index.tz_convert('UTC')  # Normalize to UTC
    df['mid'] = (df['High'] + df['Low']) / 2
    df['source'] = 'yahoo'
    df['symbol'] = symbol
    
    return df[['Open', 'High', 'Low', 'Close', 'mid', 'source', 'symbol']]


# ============================================
# Hyperliquid Data (Official SDK - No Auth Required for reads)
# ============================================

# Initialize Hyperliquid Info client (no auth needed for market data)
hl_info = Info(constants.MAINNET_API_URL, skip_ws=True)


def get_hyperliquid_candles(coin: str, interval: str = "1m", days: int = 3) -> pd.DataFrame:
    """
    Fetch historical candles from Hyperliquid using official SDK.
    
    Coins for HIP-3 markets:
    - TSLA (if available)
    - GOLD (if available)
    
    Note: Only most recent 5000 candles available via API.
    For 1m data, 5000 candles = ~3.5 days.
    """
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)
    
    try:
        candles = hl_info.candles_snapshot(coin, interval, start_time, end_time)
    except Exception as e:
        print(f"Error fetching Hyperliquid data for {coin}: {e}")
        return pd.DataFrame()
    
    if not candles:
        print(f"No candles returned for {coin}")
        return pd.DataFrame()
    
    df = pd.DataFrame(candles)
    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={
        'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'
    })
    df['Open'] = df['Open'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Close'] = df['Close'].astype(float)
    df['mid'] = (df['High'] + df['Low']) / 2
    df['source'] = 'hyperliquid'
    df['symbol'] = coin
    df = df.set_index('timestamp')
    df.index = df.index.tz_localize('UTC')  # Normalize to UTC
    
    return df[['Open', 'High', 'Low', 'Close', 'mid', 'source', 'symbol']]


def get_hyperliquid_available_coins() -> list:
    """Get list of all available coins on Hyperliquid using official SDK."""
    try:
        mids = hl_info.all_mids()
        return list(mids.keys())
    except Exception as e:
        print(f"Error fetching Hyperliquid coins: {e}")
        return []


# ============================================
# Aster Data (Official SDK - No Auth Required for Market Data)
# ============================================

# Initialize Aster client (no auth needed for market data)
aster_client = AsterClient()


def get_aster_klines(symbol: str, interval: str = "1m", limit: int = 1000) -> pd.DataFrame:
    """
    Fetch historical klines from Aster using official SDK.
    
    Symbols for stock perpetuals:
    - TSLAUSDT: Tesla perpetual
    
    Intervals: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    """
    try:
        data = aster_client.klines(symbol=symbol, interval=interval, limit=limit)
    except Exception as e:
        print(f"Error fetching Aster data for {symbol}: {e}")
        return pd.DataFrame()
    
    if not data:
        print(f"No klines returned for {symbol}")
        return pd.DataFrame()
    
    df = pd.DataFrame(data, columns=[
        'open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    df['Open'] = df['Open'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Close'] = df['Close'].astype(float)
    df['mid'] = (df['High'] + df['Low']) / 2
    df['source'] = 'aster'
    df['symbol'] = symbol
    df = df.set_index('timestamp')
    df.index = df.index.tz_localize('UTC')  # Normalize to UTC
    
    return df[['Open', 'High', 'Low', 'Close', 'mid', 'source', 'symbol']]


def get_aster_available_symbols() -> list:
    """Get list of all available symbols on Aster using official SDK."""
    try:
        info = aster_client.exchange_info()
        return [s['symbol'] for s in info.get('symbols', [])]
    except Exception as e:
        print(f"Error fetching Aster symbols: {e}")
        return []


# ============================================
# Basis Calculation
# ============================================

# NYSE market hours in UTC (9:30-16:00 ET = 14:30-21:00 UTC)
NYSE_OPEN_UTC = 14  # 14:30 UTC
NYSE_CLOSE_UTC = 21  # 21:00 UTC


def is_market_hours(timestamp) -> bool:
    """Check if timestamp falls within NYSE trading hours (UTC)."""
    hour = timestamp.hour
    minute = timestamp.minute
    weekday = timestamp.weekday()
    
    # Weekend check
    if weekday >= 5:
        return False
    
    # Market hours: 14:30 - 21:00 UTC
    if hour == 14 and minute >= 30:
        return True
    if hour > 14 and hour < 21:
        return True
    return False


def calculate_basis(tradfi_df: pd.DataFrame, defi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate basis between TradFi and DeFi prices.
    Aligns data by timestamp and calculates spread at minute-level.
    """
    tradfi_df = tradfi_df.copy()
    defi_df = defi_df.copy()
    
    # Resample to 1-minute for alignment
    tradfi_min = tradfi_df['mid'].resample('1min').last().dropna()
    defi_min = defi_df['mid'].resample('1min').last().dropna()
    
    combined = pd.DataFrame({
        'tradfi_mid': tradfi_min,
        'defi_mid': defi_min
    }).dropna()
    
    if combined.empty:
        print("Warning: No overlapping timestamps found!")
        return pd.DataFrame()
    
    combined['basis_absolute'] = combined['defi_mid'] - combined['tradfi_mid']
    combined['basis_pct'] = (combined['basis_absolute'] / combined['tradfi_mid']) * 100
    combined['basis_bps'] = combined['basis_pct'] * 100
    
    # Add market hours flag
    combined['market_open'] = combined.index.map(is_market_hours)
    
    return combined


def analyze_basis(basis_df: pd.DataFrame, asset_name: str):
    """Print basis statistics and generate chart with market hours highlighting."""
    if basis_df.empty:
        print(f"No basis data available for {asset_name}")
        return
    
    market_hours_df = basis_df[basis_df['market_open']]
    after_hours_df = basis_df[~basis_df['market_open']]
    
    print(f"\n{'='*60}")
    print(f"BASIS ANALYSIS: {asset_name}")
    print(f"{'='*60}")
    print(f"Period: {basis_df.index.min()} to {basis_df.index.max()}")
    print(f"Total observations: {len(basis_df)} ({len(market_hours_df)} during market hours)")
    
    print(f"\n--- ALL DATA ---")
    print(f"  Mean:    {basis_df['basis_bps'].mean():>8.2f} bps")
    print(f"  Std Dev: {basis_df['basis_bps'].std():>8.2f} bps")
    print(f"  Min:     {basis_df['basis_bps'].min():>8.2f} bps")
    print(f"  Max:     {basis_df['basis_bps'].max():>8.2f} bps")
    
    if not market_hours_df.empty:
        print(f"\n--- MARKET HOURS ONLY (tradeable) ---")
        print(f"  Mean:    {market_hours_df['basis_bps'].mean():>8.2f} bps")
        print(f"  Std Dev: {market_hours_df['basis_bps'].std():>8.2f} bps")
        print(f"  Min:     {market_hours_df['basis_bps'].min():>8.2f} bps")
        print(f"  Max:     {market_hours_df['basis_bps'].max():>8.2f} bps")
    
    threshold_bps = 20
    profitable_pct = (abs(basis_df['basis_bps']) > threshold_bps).mean() * 100
    print(f"\n% of time |basis| > {threshold_bps} bps (all): {profitable_pct:.1f}%")
    
    if not market_hours_df.empty:
        profitable_mkt = (abs(market_hours_df['basis_bps']) > threshold_bps).mean() * 100
        print(f"% of time |basis| > {threshold_bps} bps (market hours): {profitable_mkt:.1f}%")
    
    # Create chart with market hours shading
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Shade market hours regions
    for ax in axes:
        for idx in basis_df.index:
            if basis_df.loc[idx, 'market_open']:
                ax.axvspan(idx, idx + pd.Timedelta(minutes=1), 
                          alpha=0.1, color='green', linewidth=0)
    
    axes[0].plot(basis_df.index, basis_df['tradfi_mid'], label='TradFi (Yahoo)', alpha=0.7, linewidth=0.5)
    axes[0].plot(basis_df.index, basis_df['defi_mid'], label='DeFi', alpha=0.7, linewidth=0.5)
    axes[0].set_ylabel('Price ($)')
    axes[0].set_title(f'{asset_name}: TradFi vs DeFi Price Comparison (green = market hours)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(basis_df.index, basis_df['basis_bps'], color='purple', alpha=0.7, linewidth=0.5)
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1].axhline(y=threshold_bps, color='red', linestyle='--', alpha=0.5, label=f'+{threshold_bps} bps')
    axes[1].axhline(y=-threshold_bps, color='red', linestyle='--', alpha=0.5, label=f'-{threshold_bps} bps')
    axes[1].fill_between(basis_df.index, -threshold_bps, threshold_bps, alpha=0.1, color='red')
    axes[1].set_ylabel('Basis (bps)')
    axes[1].set_xlabel('Time (UTC)')
    axes[1].set_title(f'{asset_name}: Basis (DeFi - TradFi)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{asset_name.lower()}_basis_analysis.png', dpi=150)
    plt.show()
    
    # Save data to CSV
    basis_df.to_csv(f'{asset_name.lower()}_basis_data.csv')
    print(f"\nData saved to {asset_name.lower()}_basis_data.csv")
    
    return basis_df


# ============================================
# Asset Configuration
# ============================================

# Assets to analyze: (name, yahoo_symbol, aster_symbol, hyperliquid_symbol)
ASSETS = [
    ("TSLA", "TSLA", "TSLAUSDT", "TSLA"),
    ("AAPL", "AAPL", "AAPLUSDT", None),
    ("NVDA", "NVDA", "NVDAUSDT", None),
    ("GOLD", "GC=F", "XAUUSDT", "GOLD"),
    ("SILVER", "SI=F", "XAGUSDT", None),
    ("QQQ", "QQQ", "QQQUSDT", None),
]


# ============================================
# Main Execution
# ============================================

def main():
    print("="*60)
    print("TradFi-DeFi Basis Validation Tool")
    print("="*60)
    
    print("\n[1] Checking available markets...")
    
    # Get available symbols from both DEXs
    print("\nHyperliquid:")
    hl_coins = get_hyperliquid_available_coins()
    print(f"  {len(hl_coins)} coins available")
    
    print("\nAster (primary for RWA/stocks):")
    aster_symbols = get_aster_available_symbols()
    print(f"  {len(aster_symbols)} symbols available")
    
    # Check availability for each asset
    print("\nAsset availability:")
    for name, yahoo_sym, aster_sym, hl_sym in ASSETS:
        on_aster = "✓" if aster_sym in aster_symbols else "✗"
        on_hl = "✓" if hl_sym and hl_sym in hl_coins else "✗"
        print(f"  {name:8} | Aster: {on_aster} | Hyperliquid: {on_hl}")
    
    # Analyze each asset
    step = 2
    for name, yahoo_sym, aster_sym, hl_sym in ASSETS:
        print(f"\n[{step}] Collecting {name} data...")
        
        # TradFi data from Yahoo
        print(f"  Fetching Yahoo Finance {yahoo_sym} (1m data, 7 days)...")
        tradfi_df = get_yahoo_data(yahoo_sym, days=7)
        print(f"  Got {len(tradfi_df)} records from Yahoo")
        
        if tradfi_df.empty:
            print(f"  ⚠ No Yahoo data for {name}, skipping...")
            step += 1
            continue
        
        # DeFi data - prioritize Aster for RWA/stocks
        defi_df = pd.DataFrame()
        defi_source = None
        
        # Try Aster first (has stock perps)
        if aster_sym in aster_symbols:
            print(f"  Fetching Aster {aster_sym} (1m data)...")
            defi_df = get_aster_klines(aster_sym, interval='1m', limit=5000)
            if not defi_df.empty:
                defi_source = "Aster"
                print(f"  Got {len(defi_df)} records from Aster")
        
        # Fallback to Hyperliquid if available and Aster failed
        if defi_df.empty and hl_sym and hl_sym in hl_coins:
            print(f"  Fetching Hyperliquid {hl_sym} (1m data)...")
            defi_df = get_hyperliquid_candles(hl_sym, interval='1m', days=3)
            if not defi_df.empty:
                defi_source = "Hyperliquid"
                print(f"  Got {len(defi_df)} records from Hyperliquid")
        
        if defi_df.empty:
            print(f"  ⚠ No DeFi data for {name}, skipping...")
            step += 1
            continue
        
        # Analyze basis
        print(f"\n  Analyzing {name} basis (Yahoo vs {defi_source})...")
        basis_df = calculate_basis(tradfi_df, defi_df)
        analyze_basis(basis_df, name)
        
        step += 1
    
    print("\n" + "="*60)
    print("Validation complete! Check generated PNG charts.")
    print("="*60)


if __name__ == "__main__":
    main()
