#!/usr/bin/env python3
"""
Test suite to validate SDK connections and data retrieval.
Run with: python test_connections.py
"""

import sys
from datetime import datetime, timedelta


def test_imports():
    """Test all required imports."""
    print("\n[1] Testing imports...")
    
    errors = []
    
    try:
        import yfinance as yf
        print("  ✓ yfinance")
    except ImportError as e:
        errors.append(f"yfinance: {e}")
        print(f"  ✗ yfinance: {e}")
    
    try:
        import pandas as pd
        print("  ✓ pandas")
    except ImportError as e:
        errors.append(f"pandas: {e}")
        print(f"  ✗ pandas: {e}")
    
    try:
        import numpy as np
        print("  ✓ numpy")
    except ImportError as e:
        errors.append(f"numpy: {e}")
        print(f"  ✗ numpy: {e}")
    
    try:
        import matplotlib.pyplot as plt
        print("  ✓ matplotlib")
    except ImportError as e:
        errors.append(f"matplotlib: {e}")
        print(f"  ✗ matplotlib: {e}")
    
    try:
        from hyperliquid.info import Info
        from hyperliquid.utils import constants
        print("  ✓ hyperliquid-python-sdk")
    except ImportError as e:
        errors.append(f"hyperliquid-python-sdk: {e}")
        print(f"  ✗ hyperliquid-python-sdk: {e}")
    
    try:
        from aster.rest_api import Client as AsterClient
        print("  ✓ aster-connector-python")
    except ImportError as e:
        errors.append(f"aster-connector-python: {e}")
        print(f"  ✗ aster-connector-python: {e}")
    
    return len(errors) == 0, errors


def test_hyperliquid_connection():
    """Test Hyperliquid API connection and data retrieval."""
    print("\n[2] Testing Hyperliquid connection...")
    
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    
    try:
        hl_info = Info(constants.MAINNET_API_URL, skip_ws=True)
        
        # Test all_mids
        mids = hl_info.all_mids()
        coin_count = len(mids)
        print(f"  ✓ all_mids() - {coin_count} coins available")
        
        # Check for target assets
        has_tsla = 'TSLA' in mids
        has_gold = 'GOLD' in mids
        print(f"    TSLA available: {has_tsla}")
        print(f"    GOLD available: {has_gold}")
        
        # Test candles_snapshot with BTC (always available)
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = end_time - (60 * 60 * 1000)  # 1 hour ago
        candles = hl_info.candles_snapshot('BTC', '1m', start_time, end_time)
        print(f"  ✓ candles_snapshot() - {len(candles)} candles retrieved (BTC, 1h)")
        
        return True, {"coins": coin_count, "tsla": has_tsla, "gold": has_gold}
    
    except Exception as e:
        print(f"  ✗ Hyperliquid connection failed: {e}")
        return False, str(e)


def test_aster_connection():
    """Test Aster API connection and data retrieval."""
    print("\n[3] Testing Aster connection...")
    
    from aster.rest_api import Client as AsterClient
    
    try:
        client = AsterClient()
        
        # Test exchange_info
        info = client.exchange_info()
        symbols = [s['symbol'] for s in info.get('symbols', [])]
        symbol_count = len(symbols)
        print(f"  ✓ exchange_info() - {symbol_count} symbols available")
        
        # Check for target assets
        has_tsla = 'TSLAUSDT' in symbols
        print(f"    TSLAUSDT available: {has_tsla}")
        
        # Test klines with BTCUSDT (always available)
        klines = client.klines(symbol='BTCUSDT', interval='1m', limit=10)
        print(f"  ✓ klines() - {len(klines)} candles retrieved (BTCUSDT)")
        
        return True, {"symbols": symbol_count, "tsla": has_tsla}
    
    except Exception as e:
        print(f"  ✗ Aster connection failed: {e}")
        return False, str(e)


def test_yahoo_connection():
    """Test Yahoo Finance connection and data retrieval."""
    print("\n[4] Testing Yahoo Finance connection...")
    
    import yfinance as yf
    
    try:
        # Test TSLA
        ticker = yf.Ticker('TSLA')
        hist = ticker.history(period='1d', interval='1m')
        tsla_count = len(hist)
        print(f"  ✓ TSLA - {tsla_count} 1m candles today")
        
        # Test Gold futures
        ticker_gold = yf.Ticker('GC=F')
        hist_gold = ticker_gold.history(period='1d', interval='1m')
        gold_count = len(hist_gold)
        print(f"  ✓ GC=F (Gold) - {gold_count} 1m candles today")
        
        return True, {"tsla_candles": tsla_count, "gold_candles": gold_count}
    
    except Exception as e:
        print(f"  ✗ Yahoo Finance connection failed: {e}")
        return False, str(e)


def test_data_retrieval():
    """Test actual data retrieval from validate_basis module."""
    print("\n[5] Testing data retrieval functions...")
    
    try:
        from validate_basis import (
            get_yahoo_data,
            get_hyperliquid_candles,
            get_hyperliquid_available_coins,
            get_aster_klines,
            get_aster_available_symbols
        )
        
        # Test Hyperliquid available coins
        hl_coins = get_hyperliquid_available_coins()
        print(f"  ✓ get_hyperliquid_available_coins() - {len(hl_coins)} coins")
        
        # Test Aster available symbols
        aster_symbols = get_aster_available_symbols()
        print(f"  ✓ get_aster_available_symbols() - {len(aster_symbols)} symbols")
        
        # Test Yahoo data (small sample)
        yahoo_df = get_yahoo_data('TSLA', days=1)
        print(f"  ✓ get_yahoo_data('TSLA') - {len(yahoo_df)} records")
        
        # Test Hyperliquid candles (BTC as fallback)
        if 'TSLA' in hl_coins:
            hl_df = get_hyperliquid_candles('TSLA', interval='1m', days=1)
            print(f"  ✓ get_hyperliquid_candles('TSLA') - {len(hl_df)} records")
        else:
            hl_df = get_hyperliquid_candles('BTC', interval='1m', days=1)
            print(f"  ✓ get_hyperliquid_candles('BTC') - {len(hl_df)} records (TSLA not available)")
        
        # Test Aster klines
        if 'TSLAUSDT' in aster_symbols:
            aster_df = get_aster_klines('TSLAUSDT', interval='1m', limit=100)
            print(f"  ✓ get_aster_klines('TSLAUSDT') - {len(aster_df)} records")
        else:
            aster_df = get_aster_klines('BTCUSDT', interval='1m', limit=100)
            print(f"  ✓ get_aster_klines('BTCUSDT') - {len(aster_df)} records (TSLAUSDT not available)")
        
        return True, {}
    
    except Exception as e:
        print(f"  ✗ Data retrieval test failed: {e}")
        return False, str(e)


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("BASIS ARB - Connection & Data Test Suite")
    print("=" * 60)
    
    results = {}
    
    # Run tests
    results['imports'] = test_imports()
    
    if results['imports'][0]:
        results['hyperliquid'] = test_hyperliquid_connection()
        results['aster'] = test_aster_connection()
        results['yahoo'] = test_yahoo_connection()
        results['data_retrieval'] = test_data_retrieval()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, (passed, details) in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed! Ready for basis validation.")
    else:
        print("❌ Some tests failed. Check errors above.")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
