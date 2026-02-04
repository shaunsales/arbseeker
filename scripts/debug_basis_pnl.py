"""Debug script to understand basis arb P&L mechanics."""

import pandas as pd
import numpy as np

# Load data
tradfi = pd.read_parquet('tests/fixtures/futures/GC/15m/2025-12.parquet')
defi = pd.read_parquet('tests/fixtures/perp/PAXG/15m/2025-12.parquet')
common = tradfi.index.intersection(defi.index)
tradfi = tradfi.loc[common]
defi = defi.loc[common]

# Calculate basis
basis = (defi['close'] - tradfi['close']) / tradfi['close'] * 10000

# Find first entry at >80 bps
entry_idx = (abs(basis) > 80).idxmax()
entry_loc = basis.index.get_loc(entry_idx)
entry_basis = basis.iloc[entry_loc]

print('TRADE WALKTHROUGH - Understanding Basis Arb P&L')
print('=' * 70)
print(f'Entry time: {entry_idx}')
print(f'Entry basis: {entry_basis:.1f} bps (DeFi is cheaper)')
print()

# Entry prices
T1 = tradfi.iloc[entry_loc]['close']
D1 = defi.iloc[entry_loc]['close']
notional = 186290

print(f'Entry Prices:')
print(f'  TradFi: ${T1:.2f}')
print(f'  DeFi:   ${D1:.2f}')
print(f'  Spread: ${T1 - D1:.2f} ({entry_basis:.1f} bps)')
print()

# Position (Long DeFi, Short TradFi) - EQUAL NOTIONAL
print(f'Position (notional=${notional:,}):')
defi_units = notional / D1
tradfi_units = notional / T1
print(f'  Long DeFi:    {defi_units:.4f} units @ ${D1:.2f}')
print(f'  Short TradFi: {tradfi_units:.4f} units @ ${T1:.2f}')
print(f'  Unit difference: {abs(defi_units - tradfi_units):.4f} units')
print()

# Track for 10 bars
print('Evolution over 10 bars:')
print('-' * 70)
print(f'Bar  TradFi   DeFi     Basis    DeFi_PnL   TradFi_PnL  Combined')
print('-' * 70)

for i in range(11):
    idx = entry_loc + i
    if idx >= len(basis):
        break
    
    T2 = tradfi.iloc[idx]['close']
    D2 = defi.iloc[idx]['close']
    curr_basis = basis.iloc[idx]
    
    # P&L per leg (actual price-based)
    defi_pnl = (D2 - D1) * defi_units
    tradfi_pnl = (T1 - T2) * tradfi_units  # Short position profits when price drops
    combined = defi_pnl + tradfi_pnl
    
    # Basis-based P&L (what we calculate in spread mode)
    captured_bps = abs(entry_basis) - abs(curr_basis)
    basis_pnl = captured_bps * notional / 10000
    
    print(f'{i:>3}  {T2:>7.2f} {D2:>7.2f}  {curr_basis:>+7.1f}  {defi_pnl:>+10.2f} {tradfi_pnl:>+11.2f} {combined:>+10.2f}')

print()
print('KEY INSIGHT:')
print('With EQUAL NOTIONAL (not equal units), the combined P&L does NOT')
print('perfectly track basis change because unit counts differ.')
print()

# Show the math
print('The Math:')
print(f'  DeFi units:   {defi_units:.4f}')
print(f'  TradFi units: {tradfi_units:.4f}')
print(f'  Ratio: {defi_units/tradfi_units:.6f}')
print()
print('For a TRUE hedge (locked spread), we need EQUAL UNITS, not equal notional.')
print('But in practice, we use equal notional for simplicity.')
