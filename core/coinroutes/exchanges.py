"""
Exchange type categorization for CoinRoutes venues.

Categories:
- CEX: Centralized exchanges
- DEX: Decentralized exchanges
- MM: Market makers / liquidity providers
- OTC: OTC desks
- SIM: Simulation / paper trading venues
"""

from enum import Enum


class ExchangeType(str, Enum):
    CEX = "CEX"
    DEX = "DEX"
    MM = "MM"
    OTC = "OTC"
    SIM = "SIM"


# slug → ExchangeType
EXCHANGE_TYPES: dict[str, ExchangeType] = {
    # ── CEX ──
    "binance": ExchangeType.CEX,
    "binancefutures": ExchangeType.CEX,
    "binanceus": ExchangeType.CEX,
    "bitfinex": ExchangeType.CEX,
    "bitget": ExchangeType.CEX,
    "bitgetswaps": ExchangeType.CEX,
    "bitso": ExchangeType.CEX,
    "bitstamp": ExchangeType.CEX,
    "bitvavo": ExchangeType.CEX,
    "bullish": ExchangeType.CEX,
    "bybit": ExchangeType.CEX,
    "coinbase_adv": ExchangeType.CEX,
    "coinbase_intl": ExchangeType.CEX,
    "crypto_com": ExchangeType.CEX,
    "crypto_com_futures": ExchangeType.CEX,
    "deribit": ExchangeType.CEX,
    "gateio": ExchangeType.CEX,
    "gdax": ExchangeType.CEX,  # Coinbase Pro
    "gemini": ExchangeType.CEX,
    "huobipro": ExchangeType.CEX,
    "huobiswaps": ExchangeType.CEX,
    "kraken": ExchangeType.CEX,
    "krakenfutures": ExchangeType.CEX,
    "kucoin": ExchangeType.CEX,
    "kucoinfutures": ExchangeType.CEX,
    "lmax": ExchangeType.CEX,
    "m2": ExchangeType.CEX,
    "mexc": ExchangeType.CEX,
    "okex": ExchangeType.CEX,
    "okex_options": ExchangeType.CEX,

    # ── DEX ──
    "dydx": ExchangeType.DEX,
    "grvt": ExchangeType.DEX,
    "hyperliquid": ExchangeType.DEX,
    "hyperliquid_spot": ExchangeType.DEX,
    "lighter": ExchangeType.DEX,
    "uniswap": ExchangeType.DEX,
    "uniswap_arb": ExchangeType.DEX,
    "apex": ExchangeType.DEX,
    "aster": ExchangeType.DEX,
    "asterperps": ExchangeType.DEX,
    "edx": ExchangeType.CEX,

    # ── Market Makers / LPs ──
    "b2c2": ExchangeType.MM,
    "cumberland": ExchangeType.MM,
    "cumberlandv2": ExchangeType.MM,
    "enigma": ExchangeType.MM,
    "falconx": ExchangeType.MM,
    "finery": ExchangeType.MM,
    "finerymarkets": ExchangeType.MM,
    "flowtraders": ExchangeType.MM,
    "galaxy": ExchangeType.MM,
    "glxy": ExchangeType.MM,
    "laserdigital": ExchangeType.MM,
    "lmax_lp": ExchangeType.MM,
    "lucera": ExchangeType.MM,
    "wintermute": ExchangeType.MM,

    # ── OTC / Institutional ──
    "cqg": ExchangeType.OTC,
    "cde_cqg": ExchangeType.OTC,
    "crossover": ExchangeType.OTC,
    "ibkr": ExchangeType.OTC,
    "itbit": ExchangeType.OTC,
    "jbdrax": ExchangeType.OTC,
    "lightnet": ExchangeType.OTC,
    "lmaxfx": ExchangeType.OTC,
    "pacifica": ExchangeType.OTC,
    "route28": ExchangeType.OTC,
    "sts": ExchangeType.OTC,
    "xcoin": ExchangeType.OTC,

    # ── Simulation ──
    "sim_binance": ExchangeType.SIM,
    "sim_binancefutures": ExchangeType.SIM,
    "sim_bullish": ExchangeType.SIM,
    "sim_bybit": ExchangeType.SIM,
    "sim_cqg": ExchangeType.SIM,
    "sim_deribit": ExchangeType.SIM,
    "sim_dydx": ExchangeType.SIM,
    "sim_falconx": ExchangeType.SIM,
    "sim_gateio": ExchangeType.SIM,
    "sim_gdax": ExchangeType.SIM,
    "sim_hyperliquid": ExchangeType.SIM,
    "sim_kraken": ExchangeType.SIM,
    "sim_kucoin": ExchangeType.SIM,
    "sim_kucoinfutures": ExchangeType.SIM,
    "sim_m2": ExchangeType.SIM,
    "sim_mexc": ExchangeType.SIM,
    "sim_okex": ExchangeType.SIM,
}


def get_exchange_type(slug: str) -> str:
    """Return the exchange type for a slug, defaulting to 'CEX'."""
    return EXCHANGE_TYPES.get(slug, ExchangeType.CEX).value
