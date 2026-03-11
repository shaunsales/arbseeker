"""
CoinRoutes API routes.

Thin wrapper around the CoinRoutes client for the Web UI.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from core.coinroutes import CoinRoutesClient

router = APIRouter()

_client: CoinRoutesClient | None = None


def _get_client() -> CoinRoutesClient:
    """Lazy singleton — created on first use so env vars are loaded."""
    global _client
    if _client is None:
        _client = CoinRoutesClient()
    return _client


@router.post("/connect")
async def connect():
    """Login and verify connectivity to CoinRoutes."""
    try:
        client = _get_client()
        client.login()
        return {"connected": True, "url": client.base_url, "strategy": client.strategy}
    except Exception as e:
        return JSONResponse({"connected": False, "error": str(e)}, status_code=502)


@router.get("/strategies")
async def strategies():
    """List available CoinRoutes strategies."""
    return _get_client().get_strategies()


@router.get("/exchanges")
async def exchanges():
    """List available exchanges."""
    return _get_client().get_exchanges()


@router.get("/currency_pairs")
async def currency_pairs():
    """List available currency pairs."""
    return _get_client().get_currency_pairs(limit=500)


@router.get("/balances")
async def balances():
    """Get account balances."""
    return _get_client().get_balances()


@router.get("/positions")
async def positions():
    """Get current positions."""
    return _get_client().get_positions()


@router.get("/orders")
async def orders():
    """List recent orders."""
    return _get_client().list_orders(limit=50)
