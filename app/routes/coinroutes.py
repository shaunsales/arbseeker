"""
CoinRoutes API routes.

Thin wrapper around the CoinRoutes client for the Web UI.
Includes a WebSocket proxy for CBBO streaming market data.
"""

import asyncio
import json
import logging
import os
import time
from collections import defaultdict

import websockets
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse

from core.coinroutes import CoinRoutesClient
from core.coinroutes.exchanges import get_exchange_type

logger = logging.getLogger(__name__)
router = APIRouter()

_client: CoinRoutesClient | None = None

# Cached exchange → pairs mapping (refreshed every 10 min)
_pairs_by_exchange: dict[str, list[dict]] | None = None
_pairs_cache_time: float = 0
_PAIRS_CACHE_TTL = 600  # seconds


def _get_client() -> CoinRoutesClient:
    """Lazy singleton — created on first use so env vars are loaded."""
    global _client
    if _client is None:
        _client = CoinRoutesClient()
    return _client


def _get_pairs_by_exchange() -> dict[str, list[dict]]:
    """Return exchange→pairs mapping, cached for 10 min."""
    global _pairs_by_exchange, _pairs_cache_time
    now = time.time()
    if _pairs_by_exchange is not None and (now - _pairs_cache_time) < _PAIRS_CACHE_TTL:
        return _pairs_by_exchange

    all_pairs = _get_client().get_all_pairs_simple()
    by_exchange: dict[str, list[dict]] = defaultdict(list)
    for p in all_pairs:
        entry = {"slug": p["slug"], "product_type": p.get("product_type", "")}
        for ex in p.get("exchanges", []):
            by_exchange[ex].append(entry)

    # Sort pairs within each exchange
    for ex in by_exchange:
        by_exchange[ex].sort(key=lambda x: x["slug"])

    _pairs_by_exchange = dict(by_exchange)
    _pairs_cache_time = now
    logger.info(f"Cached {len(all_pairs)} pairs across {len(_pairs_by_exchange)} exchanges")
    return _pairs_by_exchange


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
    """List available exchanges with type classification."""
    raw = _get_client().get_exchanges()
    for ex in raw:
        ex["type"] = get_exchange_type(ex.get("slug", ""))
    return raw


@router.get("/currency_pairs")
async def currency_pairs():
    """List available currency pairs."""
    return _get_client().get_currency_pairs(limit=500)


@router.get("/currency_pairs/search")
async def search_currency_pairs(slug: str):
    """Search currency pairs by slug, returns exchange availability."""
    return _get_client().search_currency_pairs(slug)


@router.get("/exchanges/{exchange_slug}/pairs")
async def exchange_pairs(exchange_slug: str):
    """List currency pairs available on a specific exchange."""
    try:
        mapping = _get_pairs_by_exchange()
        pairs = mapping.get(exchange_slug, [])
        return {"exchange": exchange_slug, "count": len(pairs), "pairs": pairs}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)


@router.get("/exchanges/pairs/search")
async def multi_exchange_pairs(exchanges: str = Query(...), q: str = Query("")):
    """Search pairs across multiple exchanges.

    Args:
        exchanges: comma-separated exchange slugs
        q: optional base-token filter (e.g. "BTC")

    Returns pairs matching the query with which of the requested exchanges
    support each pair.
    """
    try:
        exchange_list = [e.strip() for e in exchanges.split(",") if e.strip()]
        mapping = _get_pairs_by_exchange()
        q_upper = q.upper().strip()

        # Collect pairs → set of exchanges
        pair_exchanges: dict[str, dict] = {}
        for ex_slug in exchange_list:
            for p in mapping.get(ex_slug, []):
                slug = p["slug"]
                if q_upper and q_upper not in slug:
                    continue
                if slug not in pair_exchanges:
                    pair_exchanges[slug] = {
                        "slug": slug,
                        "product_type": p["product_type"],
                        "exchanges": [],
                    }
                pair_exchanges[slug]["exchanges"].append(ex_slug)

        results = sorted(pair_exchanges.values(), key=lambda x: x["slug"])
        return {"count": len(results), "pairs": results}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)


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


@router.get("/cost_calculator/{symbol}")
async def cost_calculator(symbol: str, side: str = "both",
                          target_quantity: float | None = None,
                          markets: str | None = None,
                          depth_limit: int | None = None):
    """Query real-time liquidity via the MERX Cost Calculator."""
    try:
        return _get_client().cost_calculator(
            symbol, side=side, target_quantity=target_quantity,
            markets=markets, depth_limit=depth_limit,
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)


# ---------------------------------------------------------------------------
# CBBO WebSocket proxy
# Browser → FastAPI WS → CoinRoutes CBBO WS
# Keeps the API token server-side.
# ---------------------------------------------------------------------------

@router.websocket("/cbbo")
async def cbbo_proxy(ws: WebSocket,
                     currency_pair: str = Query(...),
                     size_filter: float = Query(0.0),
                     sample: int = Query(5)):
    """Proxy CBBO streaming data from CoinRoutes to the browser."""
    await ws.accept()

    client = _get_client()
    base = client.base_url.replace("https://", "wss://").replace("http://", "ws://")
    url = f"{base}/api/streaming/cbbo/"
    headers = {"Authorization": f"Token {client.api_token}"}

    try:
        async with websockets.connect(url, additional_headers=headers,
                                       open_timeout=30, close_timeout=10,
                                       ping_interval=20, ping_timeout=20) as upstream:
            # Subscribe
            await upstream.send(json.dumps({
                "currency_pair": currency_pair,
                "size_filter": size_filter,
                "sample": sample,
                "subscribe": True,
            }))

            # Forward upstream messages to browser
            async def upstream_to_browser():
                try:
                    async for msg in upstream:
                        await ws.send_text(msg if isinstance(msg, str) else msg.decode())
                except (websockets.exceptions.ConnectionClosed, WebSocketDisconnect):
                    pass

            # Listen for browser close to shut down cleanly
            async def browser_to_upstream():
                try:
                    while True:
                        data = await ws.receive_text()
                        # Forward any client messages (e.g. unsubscribe)
                        await upstream.send(data)
                except WebSocketDisconnect:
                    pass

            await asyncio.gather(
                upstream_to_browser(),
                browser_to_upstream(),
                return_exceptions=True,
            )
    except Exception as e:
        logger.error(f"CBBO proxy error: {e}")
        try:
            await ws.send_text(json.dumps({"type": "error", "error": str(e)}))
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass
