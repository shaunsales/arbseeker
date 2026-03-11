"""CoinRoutes API client for UAT and production trading."""

import os
import requests


class CoinRoutesClient:
    """HTTP client for the CoinRoutes SOR REST API.

    Authenticates via API token (Authorization: Token <token>).
    All trading is routed through CoinRoutes as the execution layer.
    """

    def __init__(self, base_url=None, api_token=None, strategy=None):
        self.base_url = (base_url or os.getenv("COINROUTES_URL", "https://uatj.coinroutes.io")).rstrip("/")
        self.api_token = api_token or os.getenv("COINROUTES_API_TOKEN")
        self.strategy = strategy or os.getenv("COINROUTES_STRATEGY", "atns")
        self.session = requests.Session()

    def login(self):
        """Set up token auth headers and verify connectivity."""
        if not self.api_token:
            raise ValueError(
                "No API token configured. Set COINROUTES_API_TOKEN in .env"
            )
        self.session.headers.update({
            "Authorization": f"Token {self.api_token}",
            "Accept": "application/json",
        })
        # Verify token works
        resp = self.session.get(
            f"{self.base_url}/api/strategies/", timeout=10
        )
        resp.raise_for_status()
        return True

    # ── HTTP helpers ───────────────────────────────────────────────────────

    def _get(self, path, **params):
        """GET request, return parsed JSON."""
        resp = self.session.get(
            f"{self.base_url}{path}", params=params, timeout=10
        )
        self._check(resp)
        return resp.json()

    def _post(self, path, payload=None):
        """POST request with JSON body, return parsed JSON."""
        resp = self.session.post(
            f"{self.base_url}{path}", json=payload, timeout=10
        )
        self._check(resp)
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()

    @staticmethod
    def _check(resp):
        """Raise with response body detail on HTTP errors."""
        if not resp.ok:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise requests.HTTPError(
                f"{resp.status_code} {resp.reason}: {detail}",
                response=resp,
            )

    # ── Discovery ──────────────────────────────────────────────────────────

    def get_strategies(self):
        """List available strategies."""
        return self._get("/api/strategies/")

    def get_exchanges(self):
        """List available exchanges."""
        return self._get("/api/exchanges/")

    def get_currency_pairs(self, limit=100):
        """List available currency pairs."""
        return self._get("/api/currency_pairs/", limit=limit)

    def search_currency_pairs(self, slug: str):
        """Search currency pairs by slug with exchange availability (v2 API).

        Returns a list of pairs matching the slug, each with
        currency_pair_to_exchanges showing which exchanges support it.
        """
        resp = self._get("/api/currency_pairs_v2/", slug=slug, limit=50)
        return resp.get("results", []) if isinstance(resp, dict) else resp

    def get_all_pairs_simple(self) -> list[dict]:
        """Fetch all pairs from the v2 simple endpoint (paginated).

        Returns a flat list of {slug, product_type, exchanges[]}.
        """
        results = []
        page = 1
        while True:
            resp = self._get("/api/currency_pairs_v2/simple/", limit=250, page=page)
            results.extend(resp.get("results", []))
            if not resp.get("next"):
                break
            page += 1
        return results

    def cost_calculator(self, symbol: str, side: str = "both",
                        target_quantity: float | None = None,
                        markets: str | None = None,
                        depth_limit: int | None = None):
        """Query real-time liquidity via the MERX Cost Calculator.

        Args:
            symbol: e.g. "BTC-USDT"
            side: "buy", "sell", or "both"
            target_quantity: quantity to trade
            markets: semicolon-separated list, e.g. "BINANCE;COINBASE"
            depth_limit: max order book depth to consider
        """
        params: dict = {"side": side}
        if target_quantity is not None:
            params["target_quantity"] = target_quantity
        if markets is not None:
            params["markets"] = markets
        if depth_limit is not None:
            params["depth_limit"] = depth_limit
        return self._get(f"/api/cost_calculator/{symbol}", **params)

    def get_balances(self):
        """Get account currency balances."""
        return self._get("/api/currency_balances/")

    def get_positions(self):
        """Get current positions."""
        return self._get("/api/positions/")

    # ── Orders ─────────────────────────────────────────────────────────────

    def list_orders(self, limit=50):
        """List client orders."""
        return self._get("/api/client_orders/", limit=limit)

    def get_order(self, order_id):
        """Get details of a specific order."""
        return self._get(f"/api/client_orders/{order_id}/")

    def get_order_sor_orders(self, order_id):
        """Get SOR (child) orders for a client order."""
        return self._get(f"/api/client_orders/{order_id}/sor_orders/")

    def create_order(self, currency_pair, side, quantity, order_type="sweep",
                     exchanges=None, aggression=None, limit_price=None,
                     **kwargs):
        """Place a new order.

        Args:
            currency_pair: e.g. "BTC-USDT"
            side: "buy" or "sell"
            quantity: order size as string, e.g. "0.001"
            order_type: one of "smart post", "sweep", "smart stop", "spread",
                        "pairs", "pov", "dma", "twap", "scheduled post", "time pace"
            exchanges: list of exchange names, e.g. ["sim_binance"]
            aggression: e.g. "passive", "neutral", "aggressive"
            limit_price: optional limit price as string
            **kwargs: additional API fields (see OpenAPI spec)
        """
        payload = {
            "strategy": self.strategy,
            "order_type": order_type,
            "currency_pair": currency_pair,
            "side": side,
            "quantity": str(quantity),
        }
        if exchanges is not None:
            payload["exchanges"] = exchanges
        if aggression is not None:
            payload["aggression"] = aggression
        if limit_price is not None:
            payload["limit_price"] = str(limit_price)
        payload.update(kwargs)
        return self._post("/api/client_orders/", payload)

    def cancel_order(self, order_id):
        """Cancel an active order."""
        return self._post(f"/api/client_orders/{order_id}/cancel/")
