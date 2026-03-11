import { get, post } from "./client";

export interface ConnectResult {
  connected: boolean;
  url?: string;
  strategy?: string;
  error?: string;
}

export interface Exchange {
  name: string;
  slug: string;
  type: "CEX" | "DEX" | "MM" | "OTC" | "SIM";
}

export interface PairExchange {
  exchange_id: string;
  tick_size: string;
  min_order_size: string;
  min_qty_incr: string;
}

export interface CurrencyPairV2 {
  slug: string;
  pk: number;
  product_type: string;
  tick_size: string;
  currency_pair_to_exchanges: PairExchange[];
}

export function connect(): Promise<ConnectResult> {
  return post<ConnectResult>("/coinroutes/connect", {});
}

export function getExchanges(): Promise<Exchange[]> {
  return get<Exchange[]>("/coinroutes/exchanges");
}

export function searchCurrencyPairs(slug: string): Promise<CurrencyPairV2[]> {
  return get<CurrencyPairV2[]>(`/coinroutes/currency_pairs/search?slug=${encodeURIComponent(slug)}`);
}

export interface ExchangePair {
  slug: string;
  product_type: string;
}

export interface ExchangePairsResult {
  exchange: string;
  count: number;
  pairs: ExchangePair[];
}

export function getExchangePairs(exchangeSlug: string): Promise<ExchangePairsResult> {
  return get<ExchangePairsResult>(`/coinroutes/exchanges/${encodeURIComponent(exchangeSlug)}/pairs`);
}

export interface MultiPair {
  slug: string;
  product_type: string;
  exchanges: string[];
}

export interface MultiPairResult {
  count: number;
  pairs: MultiPair[];
}

export function searchMultiExchangePairs(exchanges: string[], q: string): Promise<MultiPairResult> {
  const params = new URLSearchParams({
    exchanges: exchanges.join(","),
    q,
  });
  return get<MultiPairResult>(`/coinroutes/exchanges/pairs/search?${params}`);
}

export function getBalances(): Promise<unknown> {
  return get("/coinroutes/balances");
}

export function getPositions(): Promise<unknown> {
  return get("/coinroutes/positions");
}

export function getOrders(): Promise<unknown> {
  return get("/coinroutes/orders");
}

export interface CBBOLevel {
  exchange: string;
  level: number;
  price: string;
  qty: string;
  total_qty: string;
}

export interface CBBOMessage {
  product: string;
  markets: string[];
  bids: CBBOLevel[];
  asks: CBBOLevel[];
  generated_timestamp: string;
  type?: string;
  error?: string;
  error_text?: string;
}

export function cbboWsUrl(currencyPair: string, sample: number = 5): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/api/coinroutes/cbbo?currency_pair=${encodeURIComponent(currencyPair)}&sample=${sample}`;
}
