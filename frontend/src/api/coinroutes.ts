import { get, post } from "./client";

export interface ConnectResult {
  connected: boolean;
  url?: string;
  strategy?: string;
  error?: string;
}

export function connect(): Promise<ConnectResult> {
  return post<ConnectResult>("/coinroutes/connect", {});
}

export function getStrategies(): Promise<unknown> {
  return get("/coinroutes/strategies");
}

export function getExchanges(): Promise<unknown> {
  return get("/coinroutes/exchanges");
}

export function getCurrencyPairs(): Promise<unknown> {
  return get("/coinroutes/currency_pairs");
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
