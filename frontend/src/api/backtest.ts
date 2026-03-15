import { get, post, del } from "./client";

export interface BacktestRun {
  strategy_name: string;
  run_id: string;
  meta: Record<string, unknown>;
  folder: string;
}

export interface BacktestViewData {
  strategy_name: string;
  run_id: string;
  meta: Record<string, unknown>;
  chart_data: Record<string, unknown> & {
    ohlcv?: { time: number; open: number; high: number; low: number; close: number }[];
    volume?: { time: number; value: number; color: string }[];
    price?: { time: number; value: number }[];
    equity?: { time: number; value: number }[];
    drawdown?: { time: number; value: number }[];
    max_drawdown?: { time: number; value: number }[];
    daily_nav?: { time: number; value: number }[];
    daily_mdd?: { time: number; value: number }[];
    markers?: {
      time: number;
      position: string;
      color: string;
      shape: string;
      text: string;
    }[];
    interval?: string;
  };
  trades: {
    side: string;
    entry_time: string;
    entry_price: number;
    exit_time: string;
    exit_price: number;
    size: number;
    gross_pnl: number;
    costs: number;
    fees: number;
    funding_cost: number;
    net_pnl: number;
    bars_held: number;
    entry_reason: string;
    exit_reason: string;
    metadata?: {
      entry_context?: Record<string, Record<string, unknown> | null>;
      exit_context?: Record<string, Record<string, unknown> | null>;
    } | null;
  }[];
  indicators: {
    name: string;
    interval: string;
    display: string;
    render: Record<string, unknown>;
    series: { time: number; value: number }[];
  }[];
  funding_rates?: { month: string; rate_bps: number }[] | null;
  tearsheet_exists: boolean;
  error?: string;
}

export function listBacktests(): Promise<{
  runs: BacktestRun[];
  strategies: {
    class_name: string;
    module: string;
    has_data_spec: boolean;
    data_date_range?: { start: string; end: string } | null;
    last_modified?: string | null;
  }[];
}> {
  return get("/backtest/");
}

export function viewBacktest(
  strategyName: string,
  runId: string,
): Promise<BacktestViewData> {
  return get(`/backtest/view/${strategyName}/${runId}`);
}

export function deleteBacktest(
  strategyName: string,
  runId: string,
): Promise<{ success: boolean; deleted: string[] }> {
  return del(`/backtest/delete/${strategyName}/${runId}`);
}

export function runBacktest(req: {
  class_name: string;
  capital?: number;
  commission_bps?: number;
  slippage_bps?: number;
  funding_daily_bps?: number;
  start_date?: string; // "YYYY-MM"
  end_date?: string;   // "YYYY-MM"
  stop_loss_pct?: number | null;        // Fixed SL % from entry (null = use strategy default)
  trailing_stop_pct?: number | null;    // TSL % from best price (null = use strategy default)
}): Promise<{
  success: boolean;
  strategy_name?: string;
  run_id?: string;
  metrics?: Record<string, unknown>;
  total_trades?: number;
  error?: string;
}> {
  return post("/backtest/run", req);
}
