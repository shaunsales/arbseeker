import { get, post } from "./client";

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
  chart_data: Record<string, { time: number; value: number }[]> & {
    markers?: {
      time: number;
      position: string;
      color: string;
      shape: string;
      text: string;
    }[];
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
    net_pnl: number;
    bars_held: number;
    entry_reason: string;
    exit_reason: string;
  }[];
  tearsheet_exists: boolean;
  error?: string;
}

export function listBacktests(): Promise<{
  runs: BacktestRun[];
  strategies: { class_name: string; module: string; has_data_spec: boolean }[];
}> {
  return get("/backtest/");
}

export function viewBacktest(
  strategyName: string,
  runId: string,
): Promise<BacktestViewData> {
  return get(`/backtest/view/${strategyName}/${runId}`);
}

export function runBacktest(req: {
  class_name: string;
  capital?: number;
  commission_bps?: number;
  slippage_bps?: number;
  funding_daily_bps?: number;
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
