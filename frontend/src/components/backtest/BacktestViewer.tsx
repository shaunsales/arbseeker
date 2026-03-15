import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import {
  createChart,
  createSeriesMarkers,
  type IChartApi,
  ColorType,
  CandlestickSeries,
  LineSeries,
  AreaSeries,
  HistogramSeries,
} from "lightweight-charts";
import { DotsSeries } from "@/plugins/dots-series";
import type { BacktestViewData } from "@/api/backtest";
import { Badge } from "@/components/ui/badge";
import { ChevronRight, ChevronDown, ChevronLeft, Crosshair, ZoomIn, ZoomOut, Maximize2 } from "lucide-react";

interface Props {
  data: BacktestViewData;
}

const METRIC_DISPLAY: {
  key: string;
  label: string;
  fmt: (v: unknown) => string;
  color?: (v: unknown) => string;
}[] = [
  {
    key: "total_return_pct",
    label: "Total Return",
    fmt: (v) => `${(v as number) >= 0 ? "+" : ""}${(v as number).toFixed(2)}%`,
    color: (v) => ((v as number) >= 0 ? "text-green-400" : "text-red-400"),
  },
  {
    key: "sharpe_ratio",
    label: "Sharpe Ratio",
    fmt: (v) => (v as number).toFixed(2),
  },
  {
    key: "max_drawdown_pct",
    label: "Max Drawdown",
    fmt: (v) => `-${(v as number).toFixed(2)}%`,
    color: () => "text-red-400",
  },
  {
    key: "win_rate",
    label: "Win Rate",
    fmt: (v) => `${(v as number).toFixed(1)}%`,
  },
  {
    key: "total_trades",
    label: "Trades",
    fmt: (v) => String(Math.round(v as number)),
  },
  {
    key: "profit_factor",
    label: "Profit Factor",
    fmt: (v) =>
      v === "∞" ? "∞" : (v as number).toFixed(2),
  },
  {
    key: "initial_capital",
    label: "Initial Capital",
    fmt: (v) => `$${(v as number).toLocaleString("en", { maximumFractionDigits: 0 })}`,
  },
  {
    key: "final_capital",
    label: "Final Capital",
    fmt: (v) => `$${(v as number).toLocaleString("en", { maximumFractionDigits: 0 })}`,
    color: () => "text-gray-200",
  },
];

export default function BacktestViewer({ data }: Props) {
  const [tab, setTab] = useState<"performance" | "analysis" | "trades">("performance");
  const [selectedTradeIdx, setSelectedTradeIdx] = useState<number | null>(null);
  const metrics = (data.meta as Record<string, unknown>).metrics as
    | Record<string, unknown>
    | undefined;
  const period = metrics?.period as string | undefined;

  const handleTradeSelect = (idx: number) => {
    setSelectedTradeIdx(idx);
    setTab("analysis");
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center gap-3">
        <h2 className="text-sm font-semibold text-gray-200">
          <span className="text-blue-400">{data.strategy_name}</span>
        </h2>
        <Badge variant="secondary" className="text-[10px]">
          {data.run_id}
        </Badge>
        {period && (
          <span className="text-xs text-gray-500">
            {period.split(" to ").map((d) => d.slice(0, 10)).join(" → ")}
          </span>
        )}
        {data.tearsheet_exists && (
          <a
            href={`/api/backtest/tearsheet/${data.strategy_name}/${data.run_id}`}
            target="_blank"
            rel="noopener noreferrer"
            className="ml-auto text-xs text-blue-400 hover:underline"
          >
            Open Tearsheet →
          </a>
        )}
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-800">
        {(["performance", "analysis", "trades"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`border-b-2 px-3 py-1.5 text-xs font-medium transition ${
              tab === t
                ? "border-blue-500 text-blue-400"
                : "border-transparent text-gray-500 hover:text-gray-300"
            }`}
          >
            {t === "performance"
              ? "Performance"
              : t === "analysis"
              ? "Analysis"
              : `Trades (${data.trades.length})`}
          </button>
        ))}
      </div>

      {tab === "performance" && (
        <PerformanceTab
          data={data}
          metrics={metrics}
        />
      )}

      {tab === "analysis" && (
        <>
          {data.trades.length > 0 && (
            <TradeNavigator
              trades={data.trades}
              selectedIdx={selectedTradeIdx}
              onSelect={handleTradeSelect}
            />
          )}
          <BacktestCharts
            chartData={data.chart_data}
            indicators={data.indicators ?? []}
            trades={data.trades}
            selectedTradeIdx={selectedTradeIdx}
          />
        </>
      )}

      {tab === "trades" && (
        <TradesTable
          trades={data.trades}
          selectedTradeIdx={selectedTradeIdx}
          onTradeSelect={handleTradeSelect}
        />
      )}
    </div>
  );
}

// ── Performance Tab ──

function PerformanceTab({
  data,
  metrics,
}: {
  data: BacktestViewData;
  metrics?: Record<string, unknown>;
}) {
  const config = data.meta?.config as Record<string, unknown> | undefined;
  const costs = config?.costs as { commission_bps?: number; slippage_bps?: number; funding_daily_bps?: number } | undefined;
  const spec = config?.spec as { venue?: string; market?: string; ticker?: string; intervals?: Record<string, unknown[]> } | undefined;
  const hasDynamicFunding = data.funding_rates != null && data.funding_rates.length > 0;

  return (
    <div className="space-y-4">
      {/* Run Configuration */}
      {config && (
        <div className="rounded border border-gray-800 bg-gray-900 p-3">
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-gray-500">Run Configuration</p>
          <div className="grid grid-cols-2 gap-x-8 gap-y-1 text-[12px] sm:grid-cols-3 lg:grid-cols-4">
            {spec?.ticker && (
              <div><span className="text-gray-500">Symbol:</span> <span className="font-medium text-gray-200">{spec.ticker}</span></div>
            )}
            {spec?.venue && (
              <div><span className="text-gray-500">Venue:</span> <span className="font-medium text-gray-200">{spec.venue} {spec.market}</span></div>
            )}
            {config.capital != null && (
              <div><span className="text-gray-500">Capital:</span> <span className="font-medium text-gray-200">${Number(config.capital).toLocaleString()}</span></div>
            )}
            {config.start_date != null && config.end_date != null && (
              <div><span className="text-gray-500">Period:</span> <span className="font-medium text-gray-200">{String(config.start_date)} → {String(config.end_date)}</span></div>
            )}
            {costs?.commission_bps != null && (
              <div><span className="text-gray-500">Commission:</span> <span className="font-medium text-gray-200">{costs.commission_bps} bps</span></div>
            )}
            {costs?.slippage_bps != null && (
              <div><span className="text-gray-500">Slippage:</span> <span className="font-medium text-gray-200">{costs.slippage_bps} bps</span></div>
            )}
            <div>
              <span className="text-gray-500">Funding:</span>{" "}
              {hasDynamicFunding ? (
                <span className="font-medium text-amber-400">Dynamic (per-symbol schedule)</span>
              ) : costs?.funding_daily_bps != null && costs.funding_daily_bps > 0 ? (
                <span className="font-medium text-gray-200">{costs.funding_daily_bps} bps/day (flat)</span>
              ) : (
                <span className="font-medium text-gray-500">None</span>
              )}
            </div>
            <div>
              <span className="text-gray-500">Stop Loss:</span>{" "}
              {config.stop_loss_pct != null ? (
                <span className="font-medium text-gray-200">{String(config.stop_loss_pct)}%</span>
              ) : (
                <span className="font-medium text-gray-500">Off</span>
              )}
            </div>
            <div>
              <span className="text-gray-500">Trailing Stop:</span>{" "}
              {config.trailing_stop_pct != null ? (
                <span className="font-medium text-gray-200">{String(config.trailing_stop_pct)}%</span>
              ) : (
                <span className="font-medium text-gray-500">Strategy default</span>
              )}
            </div>
            {spec?.intervals && Object.entries(spec.intervals).map(([interval, inds]) => {
              const indList = inds as [string, Record<string, unknown>][];
              if (!indList.length) return null;
              return (
                <div key={interval} className="col-span-full">
                  <span className="text-gray-500">{interval} indicators:</span>{" "}
                  <span className="font-medium text-gray-300">
                    {indList.map(([name, params]) => {
                      const paramStr = Object.entries(params).map(([k, v]) => `${k}=${v}`).join(", ");
                      return `${name.toUpperCase()}(${paramStr})`;
                    }).join(", ")}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Metrics grid */}
      {metrics && (
        <div className="grid grid-cols-4 gap-3">
          {METRIC_DISPLAY.map(({ key, label, fmt, color }) => {
            const val = metrics[key];
            if (val == null) return null;
            const textColor = color ? color(val) : "text-gray-200";
            return (
              <div
                key={key}
                className="rounded border border-gray-800 bg-gray-900 px-3 py-2"
              >
                <p className="text-[11px] text-gray-500">{label}</p>
                <p className={`text-base font-semibold ${textColor}`}>
                  {fmt(val)}
                </p>
              </div>
            );
          })}
        </div>
      )}

      {/* NAV + Max Drawdown chart */}
      <NavDrawdownChart chartData={data.chart_data} />

      {/* Funding rate bar */}
      {data.funding_rates && data.funding_rates.length > 0 && (
        <FundingRateBar rates={data.funding_rates} />
      )}
    </div>
  );
}

// ── NAV + Max Drawdown Chart (Performance tab) ──

function NavDrawdownChart({
  chartData,
}: {
  chartData: BacktestViewData["chart_data"];
}) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const hasDailyNav = chartData.daily_nav && chartData.daily_nav.length > 0;
    const hasNav = hasDailyNav || (chartData.equity && chartData.equity.length > 0);
    if (!hasNav) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid as const, color: "#111827" },
        textColor: "#9ca3af",
      },
      grid: {
        vertLines: { color: "#1f2937" },
        horzLines: { color: "#1f2937" },
      },
      crosshair: { mode: 0 },
      rightPriceScale: { borderColor: "#374151" },
      leftPriceScale: { borderColor: "#374151", visible: true },
      timeScale: { borderColor: "#374151", timeVisible: true },
      height: 260,
      autoSize: true,
    });

    // NAV line (right scale) — prefer daily
    const navData = (hasDailyNav ? chartData.daily_nav : chartData.equity)!;
    const navSeries = chart.addSeries(LineSeries, {
      color: "#3b82f6",
      lineWidth: 2,
      title: "NAV",
      priceScaleId: "right",
      lastValueVisible: true,
      priceLineVisible: false,
    });
    navSeries.setData(navData as never[]);

    // Rolling Max Drawdown area (left scale) — prefer daily
    const mddData = chartData.daily_mdd ?? chartData.max_drawdown;
    if (mddData?.length) {
      const mddSeries = chart.addSeries(AreaSeries, {
        lineColor: "#ef4444",
        lineWidth: 1,
        topColor: "rgba(239, 68, 68, 0.0)",
        bottomColor: "rgba(239, 68, 68, 0.35)",
        title: "Max Drawdown %",
        priceScaleId: "left",
        lastValueVisible: true,
        priceLineVisible: false,
      });
      mddSeries.setData(mddData as never[]);
    }

    chart.timeScale().fitContent();
    return () => chart.remove();
  }, [chartData]);

  const hasNav = (chartData.daily_nav && chartData.daily_nav.length > 0) || (chartData.equity && chartData.equity.length > 0);
  if (!hasNav) return null;

  return (
    <div>
      <p className="mb-1 text-[11px] font-medium text-gray-500">NAV & Max Drawdown</p>
      <div ref={containerRef} className="rounded bg-gray-900" style={{ height: 260 }} />
    </div>
  );
}

// ── Funding Rate Bar (compact) ──

function FundingRateBar({
  rates,
}: {
  rates: { month: string; rate_bps: number }[];
}) {
  const avg = rates.reduce((s, r) => s + r.rate_bps, 0) / rates.length;
  const min = Math.min(...rates.map((r) => r.rate_bps));
  const max = Math.max(...rates.map((r) => r.rate_bps));
  const latest = rates[rates.length - 1];

  // Mini sparkline bars
  const maxAbs = Math.max(...rates.map((r) => Math.abs(r.rate_bps)), 0.01);

  return (
    <div className="rounded border border-gray-800 bg-gray-900 px-4 py-3">
      <div className="mb-2 flex items-center gap-4">
        <p className="text-[11px] font-medium text-gray-500">Funding Rate (bps/day)</p>
        <div className="flex items-center gap-3 text-[10px]">
          <span className="text-gray-500">
            Avg: <span className={`font-mono font-semibold ${avg >= 0 ? "text-amber-400" : "text-emerald-400"}`}>{avg.toFixed(2)}</span>
          </span>
          <span className="text-gray-500">
            Min: <span className="font-mono text-emerald-400">{min.toFixed(2)}</span>
          </span>
          <span className="text-gray-500">
            Max: <span className="font-mono text-amber-400">{max.toFixed(2)}</span>
          </span>
          <span className="text-gray-500">
            Latest ({latest.month}): <span className={`font-mono font-semibold ${latest.rate_bps >= 0 ? "text-amber-400" : "text-emerald-400"}`}>{latest.rate_bps.toFixed(2)}</span>
          </span>
        </div>
      </div>
      {/* Mini bar chart */}
      <div className="flex items-end gap-px" style={{ height: 32 }}>
        {rates.map((r) => {
          const h = Math.max(2, Math.abs(r.rate_bps) / maxAbs * 28);
          const color = r.rate_bps >= 0 ? "bg-amber-500/60" : "bg-emerald-500/60";
          return (
            <div
              key={r.month}
              className={`flex-1 rounded-t ${color}`}
              style={{ height: `${h}px` }}
              title={`${r.month}: ${r.rate_bps.toFixed(2)} bps/day`}
            />
          );
        })}
      </div>
    </div>
  );
}

// ── Trade Navigator ──

type TradeRow = BacktestViewData["trades"][number];

/** Format seconds into HH:MM:SS or Xd HH:MM:SS */
function formatDuration(seconds: number): string {
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  const hms = `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return d > 0 ? `${d}d ${hms}` : hms;
}

/** Format ISO timestamp to readable local time */
function formatTimestamp(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleString("en-GB", {
    year: "numeric", month: "short", day: "2-digit",
    hour: "2-digit", minute: "2-digit", second: "2-digit",
    hour12: false,
  });
}

function TradeNavigator({
  trades,
  selectedIdx,
  onSelect,
}: {
  trades: TradeRow[];
  selectedIdx: number | null;
  onSelect: (idx: number) => void;
}) {
  const t = selectedIdx != null ? trades[selectedIdx] : null;

  // Compute held duration in seconds
  const heldSeconds = t
    ? (new Date(t.exit_time).getTime() - new Date(t.entry_time).getTime()) / 1000
    : 0;

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900/60">
      {/* Top row: nav controls */}
      <div className="flex items-center gap-2 px-3 py-2">
        <Crosshair className="h-3.5 w-3.5 shrink-0 text-gray-500" />
        <span className="text-[11px] text-gray-500">Trade</span>

        <button
          disabled={selectedIdx == null || selectedIdx <= 0}
          onClick={() => onSelect(selectedIdx != null ? selectedIdx - 1 : 0)}
          className="rounded p-0.5 text-gray-400 hover:bg-gray-800 hover:text-gray-200 disabled:opacity-30 disabled:hover:bg-transparent"
        >
          <ChevronLeft className="h-3.5 w-3.5" />
        </button>

        <span className="min-w-[60px] text-center text-xs font-semibold tabular-nums text-gray-300">
          {selectedIdx != null ? `${selectedIdx + 1} / ${trades.length}` : `— / ${trades.length}`}
        </span>

        <button
          disabled={selectedIdx == null ? trades.length === 0 : selectedIdx >= trades.length - 1}
          onClick={() => onSelect(selectedIdx != null ? selectedIdx + 1 : 0)}
          className="rounded p-0.5 text-gray-400 hover:bg-gray-800 hover:text-gray-200 disabled:opacity-30 disabled:hover:bg-transparent"
        >
          <ChevronRight className="h-3.5 w-3.5" />
        </button>

        {t && (
          <>
            <div className="h-4 w-px bg-gray-800" />
            <Badge className={`text-[10px] ${t.side === "long" ? "bg-green-900/50 text-green-300" : "bg-red-900/50 text-red-300"}`}>
              {t.side}
            </Badge>
            <span className={`font-mono text-sm font-semibold ${t.net_pnl >= 0 ? "text-green-400" : "text-red-400"}`}>
              {t.net_pnl >= 0 ? "+" : ""}{t.net_pnl.toFixed(2)}
              <span className="ml-1 text-[10px] opacity-70">
                ({t.size ? ((t.net_pnl / t.size * 100).toFixed(2)) : "?"}%)
              </span>
            </span>
            <span className="text-[10px] text-gray-500">
              ({t.gross_pnl >= 0 ? "+" : ""}{t.gross_pnl.toFixed(2)} gross, −{t.fees.toFixed(2)} fees
              {t.funding_cost != null && t.funding_cost !== 0 && (
                <>, −{Math.abs(t.funding_cost).toFixed(2)} funding</>
              )})
            </span>
          </>
        )}

        {selectedIdx == null && trades.length > 0 && (
          <button
            onClick={() => onSelect(0)}
            className="ml-1 rounded px-2 py-0.5 text-[10px] text-blue-400 hover:bg-gray-800 transition"
          >
            Jump to first →
          </button>
        )}
      </div>

      {/* Detail row: timestamps, prices, duration, size */}
      {t && (
        <div className="grid grid-cols-[1fr_auto_1fr] gap-x-4 gap-y-0 border-t border-gray-800 px-3 py-2">
          {/* Entry column */}
          <div>
            <p className="text-[10px] font-medium text-green-500">ENTRY</p>
            <p className="font-mono text-[11px] text-gray-300">{formatTimestamp(t.entry_time)}</p>
            <p className="font-mono text-xs text-gray-200">{t.entry_price.toFixed(2)}</p>
            <p className="text-[10px] text-gray-500">{t.entry_reason}</p>
          </div>

          {/* Center: duration + size */}
          <div className="flex flex-col items-center justify-center gap-0.5 px-4">
            <span className="font-mono text-xs font-semibold text-gray-300">{formatDuration(heldSeconds)}</span>
            <span className="text-[10px] text-gray-500">{t.bars_held} bars</span>
            <span className="text-[10px] text-gray-400">Size: {t.size}</span>
          </div>

          {/* Exit column */}
          <div className="text-right">
            <p className="text-[10px] font-medium text-red-500">EXIT</p>
            <p className="font-mono text-[11px] text-gray-300">{formatTimestamp(t.exit_time)}</p>
            <p className="font-mono text-xs text-gray-200">{t.exit_price.toFixed(2)}</p>
            <p className="text-[10px] text-gray-500">{t.exit_reason}</p>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Backtest Charts sub-component ──

type IndicatorSeries = BacktestViewData["indicators"][number];

const IND_COLORS = ["#38bdf8", "#fb923c", "#a78bfa", "#4ade80", "#f472b6"];

function BacktestCharts({
  chartData,
  indicators,
  trades,
  selectedTradeIdx,
}: {
  chartData: BacktestViewData["chart_data"];
  indicators: IndicatorSeries[];
  trades: TradeRow[];
  selectedTradeIdx: number | null;
}) {
  const mainRef = useRef<HTMLDivElement>(null);
  const volumeRef = useRef<HTMLDivElement>(null);
  const legendRef = useRef<HTMLDivElement>(null);
  const chartsRef = useRef<IChartApi[]>([]);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [scaleLabel, setScaleLabel] = useState("");

  const updateScaleLabel = useCallback((from: number, to: number) => {
    const span = to - from;
    const hours = span / 3600;
    const days = hours / 24;
    if (days >= 365) setScaleLabel(`${(days / 365).toFixed(1)}y`);
    else if (days >= 30) setScaleLabel(`${Math.round(days / 30)}M`);
    else if (days >= 1) setScaleLabel(`${Math.round(days)}d`);
    else setScaleLabel(`${Math.round(hours)}h`);
  }, []);

  const zoomPreset = useCallback((seconds: number) => {
    const chart = chartsRef.current[0];
    if (!chart) return;
    const ts = chart.timeScale();
    const range = ts.getVisibleRange();
    if (!range) return;
    const mid = ((range.to as number) + (range.from as number)) / 2;
    const half = seconds / 2;
    ts.setVisibleRange({ from: (mid - half) as never, to: (mid + half) as never });
  }, []);

  const zoomBy = useCallback((factor: number) => {
    const chart = chartsRef.current[0];
    if (!chart) return;
    const ts = chart.timeScale();
    const range = ts.getVisibleRange();
    if (!range) return;
    const mid = ((range.to as number) + (range.from as number)) / 2;
    const half = ((range.to as number) - (range.from as number)) / 2 * factor;
    ts.setVisibleRange({ from: (mid - half) as never, to: (mid + half) as never });
  }, []);

  const zoomFit = useCallback(() => {
    chartsRef.current.forEach((c) => c.timeScale().fitContent());
  }, []);

  const toggleVisibility = (key: string) => {
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  // Split indicators into overlays (rendered on price pane) and panels (separate panes)
  const overlayIndicators = useMemo(
    () => indicators.filter((ind) => ind.display === "overlay" && ind.series.length > 0),
    [indicators],
  );
  const panelIndicators = useMemo(
    () => indicators.filter((ind) => ind.display === "panel" && ind.series.length > 0),
    [indicators],
  );

  // Legend items
  const legendItems = useMemo(() => {
    const items: { key: string; label: string; color: string }[] = [];
    overlayIndicators.forEach((ind, i) => {
      const color = (ind.render as { color?: string })?.color ?? IND_COLORS[i % IND_COLORS.length];
      items.push({ key: ind.name, label: ind.name, color });
    });
    panelIndicators.forEach((ind, i) => {
      items.push({ key: ind.name, label: ind.name, color: IND_COLORS[i % IND_COLORS.length] });
    });
    return items;
  }, [overlayIndicators, panelIndicators]);

  // Generate position state data from trades (for overlay histogram)
  // Highlight the selected trade brighter, dim the rest
  const positionData = useMemo(() => {
    if (!trades.length || !chartData.ohlcv?.length) return [];
    const tradeTimestamps = trades.map((t, i) => ({
      idx: i,
      entry: Math.floor(new Date(t.entry_time).getTime() / 1000),
      exit: Math.floor(new Date(t.exit_time).getTime() / 1000),
      side: t.side,
      win: t.net_pnl >= 0,
    }));
    return chartData.ohlcv.map((bar) => {
      const active = tradeTimestamps.find(
        (tt) => bar.time >= tt.entry && bar.time <= tt.exit,
      );
      if (!active) return { time: bar.time, value: 0, color: "transparent" };
      const val = active.side === "long" ? 1 : -1;
      const isSelected = selectedTradeIdx != null && active.idx === selectedTradeIdx;
      const hasSel = selectedTradeIdx != null;
      let color: string;
      if (isSelected) {
        // Bright highlight for selected trade
        color = active.win ? "rgba(34, 197, 94, 0.9)" : "rgba(239, 68, 68, 0.9)";
      } else if (hasSel) {
        // Dim non-selected trades when one is selected
        color = active.win ? "rgba(34, 197, 94, 0.15)" : "rgba(239, 68, 68, 0.15)";
      } else {
        // Normal opacity when nothing selected
        color = active.win ? "rgba(34, 197, 94, 0.5)" : "rgba(239, 68, 68, 0.5)";
      }
      return { time: bar.time, value: val, color };
    });
  }, [trades, chartData.ohlcv, selectedTradeIdx]);

  // Compute main chart height: base + pane per visible panel indicator + position pane
  const visiblePanels = panelIndicators.filter((ind) => !hidden.has(ind.name));
  const hasPositionPane = positionData.length > 0;
  const mainChartHeight = 300 + visiblePanels.length * 120 + (hasPositionPane ? 60 : 0);

  useEffect(() => {
    chartsRef.current.forEach((c) => c.remove());
    chartsRef.current = [];

    const chartOptions = {
      layout: {
        background: { type: ColorType.Solid as const, color: "#111827" },
        textColor: "#9ca3af",
      },
      grid: {
        vertLines: { color: "#1f2937" },
        horzLines: { color: "#1f2937" },
      },
      crosshair: { mode: 0 },
      rightPriceScale: { borderColor: "#374151" },
      timeScale: { borderColor: "#374151", timeVisible: true },
    };

    // Track series for crosshair legend
    const namedSeries: { name: string; color: string; series: unknown; type: "candle" | "line" | "dots" }[] = [];

    // ── Main chart: Price (pane 0) + overlay indicators + panel indicator panes ──
    const hasOhlcv = chartData.ohlcv && chartData.ohlcv.length > 0;
    const hasPrice = chartData.price && chartData.price.length > 0;

    if (mainRef.current && (hasOhlcv || hasPrice)) {
      const chart = createChart(mainRef.current, {
        ...chartOptions,
        height: mainChartHeight,
        autoSize: true,
      });
      chartsRef.current.push(chart);

      // Pane 0: Candlestick price (prefer OHLCV, fallback to line)
      let priceSeries;
      if (hasOhlcv) {
        priceSeries = chart.addSeries(CandlestickSeries, {
          upColor: "#22c55e",
          downColor: "#ef4444",
          borderDownColor: "#ef4444",
          borderUpColor: "#22c55e",
          wickDownColor: "#ef4444",
          wickUpColor: "#22c55e",
          lastValueVisible: false,
          priceLineVisible: false,
        });
        priceSeries.setData(chartData.ohlcv as never[]);
        namedSeries.push({ name: "Price", color: "#9ca3af", series: priceSeries, type: "candle" });
      } else {
        priceSeries = chart.addSeries(LineSeries, {
          color: "#9ca3af",
          lineWidth: 1,
          title: "Price",
        });
        priceSeries.setData(chartData.price as never[]);
        namedSeries.push({ name: "Price", color: "#9ca3af", series: priceSeries, type: "line" });
      }

      // Pane 0: Trade markers on price series
      if (chartData.markers?.length) {
        const sorted = [...chartData.markers]
          .sort((a, b) => a.time - b.time)
          .map((m) => ({
            time: m.time as never,
            position: m.position as never,
            color: m.color,
            shape: m.shape as never,
            text: m.text ?? "",
          }));
        createSeriesMarkers(priceSeries, sorted);
      }

      // Pane 0: Overlay indicators (PSAR dots, moving averages, etc.)
      overlayIndicators.forEach((ind, idx) => {
        if (hidden.has(ind.name)) return;
        const renderType = (ind.render as { type?: string })?.type ?? "line";

        if (renderType === "markers") {
          // PSAR-style dots
          const dotColor = (ind.render as { color?: string })?.color ?? "#f59e0b";
          const s = chart.addCustomSeries(new DotsSeries(), {
            dotColor,
            radius: 2,
            lastValueVisible: false,
            priceLineVisible: false,
            title: ind.name,
          });
          s.setData(ind.series as never[]);
          namedSeries.push({ name: ind.name, color: dotColor, series: s, type: "dots" });
        } else {
          // Line overlay
          const color = IND_COLORS[idx % IND_COLORS.length];
          const s = chart.addSeries(LineSeries, { color, lineWidth: 1, title: ind.name, lastValueVisible: false, priceLineVisible: false });
          s.setData(ind.series as never[]);
          namedSeries.push({ name: ind.name, color, series: s, type: "line" });
        }
      });

      // Pane 1+: Panel indicators (SOBV, RSI, etc.) — one pane each
      let nextPane = 1;
      panelIndicators.forEach((ind, idx) => {
        if (hidden.has(ind.name)) return;
        const pane = nextPane++;
        const color = IND_COLORS[idx % IND_COLORS.length];
        const s = chart.addSeries(LineSeries, { color, lineWidth: 2, title: ind.name, lastValueVisible: false, priceLineVisible: false }, pane);
        s.setData(ind.series as never[]);
        namedSeries.push({ name: ind.name, color, series: s, type: "line" });
      });

      // Position state histogram pane (long/short/flat, win/loss coloring)
      if (positionData.length > 0) {
        const posPaneIdx = nextPane++;
        const posSeries = chart.addSeries(HistogramSeries, {
          title: "Position",
          priceFormat: { type: "custom", formatter: (v: number) => (v > 0 ? "LONG" : v < 0 ? "SHORT" : "FLAT") },
          lastValueVisible: false,
          priceLineVisible: false,
        }, posPaneIdx);
        posSeries.setData(positionData as never[]);
      }

      // ── Crosshair legend ──
      chart.subscribeCrosshairMove((param) => {
        const legend = legendRef.current;
        if (!legend) return;
        if (!param.time) {
          legend.style.display = "none";
          return;
        }
        legend.style.display = "flex";

        // Format timestamp
        const ts = param.time as number;
        const d = new Date(ts * 1000);
        const timeStr = d.toLocaleString("en-GB", {
          year: "numeric", month: "short", day: "2-digit",
          hour: "2-digit", minute: "2-digit",
          hour12: false,
        });

        let html = `<span style="color:#9ca3af;margin-right:12px">${timeStr}</span>`;

        for (const ns of namedSeries) {
          const data = param.seriesData.get(ns.series as never);
          if (!data) continue;
          const rec = data as unknown as Record<string, unknown>;
          if (ns.type === "candle" && "close" in rec) {
            const o = (rec.open as number)?.toFixed(2);
            const h = (rec.high as number)?.toFixed(2);
            const l = (rec.low as number)?.toFixed(2);
            const c = (rec.close as number)?.toFixed(2);
            html += `<span style="color:${ns.color};margin-right:8px">O <b>${o}</b> H <b>${h}</b> L <b>${l}</b> C <b>${c}</b></span>`;
          } else if ("value" in rec) {
            const v = rec.value as number;
            if (v != null && !isNaN(v)) {
              html += `<span style="color:${ns.color};margin-right:8px">${ns.name}: <b>${v.toFixed(4)}</b></span>`;
            }
          }
        }
        legend.innerHTML = html;
      });

      chart.timeScale().fitContent();

      // Track scale label from visible range changes
      chart.timeScale().subscribeVisibleTimeRangeChange((range) => {
        if (!range) return;
        updateScaleLabel(range.from as number, range.to as number);
      });
      // Set initial label
      const initRange = chart.timeScale().getVisibleRange();
      if (initRange) updateScaleLabel(initRange.from as number, initRange.to as number);
    }

    // ── Volume chart (separate, synced) ──
    if (volumeRef.current && chartData.volume && chartData.volume.length > 0) {
      const volChart = createChart(volumeRef.current, {
        ...chartOptions,
        height: 80,
        autoSize: true,
      });
      chartsRef.current.push(volChart);

      const volSeries = volChart.addSeries(HistogramSeries, {
        priceFormat: { type: "volume" },
        priceScaleId: "",
        lastValueVisible: false,
        priceLineVisible: false,
      });
      volSeries.setData(chartData.volume as never[]);
      volChart.timeScale().fitContent();
    }

    // ── Sync all charts ──
    const charts = chartsRef.current;
    let syncSource: number | null = null;
    const releaseSyncSource = () => { syncSource = null; };
    window.addEventListener("mouseup", releaseSyncSource);
    window.addEventListener("touchend", releaseSyncSource);

    charts.forEach((chart, i) => {
      chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
        if (!range) return;
        if (syncSource !== null && syncSource !== i) return;
        syncSource = i;
        charts.forEach((other, j) => {
          if (i !== j) other.timeScale().setVisibleLogicalRange(range);
        });
      });
    });

    return () => {
      window.removeEventListener("mouseup", releaseSyncSource);
      window.removeEventListener("touchend", releaseSyncSource);
      chartsRef.current.forEach((c) => c.remove());
      chartsRef.current = [];
    };
  }, [chartData, overlayIndicators, panelIndicators, hidden, mainChartHeight, positionData]);

  // ── Scroll to selected trade (preserve user's zoom level) ──
  useEffect(() => {
    if (selectedTradeIdx == null || !trades[selectedTradeIdx]) return;
    const t = trades[selectedTradeIdx];
    const entryTs = Math.floor(new Date(t.entry_time).getTime() / 1000);
    const exitTs = Math.floor(new Date(t.exit_time).getTime() / 1000);
    const midTs = (entryTs + exitTs) / 2;

    chartsRef.current.forEach((chart) => {
      const ts = chart.timeScale();
      const range = ts.getVisibleRange();
      if (!range) return;
      const halfSpan = ((range.to as number) - (range.from as number)) / 2;
      ts.setVisibleRange({
        from: (midTs - halfSpan) as never,
        to: (midTs + halfSpan) as never,
      });
    });
  }, [selectedTradeIdx, trades]);

  return (
    <div className="space-y-1">
      {/* Legend / visibility toggles */}
      {legendItems.length > 0 && (
        <div className="flex flex-wrap gap-1.5 px-1 py-1.5">
          {legendItems.map((item) => {
            const isHidden = hidden.has(item.key);
            return (
              <button
                key={item.key}
                onClick={() => toggleVisibility(item.key)}
                className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[11px] font-medium transition-all ${
                  isHidden
                    ? "bg-gray-800/40 text-gray-600 line-through"
                    : "bg-gray-800 text-gray-300 hover:bg-gray-700"
                }`}
              >
                <span
                  className="inline-block h-2 w-2 rounded-full"
                  style={{ backgroundColor: isHidden ? "#4b5563" : item.color }}
                />
                {item.label}
              </button>
            );
          })}
        </div>
      )}

      {/* Zoom controls */}
      <div className="flex items-center gap-1 px-1 pb-1">
        <span className="mr-1 min-w-[32px] text-center font-mono text-[10px] font-semibold text-gray-400">{scaleLabel}</span>
        <button onClick={() => zoomBy(0.5)} className="rounded p-1 text-gray-500 hover:bg-gray-800 hover:text-gray-300 transition" title="Zoom in">
          <ZoomIn className="h-3.5 w-3.5" />
        </button>
        <button onClick={() => zoomBy(2)} className="rounded p-1 text-gray-500 hover:bg-gray-800 hover:text-gray-300 transition" title="Zoom out">
          <ZoomOut className="h-3.5 w-3.5" />
        </button>
        <div className="mx-1 h-3 w-px bg-gray-800" />
        {[
          { label: "1D", secs: 86400 },
          { label: "3D", secs: 86400 * 3 },
          { label: "1W", secs: 86400 * 7 },
          { label: "1M", secs: 86400 * 30 },
          { label: "3M", secs: 86400 * 90 },
        ].map((p) => (
          <button
            key={p.label}
            onClick={() => zoomPreset(p.secs)}
            className="rounded px-1.5 py-0.5 text-[10px] font-medium text-gray-500 hover:bg-gray-800 hover:text-gray-300 transition"
          >
            {p.label}
          </button>
        ))}
        <button onClick={zoomFit} className="rounded p-1 text-gray-500 hover:bg-gray-800 hover:text-gray-300 transition" title="Fit all">
          <Maximize2 className="h-3.5 w-3.5" />
        </button>
      </div>

      {/* Crosshair legend overlay */}
      <div className="relative">
        <div
          ref={legendRef}
          className="pointer-events-none absolute left-2 top-1 z-10 flex flex-wrap gap-x-1 gap-y-0 rounded bg-gray-900/80 px-2 py-1 text-[11px] font-mono backdrop-blur-sm"
          style={{ display: "none" }}
        />
        <div ref={mainRef} className="rounded bg-gray-900" style={{ height: mainChartHeight }} />
      </div>

      {chartData.volume && chartData.volume.length > 0 && (
        <div ref={volumeRef} className="rounded bg-gray-900" style={{ height: 80 }} />
      )}

    </div>
  );
}

// ── Trades Table with expandable rows (Trade Inspector) ──

function TradesTable({
  trades,
  selectedTradeIdx,
  onTradeSelect,
}: {
  trades: TradeRow[];
  selectedTradeIdx: number | null;
  onTradeSelect: (idx: number) => void;
}) {
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead className="bg-gray-800/50">
          <tr>
            <th className="w-6 px-1 py-1.5" />
            <th className="w-8 px-1 py-1.5" />
            {["Side", "Entry", "Entry Price", "Exit", "Exit Price", "Size", "Net PnL", "Bars", "Reason"].map((h) => (
              <th key={h} className="px-2 py-1.5 text-left font-medium text-gray-400">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-800">
          {trades.map((t, i) => {
            const isExpanded = expandedIdx === i;
            const isSelected = selectedTradeIdx === i;
            const hasContext = !!t.metadata?.entry_context || !!t.metadata?.exit_context;
            return (
              <TradeRowWithInspector
                key={i}
                trade={t}
                index={i}
                isExpanded={isExpanded}
                isSelected={isSelected}
                hasContext={hasContext}
                onToggle={() => setExpandedIdx(isExpanded ? null : i)}
                onZoom={() => onTradeSelect(i)}
              />
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function TradeRowWithInspector({
  trade: t,
  index,
  isExpanded,
  isSelected,
  hasContext,
  onToggle,
  onZoom,
}: {
  trade: TradeRow;
  index: number;
  isExpanded: boolean;
  isSelected: boolean;
  hasContext: boolean;
  onToggle: () => void;
  onZoom: () => void;
}) {
  return (
    <>
      <tr
        className={`cursor-pointer transition ${
          isSelected
            ? "bg-blue-900/20"
            : isExpanded
            ? "bg-gray-800/70"
            : "hover:bg-gray-800/50"
        }`}
        onClick={onToggle}
      >
        <td className="px-1 py-1 text-center text-gray-500">
          {hasContext ? (
            isExpanded ? <ChevronDown className="inline h-3 w-3" /> : <ChevronRight className="inline h-3 w-3" />
          ) : null}
        </td>
        <td className="px-1 py-1 text-center">
          <button
            onClick={(e) => { e.stopPropagation(); onZoom(); }}
            className={`rounded p-0.5 transition ${
              isSelected ? "text-blue-400" : "text-gray-600 hover:text-blue-400"
            }`}
            title="Zoom to trade on chart"
          >
            <Crosshair className="h-3 w-3" />
          </button>
        </td>
        <td className="px-2 py-1">
          <Badge className={t.side === "long" ? "bg-green-900/50 text-green-300" : "bg-red-900/50 text-red-300"}>
            {t.side}
          </Badge>
        </td>
        <td className="px-2 py-1 font-mono text-gray-400">{t.entry_time.slice(0, 16)}</td>
        <td className="px-2 py-1 text-right font-mono text-gray-300">{t.entry_price.toFixed(2)}</td>
        <td className="px-2 py-1 font-mono text-gray-400">{t.exit_time.slice(0, 16)}</td>
        <td className="px-2 py-1 text-right font-mono text-gray-300">{t.exit_price.toFixed(2)}</td>
        <td className="px-2 py-1 text-right font-mono text-gray-400">{t.size}</td>
        <td className={`px-2 py-1 text-right font-mono ${t.net_pnl >= 0 ? "text-green-400" : "text-red-400"}`}>
          {t.net_pnl.toFixed(2)}
        </td>
        <td className="px-2 py-1 text-right text-gray-400">{t.bars_held}</td>
        <td className="px-2 py-1 text-gray-500">{t.entry_reason} → {t.exit_reason}</td>
      </tr>
      {isExpanded && hasContext && (
        <tr>
          <td colSpan={11} className="bg-gray-900/80 px-4 py-3">
            <ContextPanel metadata={t.metadata!} tradeIndex={index} />
          </td>
        </tr>
      )}
    </>
  );
}

// ── Decision Context Panel ──

function ContextPanel({
  metadata,
  tradeIndex: _tradeIndex,
}: {
  metadata: NonNullable<TradeRow["metadata"]>;
  tradeIndex: number;
}) {
  const { entry_context, exit_context } = metadata;

  const intervals = Array.from(
    new Set([
      ...Object.keys(entry_context || {}),
      ...Object.keys(exit_context || {}),
    ])
  ).sort();

  const ohlcvKeys = new Set(["open_time", "open", "high", "low", "close", "volume"]);

  return (
    <div className="space-y-3">
      <p className="text-[11px] font-semibold text-gray-400">Decision Context</p>
      <div className="grid grid-cols-2 gap-4">
        <ContextSide label="Entry" context={entry_context} intervals={intervals} ohlcvKeys={ohlcvKeys} />
        <ContextSide label="Exit" context={exit_context} intervals={intervals} ohlcvKeys={ohlcvKeys} />
      </div>
    </div>
  );
}

function ContextSide({
  label,
  context,
  intervals,
  ohlcvKeys,
}: {
  label: string;
  context?: Record<string, Record<string, unknown> | null>;
  intervals: string[];
  ohlcvKeys: Set<string>;
}) {
  if (!context) {
    return (
      <div>
        <p className="mb-1 text-[11px] font-medium text-gray-500">{label}</p>
        <p className="text-[10px] text-gray-600">No context captured</p>
      </div>
    );
  }

  return (
    <div>
      <p className="mb-2 text-[11px] font-medium text-blue-400">{label}</p>
      {intervals.map((interval) => {
        const barData = context[interval];
        if (!barData) return null;

        const price = barData.close as number | undefined;
        const openTime = barData.open_time as string | undefined;
        const indicators = Object.entries(barData).filter(
          ([k]) => !ohlcvKeys.has(k)
        );

        return (
          <div key={interval} className="mb-2">
            <div className="flex items-center gap-2">
              <Badge variant="secondary" className="h-4 px-1.5 text-[10px]">
                {interval}
              </Badge>
              {price != null && (
                <span className="font-mono text-[10px] text-gray-400">
                  {Number(price).toFixed(2)}
                </span>
              )}
              {openTime && (
                <span className="text-[10px] text-gray-600">
                  {String(openTime).slice(0, 16)}
                </span>
              )}
            </div>
            {indicators.length > 0 && (
              <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 pl-1">
                {indicators.map(([key, val]) => (
                  <span key={key} className="text-[10px]">
                    <span className="text-gray-500">{key}:</span>{" "}
                    <span className="font-mono text-gray-300">
                      {typeof val === "number" ? val.toFixed(4) : String(val)}
                    </span>
                  </span>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
