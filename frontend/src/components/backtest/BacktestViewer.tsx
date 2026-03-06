import { useState, useEffect, useRef, useMemo } from "react";
import {
  createChart,
  createSeriesMarkers,
  type IChartApi,
  ColorType,
  LineSeries,
  AreaSeries,
} from "lightweight-charts";
import type { BacktestViewData } from "@/api/backtest";
import { Badge } from "@/components/ui/badge";
import { ChevronRight, ChevronDown } from "lucide-react";

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
  const [tab, setTab] = useState<"charts" | "trades">("charts");
  const metrics = (data.meta as Record<string, unknown>).metrics as
    | Record<string, unknown>
    | undefined;
  const period = metrics?.period as string | undefined;

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

      {/* Metrics */}
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

      {/* Tabs */}
      <div className="flex border-b border-gray-800">
        {(["charts", "trades"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`border-b-2 px-3 py-1.5 text-xs font-medium transition ${
              tab === t
                ? "border-blue-500 text-blue-400"
                : "border-transparent text-gray-500 hover:text-gray-300"
            }`}
          >
            {t === "charts" ? "Charts" : `Trades (${data.trades.length})`}
          </button>
        ))}
      </div>

      {tab === "charts" && <BacktestCharts chartData={data.chart_data} indicators={data.indicators ?? []} />}

      {tab === "trades" && <TradesTable trades={data.trades} />}
    </div>
  );
}

// ── Backtest Charts sub-component ──

type IndicatorSeries = BacktestViewData["indicators"][number];

const IND_COLORS = ["#38bdf8", "#fb923c", "#a78bfa", "#4ade80", "#f472b6"];

function BacktestCharts({
  chartData,
  indicators,
}: {
  chartData: BacktestViewData["chart_data"];
  indicators: IndicatorSeries[];
}) {
  const priceRef = useRef<HTMLDivElement>(null);
  const equityRef = useRef<HTMLDivElement>(null);
  const indRefs = useRef<Map<string, HTMLDivElement>>(new Map());
  const chartsRef = useRef<IChartApi[]>([]);

  // Group related indicators into panels (ADX+DMP+DMN, MACD+MACDh+MACDs, etc.)
  const indGroups = useMemo(() => {
    const groups = new Map<string, IndicatorSeries[]>();
    for (const ind of indicators) {
      if (!ind.series.length) continue;
      const baseName = ind.name.replace(/_\d+$/, "");
      const groupKey =
        ["ADX", "DMP", "DMN"].includes(baseName) ? `${ind.interval}_ADX` :
        ["MACD", "MACDh", "MACDs"].includes(baseName) ? `${ind.interval}_MACD` :
        `${ind.interval}_${ind.name}`;
      if (!groups.has(groupKey)) groups.set(groupKey, []);
      groups.get(groupKey)!.push(ind);
    }
    return Array.from(groups.entries());
  }, [indicators]);

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
      timeScale: { borderColor: "#374151", timeVisible: true, minBarSpacing: 0.001 },
    };

    // Price chart with trade markers
    if (priceRef.current && chartData.price?.length) {
      const chart = createChart(priceRef.current, {
        ...chartOptions,
        height: 300,
      });
      chartsRef.current.push(chart);

      const priceSeries = chart.addSeries(LineSeries, {
        color: "#9ca3af",
        lineWidth: 1,
        title: "Price",
      });
      priceSeries.setData(chartData.price as never[]);

      if (chartData.markers?.length) {
        const sorted = [...chartData.markers]
          .sort((a, b) => a.time - b.time)
          .map((m) => ({
            time: m.time as never,
            position: m.position as never,
            color: m.color,
            shape: m.shape as never,
            text: "",
          }));
        createSeriesMarkers(priceSeries, sorted);
      }
      chart.timeScale().fitContent();
    }

    // Indicator panels — each group gets its own chart
    indGroups.forEach(([groupKey, group]: [string, IndicatorSeries[]]) => {
      const el = indRefs.current.get(groupKey);
      if (!el) return;

      const chart = createChart(el, {
        ...chartOptions,
        height: 150,
      });
      chartsRef.current.push(chart);

      group.forEach((ind: IndicatorSeries, idx: number) => {
        const s = chart.addSeries(LineSeries, {
          color: IND_COLORS[idx % IND_COLORS.length],
          lineWidth: 1,
          title: ind.name,
        });
        s.setData(ind.series as never[]);
      });
      chart.timeScale().fitContent();
    });

    // Equity + Drawdown on a single chart with dual Y-axes
    if (equityRef.current && chartData.equity?.length) {
      const chart = createChart(equityRef.current, {
        ...chartOptions,
        height: 200,
        rightPriceScale: { borderColor: "#374151" },
        leftPriceScale: { borderColor: "#374151", visible: true },
      });
      chartsRef.current.push(chart);

      const eqSeries = chart.addSeries(LineSeries, {
        color: "#3b82f6",
        lineWidth: 2,
        title: "Equity",
        priceScaleId: "right",
      });
      eqSeries.setData(chartData.equity as never[]);

      if (chartData.drawdown?.length) {
        const ddSeries = chart.addSeries(AreaSeries, {
          lineColor: "#ef4444",
          lineWidth: 1,
          topColor: "rgba(239, 68, 68, 0.0)",
          bottomColor: "rgba(239, 68, 68, 0.3)",
          title: "Drawdown %",
          priceScaleId: "left",
        });
        ddSeries.setData(chartData.drawdown as never[]);
      }
      chart.timeScale().fitContent();
    }

    // Sync: all series now share the same timestamps (LOCF-aligned),
    // so simple logical-range sync with source-locking works cleanly.
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
  }, [chartData, indGroups]);

  return (
    <div className="space-y-1">
      <p className="text-[11px] font-medium text-gray-500">Price + Trades</p>
      <div ref={priceRef} className="rounded bg-gray-900" />

      {indGroups.map(([groupKey, group]: [string, IndicatorSeries[]]) => (
        <div key={groupKey}>
          <p className="mt-2 text-[11px] font-medium text-gray-500">
            {group.map((i: IndicatorSeries) => i.name).join(" / ")} ({group[0].interval})
          </p>
          <div
            ref={(el) => { if (el) indRefs.current.set(groupKey, el); }}
            className="rounded bg-gray-900"
          />
        </div>
      ))}

      <p className="mt-2 text-[11px] font-medium text-gray-500">Equity / Drawdown</p>
      <div ref={equityRef} className="rounded bg-gray-900" />
    </div>
  );
}

// ── Trades Table with expandable rows (Trade Inspector) ──

type TradeRow = BacktestViewData["trades"][number];

function TradesTable({ trades }: { trades: TradeRow[] }) {
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead className="bg-gray-800/50">
          <tr>
            <th className="w-6 px-1 py-1.5" />
            {["Side", "Entry", "Entry Price", "Exit", "Exit Price", "Size", "Net PnL", "Bars", "Reason"].map((h) => (
              <th key={h} className="px-2 py-1.5 text-left font-medium text-gray-400">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-800">
          {trades.map((t, i) => {
            const isExpanded = expandedIdx === i;
            const hasContext = !!t.metadata?.entry_context || !!t.metadata?.exit_context;
            return (
              <TradeRowWithInspector
                key={i}
                trade={t}
                index={i}
                isExpanded={isExpanded}
                hasContext={hasContext}
                onToggle={() => setExpandedIdx(isExpanded ? null : i)}
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
  hasContext,
  onToggle,
}: {
  trade: TradeRow;
  index: number;
  isExpanded: boolean;
  hasContext: boolean;
  onToggle: () => void;
}) {
  return (
    <>
      <tr
        className={`cursor-pointer transition ${isExpanded ? "bg-gray-800/70" : "hover:bg-gray-800/50"}`}
        onClick={onToggle}
      >
        <td className="px-1 py-1 text-center text-gray-500">
          {hasContext ? (
            isExpanded ? <ChevronDown className="inline h-3 w-3" /> : <ChevronRight className="inline h-3 w-3" />
          ) : null}
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
          <td colSpan={10} className="bg-gray-900/80 px-4 py-3">
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
