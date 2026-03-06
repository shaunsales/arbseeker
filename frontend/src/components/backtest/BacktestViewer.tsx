import { useState } from "react";
import { useEffect, useRef } from "react";
import {
  createChart,
  createSeriesMarkers,
  type IChartApi,
  ColorType,
  LineSeries,
} from "lightweight-charts";
import type { BacktestViewData } from "@/api/backtest";
import { Badge } from "@/components/ui/badge";

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

      {tab === "charts" && <BacktestCharts chartData={data.chart_data} />}

      {tab === "trades" && (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="bg-gray-800/50">
              <tr>
                {[
                  "Side",
                  "Entry",
                  "Entry Price",
                  "Exit",
                  "Exit Price",
                  "Size",
                  "Net PnL",
                  "Bars",
                  "Reason",
                ].map((h) => (
                  <th
                    key={h}
                    className="px-2 py-1.5 text-left font-medium text-gray-400"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {data.trades.map((t, i) => (
                <tr key={i} className="hover:bg-gray-800/50">
                  <td className="px-2 py-1">
                    <Badge
                      className={
                        t.side === "long"
                          ? "bg-green-900/50 text-green-300"
                          : "bg-red-900/50 text-red-300"
                      }
                    >
                      {t.side}
                    </Badge>
                  </td>
                  <td className="px-2 py-1 font-mono text-gray-400">
                    {t.entry_time.slice(0, 16)}
                  </td>
                  <td className="px-2 py-1 text-right font-mono text-gray-300">
                    {t.entry_price.toFixed(2)}
                  </td>
                  <td className="px-2 py-1 font-mono text-gray-400">
                    {t.exit_time.slice(0, 16)}
                  </td>
                  <td className="px-2 py-1 text-right font-mono text-gray-300">
                    {t.exit_price.toFixed(2)}
                  </td>
                  <td className="px-2 py-1 text-right font-mono text-gray-400">
                    {t.size}
                  </td>
                  <td
                    className={`px-2 py-1 text-right font-mono ${
                      t.net_pnl >= 0 ? "text-green-400" : "text-red-400"
                    }`}
                  >
                    {t.net_pnl.toFixed(2)}
                  </td>
                  <td className="px-2 py-1 text-right text-gray-400">
                    {t.bars_held}
                  </td>
                  <td className="px-2 py-1 text-gray-500">
                    {t.entry_reason} → {t.exit_reason}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ── Backtest Charts sub-component ──

function BacktestCharts({
  chartData,
}: {
  chartData: BacktestViewData["chart_data"];
}) {
  const priceRef = useRef<HTMLDivElement>(null);
  const equityRef = useRef<HTMLDivElement>(null);
  const drawdownRef = useRef<HTMLDivElement>(null);
  const chartsRef = useRef<IChartApi[]>([]);

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

    // Price chart with trade markers
    if (priceRef.current && chartData.price?.length) {
      const chart = createChart(priceRef.current, {
        ...chartOptions,
        height: 300,
      });
      chartsRef.current.push(chart);

      const s = chart.addSeries(LineSeries, {
        color: "#9ca3af",
        lineWidth: 1,
        title: "Price",
      });
      s.setData(chartData.price as never[]);

      // Trade markers — shapes only (text would overlap with 100+ trades)
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
        createSeriesMarkers(s, sorted);
      }
      chart.timeScale().fitContent();
    }

    // Equity curve
    if (equityRef.current && chartData.equity?.length) {
      const chart = createChart(equityRef.current, {
        ...chartOptions,
        height: 200,
      });
      chartsRef.current.push(chart);

      const s = chart.addSeries(LineSeries, {
        color: "#3b82f6",
        lineWidth: 2,
        title: "Equity",
      });
      s.setData(chartData.equity as never[]);
      chart.timeScale().fitContent();
    }

    // Drawdown
    if (drawdownRef.current && chartData.drawdown?.length) {
      const chart = createChart(drawdownRef.current, {
        ...chartOptions,
        height: 120,
      });
      chartsRef.current.push(chart);

      const s = chart.addSeries(LineSeries, {
        color: "#ef4444",
        lineWidth: 1,
        title: "Drawdown %",
      });
      s.setData(chartData.drawdown as never[]);
      chart.timeScale().fitContent();
    }

    // Sync all chart time scales
    const charts = chartsRef.current;
    for (let i = 0; i < charts.length; i++) {
      for (let j = 0; j < charts.length; j++) {
        if (i === j) continue;
        const src = charts[i];
        const dst = charts[j];
        src.timeScale().subscribeVisibleLogicalRangeChange((range) => {
          if (range) dst.timeScale().setVisibleLogicalRange(range);
        });
      }
    }

    return () => {
      chartsRef.current.forEach((c) => c.remove());
      chartsRef.current = [];
    };
  }, [chartData]);

  return (
    <div className="space-y-1">
      <p className="text-[11px] font-medium text-gray-500">Price + Trades</p>
      <div ref={priceRef} className="rounded bg-gray-900" />
      <p className="mt-2 text-[11px] font-medium text-gray-500">Equity Curve</p>
      <div ref={equityRef} className="rounded bg-gray-900" />
      <p className="mt-2 text-[11px] font-medium text-gray-500">Drawdown</p>
      <div ref={drawdownRef} className="rounded bg-gray-900" />
    </div>
  );
}
