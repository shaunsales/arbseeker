import { useState } from "react";
import { useEffect, useRef } from "react";
import {
  createChart,
  type IChartApi,
  ColorType,
  LineSeries,
} from "lightweight-charts";
import type { BacktestViewData } from "@/api/backtest";
import { Badge } from "@/components/ui/badge";

interface Props {
  data: BacktestViewData;
}

const METRIC_LABELS: Record<string, string> = {
  total_return_pct: "Total Return",
  sharpe_ratio: "Sharpe",
  max_drawdown_pct: "Max Drawdown",
  win_rate: "Win Rate",
  total_trades: "Trades",
  profit_factor: "Profit Factor",
  avg_trade_pnl: "Avg Trade PnL",
  avg_bars_held: "Avg Bars Held",
};

export default function BacktestViewer({ data }: Props) {
  const [tab, setTab] = useState<"charts" | "trades">("charts");

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center gap-3">
        <h2 className="text-sm font-semibold text-gray-200">
          <span className="text-blue-400">{data.strategy_name}</span>
        </h2>
        <Badge variant="secondary" className="text-[10px]">
          {data.run_id.slice(0, 12)}
        </Badge>
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
      <div className="grid grid-cols-4 gap-3">
        {Object.entries(data.meta).map(([key, val]) => {
          const label = METRIC_LABELS[key] || key;
          if (typeof val !== "number" && typeof val !== "string") return null;
          const isReturn = key.includes("return");
          const isDrawdown = key.includes("drawdown");
          let color = "text-gray-200";
          if (isReturn) color = (val as number) >= 0 ? "text-green-400" : "text-red-400";
          if (isDrawdown) color = "text-red-400";

          return (
            <div
              key={key}
              className="rounded border border-gray-800 bg-gray-900 p-3"
            >
              <p className="text-xs text-gray-500">{label}</p>
              <p className={`text-lg font-semibold ${color}`}>
                {typeof val === "number"
                  ? key.includes("pct") || key.includes("rate")
                    ? `${val.toFixed(2)}%`
                    : val.toFixed(2)
                  : val}
              </p>
            </div>
          );
        })}
      </div>

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

    // Equity curve
    if (equityRef.current && chartData.equity) {
      const chart = createChart(equityRef.current, {
        ...chartOptions,
        height: 250,
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
    if (drawdownRef.current && chartData.drawdown) {
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

    // Sync
    if (chartsRef.current.length === 2) {
      const [eq, dd] = chartsRef.current;
      eq.timeScale().subscribeVisibleLogicalRangeChange((range) => {
        if (range) dd.timeScale().setVisibleLogicalRange(range);
      });
      dd.timeScale().subscribeVisibleLogicalRangeChange((range) => {
        if (range) eq.timeScale().setVisibleLogicalRange(range);
      });
    }

    return () => {
      chartsRef.current.forEach((c) => c.remove());
      chartsRef.current = [];
    };
  }, [chartData]);

  return (
    <div className="space-y-1">
      <div ref={equityRef} className="rounded bg-gray-900" />
      <div ref={drawdownRef} className="rounded bg-gray-900" />
    </div>
  );
}
