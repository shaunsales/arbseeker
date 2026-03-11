import { useEffect, useRef } from "react";
import {
  createChart,
  type IChartApi,
  ColorType,
  CandlestickSeries,
  LineSeries,
  HistogramSeries,
} from "lightweight-charts";
import type { StrategyChartData, ComputeIndicatorsResponse } from "@/types/api";

interface Props {
  chartData: StrategyChartData;
  separateCols: string[];
  adHocData?: ComputeIndicatorsResponse | null;
}

const COLORS = [
  "#f59e0b", "#8b5cf6", "#ef4444", "#10b981", "#3b82f6",
  "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#84cc16",
];

// Second palette for ad-hoc indicators (brighter/distinct)
const ADHOC_COLORS = [
  "#38bdf8", "#fb923c", "#a78bfa", "#4ade80", "#f472b6",
  "#facc15", "#22d3ee", "#e879f9", "#34d399", "#fbbf24",
];

export default function StrategyChart({ chartData, separateCols, adHocData }: Props) {
  const priceRef = useRef<HTMLDivElement>(null);
  const indicatorRef = useRef<HTMLDivElement>(null);
  const volumeRef = useRef<HTMLDivElement>(null);
  const adHocPanelRef = useRef<HTMLDivElement>(null);
  const chartsRef = useRef<IChartApi[]>([]);

  const hasAdHocPanels = adHocData && Object.keys(adHocData.panels).length > 0;

  useEffect(() => {
    chartsRef.current.forEach((c) => c.remove());
    chartsRef.current = [];

    if (!chartData?.ohlcv?.length) return;

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

    // ── Price chart ──
    if (priceRef.current) {
      const chart = createChart(priceRef.current, {
        ...chartOptions,
        height: 300,
      });
      chartsRef.current.push(chart);

      const candleSeries = chart.addSeries(CandlestickSeries, {
        upColor: "#22c55e",
        downColor: "#ef4444",
        borderDownColor: "#ef4444",
        borderUpColor: "#22c55e",
        wickDownColor: "#ef4444",
        wickUpColor: "#22c55e",
      });
      candleSeries.setData(chartData.ohlcv as never[]);

      // Strategy overlays (from built data)
      let ci = 0;
      for (const [name, data] of Object.entries(chartData.overlays || {})) {
        const series = chart.addSeries(LineSeries, {
          color: COLORS[ci % COLORS.length],
          lineWidth: 1,
          title: name,
        });
        series.setData(data as never[]);
        ci++;
      }

      // Ad-hoc overlays (from indicator picker)
      if (adHocData?.overlays) {
        let aci = 0;
        for (const [name, data] of Object.entries(adHocData.overlays)) {
          const series = chart.addSeries(LineSeries, {
            color: ADHOC_COLORS[aci % ADHOC_COLORS.length],
            lineWidth: 2,
            title: name,
          });
          series.setData(data as never[]);
          aci++;
        }
      }

      chart.timeScale().fitContent();
    }

    // ── Strategy indicator chart (from built data) ──
    if (indicatorRef.current && separateCols.length > 0) {
      const chart = createChart(indicatorRef.current, {
        ...chartOptions,
        height: 150,
      });
      chartsRef.current.push(chart);

      let ci = 0;
      for (const [name, data] of Object.entries(chartData.indicators || {})) {
        const series = chart.addSeries(LineSeries, {
          color: COLORS[ci % COLORS.length],
          lineWidth: 1,
          title: name,
        });
        series.setData(data as never[]);
        ci++;
      }

      chart.timeScale().fitContent();
    }

    // ── Ad-hoc panel indicators ──
    if (adHocPanelRef.current && hasAdHocPanels) {
      const chart = createChart(adHocPanelRef.current, {
        ...chartOptions,
        height: 150,
      });
      chartsRef.current.push(chart);

      let aci = 0;
      for (const [name, data] of Object.entries(adHocData!.panels)) {
        const series = chart.addSeries(LineSeries, {
          color: ADHOC_COLORS[aci % ADHOC_COLORS.length],
          lineWidth: 2,
          title: name,
        });
        series.setData(data as never[]);
        aci++;
      }

      chart.timeScale().fitContent();
    }

    // ── Volume chart ──
    if (volumeRef.current && chartData.volume?.length) {
      const chart = createChart(volumeRef.current, {
        ...chartOptions,
        height: 80,
      });
      chartsRef.current.push(chart);

      const volSeries = chart.addSeries(HistogramSeries, {
        priceFormat: { type: "volume" },
        priceScaleId: "",
      });
      volSeries.setData(chartData.volume as never[]);

      chart.timeScale().fitContent();
    }

    // Sync time scales
    if (chartsRef.current.length > 1) {
      const primary = chartsRef.current[0];
      const rest = chartsRef.current.slice(1);

      primary.timeScale().subscribeVisibleLogicalRangeChange((range) => {
        if (range) {
          rest.forEach((c) => c.timeScale().setVisibleLogicalRange(range));
        }
      });

      rest.forEach((c) => {
        c.timeScale().subscribeVisibleLogicalRangeChange((range) => {
          if (range) {
            primary.timeScale().setVisibleLogicalRange(range);
            rest
              .filter((r) => r !== c)
              .forEach((r) => r.timeScale().setVisibleLogicalRange(range));
          }
        });
      });
    }

    return () => {
      chartsRef.current.forEach((c) => c.remove());
      chartsRef.current = [];
    };
  }, [chartData, separateCols, adHocData, hasAdHocPanels]);

  return (
    <div className="space-y-1">
      <div ref={priceRef} className="rounded bg-gray-900" />
      {separateCols.length > 0 && (
        <div ref={indicatorRef} className="rounded bg-gray-900" />
      )}
      {hasAdHocPanels && (
        <div ref={adHocPanelRef} className="rounded bg-gray-900" />
      )}
      {chartData?.volume?.length > 0 && (
        <div ref={volumeRef} className="rounded bg-gray-900" />
      )}
    </div>
  );
}
