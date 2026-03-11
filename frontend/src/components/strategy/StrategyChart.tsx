import { useEffect, useRef, useState, useMemo } from "react";
import {
  createChart,
  type IChartApi,
  type LineWidth,
  ColorType,
  CandlestickSeries,
  LineSeries,
  HistogramSeries,
} from "lightweight-charts";
import type {
  StrategyChartData,
  ComputeIndicatorsResponse,
  IndicatorResult,
  RenderSpec,
} from "@/types/api";

interface Props {
  chartData: StrategyChartData;
  separateCols: string[];
  adHocData?: ComputeIndicatorsResponse | null;
  expanded?: boolean;
}

const COLORS = [
  "#f59e0b", "#8b5cf6", "#3b82f6", "#ec4899", "#14b8a6",
  "#f97316", "#6366f1", "#38bdf8", "#e879f9", "#facc15",
];

const ADHOC_COLORS = [
  "#38bdf8", "#fb923c", "#a78bfa", "#f472b6", "#facc15",
  "#22d3ee", "#e879f9", "#f59e0b", "#818cf8", "#fbbf24",
];

// ── Render helpers ──

type TimeValue = { time: number; value: number };

function findCol(series: Record<string, TimeValue[]>, prefix: string): [string, TimeValue[]] | null {
  for (const [k, v] of Object.entries(series)) {
    if (k.startsWith(prefix)) return [k, v];
  }
  return null;
}

function renderLineOnChart(
  chart: IChartApi,
  data: TimeValue[],
  title: string,
  color: string,
  lineWidth: LineWidth = 2,
  paneIndex?: number,
) {
  const s = chart.addSeries(LineSeries, { color, lineWidth, title }, paneIndex);
  s.setData(data as never[]);
  return s;
}

function addLevelLines(chart: IChartApi, levels: number[], paneIndex?: number, color = "rgba(120,120,120,0.3)") {
  for (const level of levels) {
    const s = chart.addSeries(LineSeries, {
      color,
      lineWidth: 1,
      lineStyle: 2, // dashed
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    }, paneIndex);
    // Create a minimal 2-point line spanning the data
    s.setData([
      { time: 0 as never, value: level },
      { time: 9999999999 as never, value: level },
    ]);
  }
}

function renderOverlayIndicator(
  chart: IChartApi,
  ind: IndicatorResult,
  colorIdx: number,
) {
  const render = ind.render;
  const color = ADHOC_COLORS[colorIdx % ADHOC_COLORS.length];

  switch (render.type) {
    case "line": {
      // Simple line overlay — one series per non-underscore column
      for (const col of ind.columns) {
        const data = ind.series[col];
        if (data?.length) renderLineOnChart(chart, data, col, color);
      }
      break;
    }

    case "markers": {
      // PSAR-style: render as a thin line with small point markers (not heavy chart markers)
      const markerColor = render.color ?? "#f59e0b";
      for (const col of ind.columns) {
        const data = ind.series[col];
        if (!data?.length) continue;
        const s = chart.addSeries(LineSeries, {
          color: markerColor,
          lineVisible: false,
          pointMarkersVisible: true,
          pointMarkersRadius: render.size ?? 3,
          crosshairMarkerVisible: false,
          lastValueVisible: false,
          priceLineVisible: false,
          title: col,
        });
        s.setData(data as never[]);
      }
      break;
    }

    case "bands": {
      const upper = findCol(ind.series, render.upper_prefix);
      const middle = findCol(ind.series, render.middle_prefix);
      const lower = findCol(ind.series, render.lower_prefix);
      const bandColor = color;
      if (upper) renderLineOnChart(chart, upper[1], upper[0], bandColor + "99", 1);
      if (middle) renderLineOnChart(chart, middle[1], middle[0], bandColor, 1);
      if (lower) renderLineOnChart(chart, lower[1], lower[0], bandColor + "99", 1);
      break;
    }

    case "colored_line": {
      // SuperTrend: color changes based on price vs indicator
      const closeMap = new Map<number, number>();
      if (ind.series._close) {
        for (const pt of ind.series._close) closeMap.set(pt.time, pt.value);
      }
      for (const col of ind.columns) {
        const raw = ind.series[col];
        if (!raw?.length) continue;
        const coloredData = raw.map((pt) => {
          const close = closeMap.get(pt.time);
          const isAbove = close !== undefined && pt.value > close;
          return {
            time: pt.time,
            value: pt.value,
            color: isAbove ? render.above_color : render.below_color,
          };
        });
        const s = chart.addSeries(LineSeries, { lineWidth: 2, title: col });
        s.setData(coloredData as never[]);
      }
      break;
    }

    default: {
      // Fallback: line for each column
      for (const col of ind.columns) {
        const data = ind.series[col];
        if (data?.length) renderLineOnChart(chart, data, col, color);
      }
    }
  }
}

function renderPanelIndicator(
  chart: IChartApi,
  ind: IndicatorResult,
  colorIdx: number,
  paneIndex?: number,
) {
  const render: RenderSpec = ind.render;
  const baseColor = ADHOC_COLORS[colorIdx % ADHOC_COLORS.length];

  switch (render.type) {
    case "line": {
      for (const col of ind.columns) {
        const data = ind.series[col];
        if (data?.length) renderLineOnChart(chart, data, col, baseColor, 2, paneIndex);
      }
      if (render.levels?.length) addLevelLines(chart, render.levels, paneIndex);
      break;
    }

    case "histogram": {
      for (const col of ind.columns) {
        const data = ind.series[col];
        if (!data?.length) continue;
        const histData = data.map((pt) => ({
          time: pt.time,
          value: pt.value,
          color: pt.value >= 0 ? "#38bdf880" : "#f472b680",
        }));
        const s = chart.addSeries(HistogramSeries, { priceScaleId: "", title: col }, paneIndex);
        s.setData(histData as never[]);
      }
      break;
    }

    case "composite": {
      // MACD style: mix of lines and histograms
      let partIdx = 0;
      for (const part of render.parts) {
        const match = findCol(ind.series, part.prefix);
        if (!match) continue;
        const [colName, data] = match;
        if (part.style === "histogram") {
          const histData = data.map((pt) => ({
            time: pt.time,
            value: pt.value,
            color: pt.value >= 0 ? "#38bdf880" : "#f472b680",
          }));
          const s = chart.addSeries(HistogramSeries, { priceScaleId: "", title: part.label || colName }, paneIndex);
          s.setData(histData as never[]);
        } else {
          const c = ADHOC_COLORS[(colorIdx + partIdx) % ADHOC_COLORS.length];
          renderLineOnChart(chart, data, part.label || colName, c, 2, paneIndex);
        }
        partIdx++;
      }
      break;
    }

    case "multi_line": {
      for (const lineDef of render.lines) {
        const match = findCol(ind.series, lineDef.prefix);
        if (!match) continue;
        const [colName, data] = match;
        renderLineOnChart(chart, data, lineDef.label || colName, lineDef.color, 2, paneIndex);
      }
      if (render.levels?.length) addLevelLines(chart, render.levels, paneIndex);
      break;
    }

    default: {
      // Fallback: line for each column
      for (const col of ind.columns) {
        const data = ind.series[col];
        if (data?.length) renderLineOnChart(chart, data, col, baseColor, 2, paneIndex);
      }
    }
  }
}

// ── Legend item type ──

interface LegendItem {
  key: string;
  label: string;
  color: string;
}

// ── Chart component ──

export default function StrategyChart({ chartData, separateCols, adHocData, expanded }: Props) {
  const mainRef = useRef<HTMLDivElement>(null);
  const volumeRef = useRef<HTMLDivElement>(null);
  const chartsRef = useRef<IChartApi[]>([]);
  const [hidden, setHidden] = useState<string[]>([]);

  const hiddenSet = useMemo(() => new Set(hidden), [hidden]);

  const toggleVisibility = (key: string) => {
    setHidden((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key]
    );
  };

  // Build legend items from all sources
  const legendItems = useMemo(() => {
    const items: LegendItem[] = [];

    // Built-data overlays (e.g. SMA_50)
    let ci = 0;
    for (const col of Object.keys(chartData?.overlays || {})) {
      items.push({ key: `built:${col}`, label: col, color: COLORS[ci % COLORS.length] });
      ci++;
    }

    // Built-data separate indicators (e.g. ADX_14)
    ci = 0;
    for (const col of Object.keys(chartData?.indicators || {})) {
      items.push({ key: `built:${col}`, label: col, color: COLORS[ci % COLORS.length] });
      ci++;
    }

    // Ad-hoc indicators
    if (adHocData?.results) {
      let oi = 0;
      for (const ind of adHocData.results) {
        if (ind.error) continue;
        items.push({
          key: `adhoc:${ind.name}`,
          label: ind.label,
          color: ADHOC_COLORS[oi % ADHOC_COLORS.length],
        });
        oi++;
      }
    }

    return items;
  }, [chartData, adHocData]);

  const adHocOverlayResults = adHocData?.results.filter((r) => r.display === "overlay" && !r.error) ?? [];
  const adHocPanelResults = adHocData?.results.filter((r) => r.display === "panel" && !r.error) ?? [];

  // Count visible panel indicators to compute chart height
  const visibleBuiltPanels = Object.keys(chartData?.indicators || {}).filter(
    (col) => !hiddenSet.has(`built:${col}`)
  );
  const visibleAdHocPanels = adHocPanelResults.filter(
    (ind) => !hiddenSet.has(`adhoc:${ind.name}`)
  );
  // Group built panels into one pane, each ad-hoc panel gets its own pane
  const numPanelPanes =
    (visibleBuiltPanels.length > 0 ? 1 : 0) + visibleAdHocPanels.length;
  const baseHeight = expanded ? 500 : 300;
  const paneHeight = expanded ? 160 : 120;
  const mainChartHeight = baseHeight + numPanelPanes * paneHeight;

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

    // ── Single main chart (price pane 0 + indicator panes 1+) ──
    if (!mainRef.current) return;

    const chart = createChart(mainRef.current, {
      ...chartOptions,
      height: mainChartHeight,
      autoSize: true,
    });
    chartsRef.current.push(chart);

    // Pane 0: Price candles
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderDownColor: "#ef4444",
      borderUpColor: "#22c55e",
      wickDownColor: "#ef4444",
      wickUpColor: "#22c55e",
    });
    candleSeries.setData(chartData.ohlcv as never[]);

    // Pane 0: Strategy overlays (from built data) — filtered by visibility
    let ci = 0;
    for (const [name, data] of Object.entries(chartData.overlays || {})) {
      if (!hiddenSet.has(`built:${name}`)) {
        renderLineOnChart(chart, data as TimeValue[], name, COLORS[ci % COLORS.length], 1);
      }
      ci++;
    }

    // Pane 0: Ad-hoc overlay indicators — filtered by visibility
    let oi = 0;
    for (const ind of adHocOverlayResults) {
      if (!hiddenSet.has(`adhoc:${ind.name}`)) {
        renderOverlayIndicator(chart, ind, oi);
      }
      oi++;
    }

    // Track next pane index (pane 0 = price, pane 1+ = panels)
    let nextPane = 1;

    // Pane 1: Built-data separate indicators (all on one pane) — filtered
    if (visibleBuiltPanels.length > 0) {
      const pane = nextPane++;
      ci = 0;
      for (const [name, data] of Object.entries(chartData.indicators || {})) {
        if (!hiddenSet.has(`built:${name}`)) {
          renderLineOnChart(chart, data as TimeValue[], name, COLORS[ci % COLORS.length], 1, pane);
        }
        ci++;
      }
    }

    // Pane 2+: Ad-hoc panel indicators (one pane each) — filtered
    let panelIdx = 0;
    for (const ind of adHocPanelResults) {
      if (!hiddenSet.has(`adhoc:${ind.name}`)) {
        renderPanelIndicator(chart, ind, panelIdx, nextPane++);
      }
      panelIdx++;
    }

    chart.timeScale().fitContent();

    // ── Volume chart (separate, small) ──
    if (volumeRef.current && chartData.volume?.length) {
      const volChart = createChart(volumeRef.current, {
        ...chartOptions,
        height: 80,
        autoSize: true,
      });
      chartsRef.current.push(volChart);

      const volSeries = volChart.addSeries(HistogramSeries, {
        priceFormat: { type: "volume" },
        priceScaleId: "",
      });
      volSeries.setData(chartData.volume as never[]);
      volChart.timeScale().fitContent();

      // Sync main chart ↔ volume chart
      chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
        if (range) volChart.timeScale().setVisibleLogicalRange(range);
      });
      volChart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
        if (range) chart.timeScale().setVisibleLogicalRange(range);
      });
    }

    return () => {
      chartsRef.current.forEach((c) => c.remove());
      chartsRef.current = [];
    };
  }, [chartData, separateCols, adHocData, adHocOverlayResults.length, adHocPanelResults.length, hidden, mainChartHeight, expanded]);

  return (
    <div className="space-y-1">
      {/* Legend / visibility toggles */}
      {legendItems.length > 0 && (
        <div className="flex flex-wrap gap-1.5 px-1 py-1.5">
          {legendItems.map((item) => {
            const isHidden = hiddenSet.has(item.key);
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
                  style={{
                    backgroundColor: isHidden ? "#4b5563" : item.color,
                  }}
                />
                {item.label}
              </button>
            );
          })}
        </div>
      )}
      <div ref={mainRef} className="rounded bg-gray-900" style={{ height: mainChartHeight }} />
      {chartData?.volume?.length > 0 && (
        <div ref={volumeRef} className="rounded bg-gray-900" style={{ height: 80 }} />
      )}
    </div>
  );
}
