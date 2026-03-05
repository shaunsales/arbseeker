import { useEffect, useRef } from "react";
import {
  createChart,
  type IChartApi,
  ColorType,
  CandlestickSeries,
  HistogramSeries,
} from "lightweight-charts";

interface ChartData {
  timestamps: string[];
  open: number[];
  high: number[];
  low: number[];
  close: number[];
  volume: number[];
}

interface Props {
  chartData: ChartData;
}

export default function OhlcvChart({ chartData }: Props) {
  const priceRef = useRef<HTMLDivElement>(null);
  const volumeRef = useRef<HTMLDivElement>(null);
  const chartsRef = useRef<IChartApi[]>([]);

  useEffect(() => {
    chartsRef.current.forEach((c) => c.remove());
    chartsRef.current = [];

    if (!chartData.timestamps.length) return;

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

    // Build data arrays
    const ohlcv = chartData.timestamps.map((t, i) => ({
      time: t,
      open: chartData.open[i],
      high: chartData.high[i],
      low: chartData.low[i],
      close: chartData.close[i],
    }));

    const volume = chartData.timestamps.map((t, i) => ({
      time: t,
      value: chartData.volume[i],
      color: chartData.close[i] >= chartData.open[i] ? "#22c55e80" : "#ef444480",
    }));

    // Price chart
    if (priceRef.current) {
      const chart = createChart(priceRef.current, {
        ...chartOptions,
        height: 320,
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
      candleSeries.setData(ohlcv as never[]);
      chart.timeScale().fitContent();
    }

    // Volume chart
    if (volumeRef.current) {
      const chart = createChart(volumeRef.current, {
        ...chartOptions,
        height: 80,
      });
      chartsRef.current.push(chart);

      const volSeries = chart.addSeries(HistogramSeries, {
        priceFormat: { type: "volume" },
        priceScaleId: "",
      });
      volSeries.setData(volume as never[]);
      chart.timeScale().fitContent();
    }

    // Sync time scales
    if (chartsRef.current.length === 2) {
      const [price, vol] = chartsRef.current;
      price.timeScale().subscribeVisibleLogicalRangeChange((range) => {
        if (range) vol.timeScale().setVisibleLogicalRange(range);
      });
      vol.timeScale().subscribeVisibleLogicalRangeChange((range) => {
        if (range) price.timeScale().setVisibleLogicalRange(range);
      });
    }

    return () => {
      chartsRef.current.forEach((c) => c.remove());
      chartsRef.current = [];
    };
  }, [chartData]);

  return (
    <div className="space-y-1">
      <div ref={priceRef} className="rounded bg-gray-900" />
      <div ref={volumeRef} className="rounded bg-gray-900" />
    </div>
  );
}
