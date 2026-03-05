import { useEffect, useRef } from "react";
import {
  createChart,
  type IChartApi,
  ColorType,
  CandlestickSeries,
  HistogramSeries,
} from "lightweight-charts";
import type { OhlcvChartData } from "@/types/api";

interface Props {
  chartData: OhlcvChartData;
  height?: number;
}

export default function OhlcvChart({ chartData, height = 400 }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
    }
    if (!containerRef.current || !chartData.ohlcv.length) return;

    const chart = createChart(containerRef.current, {
      height,
      layout: {
        background: { type: ColorType.Solid, color: "#111827" },
        textColor: "#9ca3af",
      },
      grid: {
        vertLines: { color: "#1f2937" },
        horzLines: { color: "#1f2937" },
      },
      crosshair: { mode: 0 },
      rightPriceScale: { borderColor: "#374151" },
      timeScale: {
        borderColor: "#374151",
        timeVisible: true,
        secondsVisible: false,
      },
    });
    chartRef.current = chart;

    // Candlestick series
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderDownColor: "#ef4444",
      borderUpColor: "#22c55e",
      wickDownColor: "#ef4444",
      wickUpColor: "#22c55e",
    });
    candleSeries.setData(chartData.ohlcv as never[]);

    // Volume as overlay on bottom of same chart
    if (chartData.volume.length) {
      const volSeries = chart.addSeries(HistogramSeries, {
        priceFormat: { type: "volume" },
        priceScaleId: "vol",
      });
      volSeries.priceScale().applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
      });
      volSeries.setData(chartData.volume as never[]);
    }

    chart.timeScale().fitContent();

    // Resize observer
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        chart.applyOptions({ width: entry.contentRect.width });
      }
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, [chartData, height]);

  return <div ref={containerRef} className="rounded bg-gray-900" />;
}
