import { useEffect, useRef } from "react";
import {
  createChart,
  type IChartApi,
  ColorType,
  LineSeries,
  HistogramSeries,
} from "lightweight-charts";

interface ChartPoint {
  time: number;
  value: number;
  color?: string;
}

interface Props {
  chartData: Record<string, ChartPoint[]>;
  quoteVenues: string[];
}

const COLORS = [
  "#3b82f6", "#f59e0b", "#8b5cf6", "#ef4444", "#10b981",
  "#ec4899", "#14b8a6", "#f97316",
];

export default function BasisChart({ chartData, quoteVenues }: Props) {
  const priceRef = useRef<HTMLDivElement>(null);
  const basisRef = useRef<HTMLDivElement>(null);
  const qualityRef = useRef<HTMLDivElement>(null);
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

    // Price chart
    if (priceRef.current) {
      const chart = createChart(priceRef.current, {
        ...chartOptions,
        height: 200,
      });
      chartsRef.current.push(chart);

      // Base price
      if (chartData.base_price) {
        const s = chart.addSeries(LineSeries, {
          color: "#60a5fa",
          lineWidth: 1,
          title: "Base",
        });
        s.setData(chartData.base_price as never[]);
      }

      // Quote prices
      quoteVenues.forEach((venue, i) => {
        const key = `${venue}_price`;
        if (chartData[key]) {
          const s = chart.addSeries(LineSeries, {
            color: COLORS[(i + 1) % COLORS.length],
            lineWidth: 1,
            title: venue,
          });
          s.setData(chartData[key] as never[]);
        }
      });

      chart.timeScale().fitContent();
    }

    // Basis chart
    if (basisRef.current) {
      const chart = createChart(basisRef.current, {
        ...chartOptions,
        height: 180,
      });
      chartsRef.current.push(chart);

      quoteVenues.forEach((venue, i) => {
        const key = `${venue}_basis_bps`;
        if (chartData[key]) {
          const s = chart.addSeries(LineSeries, {
            color: COLORS[(i + 1) % COLORS.length],
            lineWidth: 1,
            title: `${venue} basis (bps)`,
          });
          s.setData(chartData[key] as never[]);
        }
      });

      chart.timeScale().fitContent();
    }

    // Quality chart
    if (qualityRef.current && chartData.quality?.length) {
      const chart = createChart(qualityRef.current, {
        ...chartOptions,
        height: 40,
      });
      chartsRef.current.push(chart);

      const s = chart.addSeries(HistogramSeries, {
        priceScaleId: "",
      });
      s.setData(chartData.quality as never[]);
      chart.timeScale().fitContent();
    }

    // Sync time scales
    if (chartsRef.current.length > 1) {
      const primary = chartsRef.current[0];
      const rest = chartsRef.current.slice(1);
      primary.timeScale().subscribeVisibleLogicalRangeChange((range) => {
        if (range) rest.forEach((c) => c.timeScale().setVisibleLogicalRange(range));
      });
      rest.forEach((c) => {
        c.timeScale().subscribeVisibleLogicalRangeChange((range) => {
          if (range) {
            primary.timeScale().setVisibleLogicalRange(range);
            rest.filter((r) => r !== c).forEach((r) => r.timeScale().setVisibleLogicalRange(range));
          }
        });
      });
    }

    return () => {
      chartsRef.current.forEach((c) => c.remove());
      chartsRef.current = [];
    };
  }, [chartData, quoteVenues]);

  return (
    <div className="space-y-1">
      <div ref={priceRef} className="rounded bg-gray-900" />
      <div ref={basisRef} className="rounded bg-gray-900" />
      {chartData.quality?.length > 0 && (
        <div ref={qualityRef} className="rounded bg-gray-900" />
      )}
    </div>
  );
}
