import { useCallback, useState, useEffect } from "react";
import type { IChartApi } from "lightweight-charts";
import { ZoomIn, ZoomOut, Maximize2 } from "lucide-react";

interface Props {
  chartsRef: React.RefObject<IChartApi[]>;
  /** Increment when charts are recreated to re-subscribe to range events */
  chartVersion?: number;
}

const PRESETS = [
  { label: "1D", secs: 86400 },
  { label: "3D", secs: 86400 * 3 },
  { label: "1W", secs: 86400 * 7 },
  { label: "1M", secs: 86400 * 30 },
  { label: "3M", secs: 86400 * 90 },
];

const btnClass =
  "rounded p-1 text-gray-500 hover:bg-gray-800 hover:text-gray-300 transition";
const presetBtnClass =
  "rounded px-1.5 py-0.5 text-[10px] font-medium text-gray-500 hover:bg-gray-800 hover:text-gray-300 transition";

export default function ChartZoomControls({ chartsRef, chartVersion }: Props) {
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

  const zoomPreset = useCallback(
    (seconds: number) => {
      const chart = chartsRef.current?.[0];
      if (!chart) return;
      const ts = chart.timeScale();
      const range = ts.getVisibleRange();
      if (!range) return;
      // Anchor to the right edge (most recent data)
      const end = range.to as number;
      ts.setVisibleRange({
        from: (end - seconds) as never,
        to: end as never,
      });
    },
    [chartsRef],
  );

  const zoomBy = useCallback(
    (factor: number) => {
      const chart = chartsRef.current?.[0];
      if (!chart) return;
      const ts = chart.timeScale();
      const range = ts.getVisibleRange();
      if (!range) return;
      const mid = ((range.to as number) + (range.from as number)) / 2;
      const half =
        (((range.to as number) - (range.from as number)) / 2) * factor;
      ts.setVisibleRange({
        from: (mid - half) as never,
        to: (mid + half) as never,
      });
    },
    [chartsRef],
  );

  const zoomFit = useCallback(() => {
    chartsRef.current?.forEach((c) => c.timeScale().fitContent());
  }, [chartsRef]);

  // Subscribe to visible range changes on the first chart
  useEffect(() => {
    const chart = chartsRef.current?.[0];
    if (!chart) return;

    const handler = (range: { from: unknown; to: unknown } | null) => {
      if (!range) return;
      updateScaleLabel(range.from as number, range.to as number);
    };
    chart.timeScale().subscribeVisibleTimeRangeChange(handler);

    // Set initial label
    const initRange = chart.timeScale().getVisibleRange();
    if (initRange)
      updateScaleLabel(initRange.from as number, initRange.to as number);

    return () => {
      try {
        chart.timeScale().unsubscribeVisibleTimeRangeChange(handler);
      } catch {
        // chart may already be disposed
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chartsRef, updateScaleLabel, chartVersion]);

  return (
    <div className="flex items-center gap-1 px-1 pb-1">
      <span className="mr-1 min-w-[32px] text-center font-mono text-[10px] font-semibold text-gray-400">
        {scaleLabel}
      </span>
      <button
        onClick={() => zoomBy(0.5)}
        className={btnClass}
        title="Zoom in"
      >
        <ZoomIn className="h-3.5 w-3.5" />
      </button>
      <button
        onClick={() => zoomBy(2)}
        className={btnClass}
        title="Zoom out"
      >
        <ZoomOut className="h-3.5 w-3.5" />
      </button>
      <div className="mx-1 h-3 w-px bg-gray-800" />
      {PRESETS.map((p) => (
        <button
          key={p.label}
          onClick={() => zoomPreset(p.secs)}
          className={presetBtnClass}
        >
          {p.label}
        </button>
      ))}
      <button onClick={zoomFit} className={btnClass} title="Fit all">
        <Maximize2 className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}
