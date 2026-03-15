import { useState, useMemo, useCallback } from "react";
import { Lock } from "lucide-react";
import { Badge } from "@/components/ui/badge";

const MONTH_LABELS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/** Generate all YYYY-MM strings between start and end (inclusive). */
export function expandMonthRange(start: string, end: string): string[] {
  const result: string[] = [];
  let [y, m] = start.split("-").map(Number);
  const [ey, em] = end.split("-").map(Number);
  while (y < ey || (y === ey && m <= em)) {
    result.push(`${y}-${String(m).padStart(2, "0")}`);
    m++;
    if (m > 12) { m = 1; y++; }
  }
  return result;
}

export interface PerIntervalInfo {
  start: string;
  end: string;
  count: number;
  months: string[];
}

export interface MonthRangePickerProps {
  /** All months that can potentially be shown (YYYY-MM strings, sorted). */
  months: string[];
  /** Currently selected start (YYYY-MM) or "" */
  startDate: string;
  /** Currently selected end (YYYY-MM) or "" */
  endDate: string;
  /** Callback when range changes */
  onRangeChange: (start: string, end: string) => void;
  /** Optional set of warmup-only months (selectable=false, shown with lock icon). */
  warmupMonths?: string[];
  /** Label for warmup legend (e.g., "Warmup (2mo)"). Defaults to "Warmup". */
  warmupLabel?: string;
  /** Optional per-interval summary badges. */
  perInterval?: Record<string, PerIntervalInfo>;
  /** Whether to show the legend row. Defaults to true. */
  showLegend?: boolean;
}

export default function MonthRangePicker({
  months,
  startDate,
  endDate,
  onRangeChange,
  warmupMonths,
  warmupLabel = "Warmup",
  perInterval,
  showLegend = true,
}: MonthRangePickerProps) {
  const availableSet = useMemo(() => new Set(months), [months]);
  const warmupSet = useMemo(() => new Set(warmupMonths ?? []), [warmupMonths]);

  // Figure out year range to display
  const years = useMemo(() => {
    if (months.length === 0) return [];
    const firstYear = parseInt(months[0].slice(0, 4));
    const lastYear = parseInt(months[months.length - 1].slice(0, 4));
    const result: number[] = [];
    for (let y = firstYear; y <= lastYear; y++) result.push(y);
    return result;
  }, [months]);

  // Selection state
  const [picking, setPicking] = useState<"idle" | "picking_end">("idle");

  const isSelectable = useCallback(
    (month: string) => availableSet.has(month) && !warmupSet.has(month),
    [availableSet, warmupSet],
  );

  const isInRange = useCallback(
    (month: string) => {
      if (!startDate || !endDate) return false;
      return month >= startDate && month <= endDate;
    },
    [startDate, endDate],
  );

  function handleClick(month: string) {
    if (!isSelectable(month)) return;

    if (picking === "idle") {
      onRangeChange(month, "");
      setPicking("picking_end");
    } else {
      if (month < startDate) {
        onRangeChange(month, startDate);
      } else {
        onRangeChange(startDate, month);
      }
      setPicking("idle");
    }
  }

  function getMonthStyle(month: string) {
    const hasData = availableSet.has(month);
    const isWarmup = warmupSet.has(month);
    const selectable = hasData && !isWarmup;
    const inRange = isInRange(month);
    const isStart = month === startDate;
    const isEnd = month === endDate;
    const isEndpoint = isStart || isEnd;

    if (!hasData) return "text-gray-700 cursor-default";
    if (isWarmup) return "text-gray-600 bg-amber-950/20 cursor-not-allowed";
    if (isEndpoint) return "bg-blue-600 text-white font-semibold cursor-pointer";
    if (inRange) return "bg-blue-900/40 text-blue-300 cursor-pointer";
    if (selectable) return "text-gray-300 hover:bg-gray-700 cursor-pointer";
    return "text-gray-600 cursor-default";
  }

  return (
    <div className="space-y-3">
      {/* Per-interval summary badges */}
      {perInterval && (
        <div className="flex flex-wrap gap-2">
          {Object.entries(perInterval).map(([iv, info]) => (
            <Badge
              key={iv}
              variant="secondary"
              className="bg-gray-800 text-gray-400 hover:bg-gray-800"
            >
              <span className="font-semibold text-blue-400">{iv}</span>
              <span className="mx-1.5 text-gray-600">·</span>
              {info.start} → {info.end}
              <span className="mx-1.5 text-gray-600">·</span>
              <span className="text-gray-500">{info.count} months</span>
            </Badge>
          ))}
        </div>
      )}

      {/* Legend */}
      {showLegend && (
        <div className="flex items-center gap-4 text-[11px] text-gray-500">
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-3 w-3 rounded-sm bg-gray-700" />
            No data
          </span>
          {warmupMonths && warmupMonths.length > 0 && (
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-3 w-3 rounded-sm bg-amber-950/40 border border-amber-900/30" />
              <Lock className="h-3 w-3 text-amber-600" />
              {warmupLabel}
            </span>
          )}
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-3 w-3 rounded-sm bg-gray-800 border border-gray-700" />
            Available
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-3 w-3 rounded-sm bg-blue-600" />
            Selected
          </span>
        </div>
      )}

      {/* Instruction */}
      <p className="text-xs text-gray-500">
        {picking === "idle"
          ? startDate && endDate
            ? "Click a month to start a new selection"
            : "Click a month to set the start date"
          : "Click a month to set the end date"}
      </p>

      {/* Year grids */}
      <div className="space-y-3">
        {years.map((year) => (
          <div key={year}>
            <div className="mb-1.5 text-xs font-semibold text-gray-400">
              {year}
            </div>
            <div className="grid grid-cols-6 gap-1">
              {MONTH_LABELS.map((label, mi) => {
                const month = `${year}-${String(mi + 1).padStart(2, "0")}`;
                const hasData = availableSet.has(month);
                const isWarmup = warmupSet.has(month);

                return (
                  <button
                    key={month}
                    onClick={() => handleClick(month)}
                    disabled={!hasData || isWarmup}
                    title={
                      isWarmup
                        ? `${label} ${year} — warmup period`
                        : !hasData
                          ? `${label} ${year} — no data`
                          : `${label} ${year}`
                    }
                    className={`relative rounded-md px-2 py-1.5 text-xs font-medium transition-colors ${getMonthStyle(month)}`}
                  >
                    {label}
                    {isWarmup && (
                      <Lock className="absolute top-0.5 right-0.5 h-2.5 w-2.5 text-amber-700" />
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {/* Selected range display */}
      {startDate && endDate && (
        <div className="rounded-md border border-blue-900/50 bg-blue-950/30 px-3 py-2 text-xs text-blue-300">
          Selected: <span className="font-semibold">{startDate}</span> →{" "}
          <span className="font-semibold">{endDate}</span>
          {(() => {
            const count = months.filter(
              (m) => m >= startDate && m <= endDate && !warmupSet.has(m),
            ).length;
            return <span className="text-blue-400/70"> ({count} months)</span>;
          })()}
        </div>
      )}
    </div>
  );
}
