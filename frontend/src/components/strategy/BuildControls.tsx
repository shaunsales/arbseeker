import { useState, useEffect } from "react";
import { getAvailableDates, buildStrategy } from "@/api/strategy";
import { Button } from "@/components/ui/button";
import MonthRangePicker from "./MonthRangePicker";
import type { AvailableDates } from "@/types/api";

interface Props {
  className: string;
  onBuilt: () => void;
}

export default function BuildControls({ className, onBuilt }: Props) {
  const [dates, setDates] = useState<AvailableDates | null>(null);
  const [loading, setLoading] = useState(true);
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [building, setBuilding] = useState(false);
  const [result, setResult] = useState<{
    success: boolean;
    message?: string;
  } | null>(null);

  useEffect(() => {
    setLoading(true);
    getAvailableDates(className)
      .then((d) => {
        setDates(d);
        if (d.earliest_start) setStartDate(d.earliest_start);
        if (d.latest_end) setEndDate(d.latest_end);
      })
      .finally(() => setLoading(false));
  }, [className]);

  async function handleBuild() {
    setBuilding(true);
    setResult(null);
    try {
      const res = await buildStrategy({
        class_name: className,
        start_date: startDate,
        end_date: endDate,
      });
      setResult(res);
      if (res.success) {
        setTimeout(() => onBuilt(), 1000);
      }
    } catch (e) {
      setResult({
        success: false,
        message: e instanceof Error ? e.message : "Build failed",
      });
    }
    setBuilding(false);
  }

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-5 space-y-5">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-400">
        Build Data
      </h3>

      {loading ? (
        <p className="text-xs text-gray-500">Loading available dates...</p>
      ) : !dates ? (
        <p className="text-xs text-red-400">Failed to load available dates</p>
      ) : (
        <>
          <MonthRangePicker
            dates={dates}
            startDate={startDate}
            endDate={endDate}
            onRangeChange={(s, e) => {
              setStartDate(s);
              setEndDate(e);
            }}
          />

          <Button
            onClick={handleBuild}
            disabled={building || !startDate || !endDate}
            className="w-full"
          >
            {building ? "Building..." : "Build Data"}
          </Button>

          {/* Result */}
          {result && (
            <div
              className={`rounded border p-2.5 text-xs ${
                result.success
                  ? "border-green-800 bg-green-900/30 text-green-300"
                  : "border-red-800 bg-red-900/30 text-red-300"
              }`}
            >
              {result.message || (result.success ? "Build complete!" : "Build failed")}
            </div>
          )}
        </>
      )}
    </div>
  );
}
