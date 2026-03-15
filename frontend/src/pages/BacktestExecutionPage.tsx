import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { listBacktests } from "@/api/backtest";
import RunBacktestForm, { type BacktestStrategy } from "@/components/backtest/RunBacktestForm";
import { FlaskConical, ArrowLeft, Clock } from "lucide-react";

function timeAgo(iso: string): string {
  const ms = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(ms / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

export default function BacktestExecutionPage() {
  const [selected, setSelected] = useState<BacktestStrategy | null>(null);

  const { data } = useQuery({
    queryKey: ["backtest-runs"],
    queryFn: listBacktests,
  });

  const strategies = useMemo(() => {
    const list = (data?.strategies ?? []).filter((s) => s.has_data_spec);
    return [...list].sort((a, b) => {
      if (a.last_modified && b.last_modified) return b.last_modified.localeCompare(a.last_modified);
      if (a.last_modified) return -1;
      if (b.last_modified) return 1;
      return a.class_name.localeCompare(b.class_name);
    });
  }, [data?.strategies]);

  // ── Phase 1: Strategy selector ──
  if (!selected) {
    return (
      <div className="flex h-full flex-col">
        <div className="border-b border-gray-800 px-6 py-4">
          <h1 className="text-sm font-semibold text-gray-200">Select Strategy</h1>
          <p className="text-xs text-gray-500">
            {strategies.length} strateg{strategies.length !== 1 ? "ies" : "y"} with built data
          </p>
        </div>

        <div className="flex-1 overflow-y-auto px-6 py-4">
          {strategies.length === 0 ? (
            <div className="py-12 text-center">
              <p className="text-sm text-gray-500">No strategies with built data found.</p>
              <p className="mt-1 text-xs text-gray-600">
                Build strategy data on the Strategies page first.
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              {strategies.map((s) => (
                <button
                  key={s.class_name}
                  onClick={() => setSelected(s)}
                  className="flex w-full items-center gap-4 rounded-lg border border-gray-800 bg-gray-900/50 px-5 py-3.5 text-left text-gray-300 transition hover:border-gray-700 hover:bg-gray-800/40"
                >
                  <FlaskConical className="h-4 w-4 shrink-0 text-gray-500" />
                  <div className="flex-1 min-w-0">
                    <span className="text-sm font-semibold">{s.class_name}</span>
                    <span className="ml-2 text-xs text-gray-500">{s.module}</span>
                    {s.last_modified && (
                      <div className="mt-0.5 flex items-center gap-1 text-[10px] text-gray-600">
                        <Clock className="h-2.5 w-2.5" />
                        Modified {timeAgo(s.last_modified)}
                      </div>
                    )}
                  </div>
                  {s.data_date_range && (
                    <span className="text-[10px] text-gray-500">
                      {s.data_date_range.start} → {s.data_date_range.end}
                    </span>
                  )}
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-blue-900/30 px-2 py-0.5 text-[10px] font-medium text-blue-400">
                    <span className="inline-block h-1.5 w-1.5 rounded-full bg-blue-400" />
                    Data ready
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  }

  // ── Phase 2: Compact bar + config form ──
  return (
    <div className="flex h-full flex-col">
      {/* Strategy bar */}
      <div className="flex items-center gap-3 border-b border-gray-800 px-4 py-2">
        <button
          onClick={() => setSelected(null)}
          className="flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs text-gray-400 hover:bg-gray-800 hover:text-gray-200 transition"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Change
        </button>
        <div className="h-4 w-px bg-gray-800" />
        <FlaskConical className="h-3.5 w-3.5 text-gray-500" />
        <span className="text-xs font-semibold text-gray-300">{selected.class_name}</span>
        <span className="text-[10px] text-gray-500">{selected.module}</span>
        {selected.last_modified && (
          <span className="flex items-center gap-1 text-[10px] text-gray-600">
            <Clock className="h-2.5 w-2.5" />
            {timeAgo(selected.last_modified)}
          </span>
        )}
      </div>

      {/* Config form */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="mx-auto max-w-lg">
          <RunBacktestForm
            key={selected.class_name}
            strategy={selected}
          />
        </div>
      </div>
    </div>
  );
}
