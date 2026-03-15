import { useState, useMemo } from "react";
import type { BacktestRun } from "@/api/backtest";
import { ChevronRight, Search, ArrowLeft } from "lucide-react";

interface Props {
  runs: BacktestRun[];
  selected: BacktestRun | null;
  onSelect: (run: BacktestRun) => void;
}

/** Parse run_id like "20260315_141351" → formatted timestamp */
function formatRunId(runId: string): string {
  const m = runId.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})/);
  if (!m) return runId;
  return `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}`;
}

/** Parse run_id → Date object */
function runIdToDate(runId: string): Date | null {
  const m = runId.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/);
  if (!m) return null;
  return new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]);
}

/** Relative time string like "5 mins ago", "2 hours ago", "3 days ago" */
function timeAgo(date: Date): string {
  const seconds = Math.floor((Date.now() - date.getTime()) / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} min${minutes === 1 ? "" : "s"} ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} hour${hours === 1 ? "" : "s"} ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days} day${days === 1 ? "" : "s"} ago`;
  const months = Math.floor(days / 30);
  return `${months} month${months === 1 ? "" : "s"} ago`;
}

// ── Compact top bar shown when a run is loaded ──

export function BacktestRunBar({
  selected,
  onChangeRun,
}: {
  selected: BacktestRun;
  onChangeRun: () => void;
}) {
  const meta = selected.meta as Record<string, unknown>;
  const retPct = meta.total_return_pct as number | undefined;
  const trades = meta.total_trades as number | undefined;
  const date = runIdToDate(selected.run_id);

  return (
    <div className="flex items-center gap-3 border-b border-gray-800 px-4 py-2">
      <button
        onClick={onChangeRun}
        className="flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs text-gray-400 hover:bg-gray-800 hover:text-gray-200 transition"
      >
        <ArrowLeft className="h-3.5 w-3.5" />
        Change
      </button>
      <div className="h-4 w-px bg-gray-800" />
      <span className="text-xs font-semibold text-gray-300">{selected.strategy_name}</span>
      <span className="text-xs tabular-nums text-gray-500">{formatRunId(selected.run_id)}</span>
      {date && <span className="text-[10px] text-gray-600">{timeAgo(date)}</span>}
      {trades != null && <span className="text-[10px] text-gray-500">{trades} trades</span>}
      {retPct != null && (
        <span className={`text-xs font-medium ${retPct >= 0 ? "text-green-400" : "text-red-400"}`}>
          {retPct >= 0 ? "+" : ""}{retPct.toFixed(2)}%
        </span>
      )}
    </div>
  );
}

// ── Full-page run selector panel ──

export default function BacktestRunSelector({ runs, selected, onSelect }: Props) {
  const [search, setSearch] = useState("");
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(() => {
    // Auto-expand all groups initially
    return new Set(runs.map((r) => r.strategy_name));
  });

  // Group runs by strategy, sorted by timestamp desc within each group
  const groups = useMemo(() => {
    const map = new Map<string, BacktestRun[]>();
    for (const run of runs) {
      const key = run.strategy_name;
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(run);
    }
    for (const [, group] of map) {
      group.sort((a, b) => b.run_id.localeCompare(a.run_id));
    }
    return Array.from(map.entries()).sort(([a], [b]) => a.localeCompare(b));
  }, [runs]);

  // Filter by search
  const filteredGroups = useMemo(() => {
    if (!search.trim()) return groups;
    const q = search.toLowerCase();
    return groups
      .map(([name, groupRuns]) => {
        if (name.toLowerCase().includes(q)) return [name, groupRuns] as const;
        const filtered = groupRuns.filter(
          (r) => r.run_id.includes(q) || formatRunId(r.run_id).includes(q),
        );
        return filtered.length > 0 ? [name, filtered] as const : null;
      })
      .filter(Boolean) as [string, BacktestRun[]][];
  }, [groups, search]);

  const toggleGroup = (name: string) => {
    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center gap-4 border-b border-gray-800 px-6 py-4">
        <div>
          <h1 className="text-sm font-semibold text-gray-200">Select Backtest Run</h1>
          <p className="text-xs text-gray-500">{runs.length} run{runs.length !== 1 ? "s" : ""} across {groups.length} strateg{groups.length !== 1 ? "ies" : "y"}</p>
        </div>
        <div className="ml-auto flex items-center gap-2 rounded-md border border-gray-700 bg-gray-900 px-3 py-1.5">
          <Search className="h-3.5 w-3.5 text-gray-500" />
          <input
            autoFocus
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Filter by strategy or date…"
            className="w-64 bg-transparent text-xs text-gray-300 placeholder:text-gray-600 outline-none"
          />
        </div>
      </div>

      {/* Strategy groups */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        {filteredGroups.length === 0 ? (
          <p className="py-12 text-center text-sm text-gray-600">No matching runs</p>
        ) : (
          <div className="space-y-2">
            {filteredGroups.map(([strategyName, groupRuns]) => {
              const isExpanded = expandedGroups.has(strategyName);
              return (
                <div key={strategyName} className="rounded-lg border border-gray-800 bg-gray-900/50">
                  {/* Strategy header */}
                  <button
                    onClick={() => toggleGroup(strategyName)}
                    className="flex w-full items-center gap-3 px-4 py-3 text-left hover:bg-gray-800/30 transition rounded-lg"
                  >
                    <ChevronRight
                      className={`h-4 w-4 shrink-0 text-gray-500 transition-transform ${isExpanded ? "rotate-90" : ""}`}
                    />
                    <span className="text-sm font-semibold text-gray-200">{strategyName}</span>
                    <span className="ml-auto text-xs text-gray-500">
                      {groupRuns.length} run{groupRuns.length !== 1 ? "s" : ""}
                    </span>
                  </button>

                  {/* Run list */}
                  {isExpanded && (
                    <div className="border-t border-gray-800">
                      {groupRuns.map((run) => {
                        const meta = run.meta as Record<string, unknown>;
                        const retPct = meta.total_return_pct as number | undefined;
                        const trades = meta.total_trades as number | undefined;
                        const maxDd = meta.max_drawdown_pct as number | undefined;
                        const winRate = meta.win_rate_pct as number | undefined;
                        const date = runIdToDate(run.run_id);
                        const isActive =
                          selected?.run_id === run.run_id &&
                          selected?.strategy_name === run.strategy_name;

                        return (
                          <button
                            key={run.run_id}
                            onClick={() => onSelect(run)}
                            className={`flex w-full items-center gap-4 px-4 py-2.5 pl-11 text-left transition ${
                              isActive
                                ? "bg-blue-900/20 text-blue-300"
                                : "text-gray-400 hover:bg-gray-800/40 hover:text-gray-200"
                            }`}
                          >
                            {/* Timestamp + relative time */}
                            <div className="min-w-[160px]">
                              <span className="text-xs tabular-nums">{formatRunId(run.run_id)}</span>
                              {date && (
                                <span className="ml-2 text-[10px] text-gray-600">{timeAgo(date)}</span>
                              )}
                            </div>

                            {/* Metrics */}
                            <div className="flex flex-1 items-center gap-4">
                              {trades != null && (
                                <span className="text-[10px] text-gray-500">{trades} trades</span>
                              )}
                              {winRate != null && (
                                <span className="text-[10px] text-gray-500">{winRate.toFixed(0)}% win</span>
                              )}
                              {maxDd != null && (
                                <span className="text-[10px] text-gray-500">{maxDd.toFixed(1)}% DD</span>
                              )}
                            </div>

                            {/* Return */}
                            {retPct != null && (
                              <span
                                className={`min-w-[56px] text-right text-xs font-semibold ${
                                  retPct >= 0 ? "text-green-400" : "text-red-400"
                                }`}
                              >
                                {retPct >= 0 ? "+" : ""}{retPct.toFixed(2)}%
                              </span>
                            )}
                          </button>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
