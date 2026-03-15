import { useState, useMemo, useRef, useEffect } from "react";
import type { BacktestRun } from "@/api/backtest";
import { ChevronDown, ChevronRight, Clock, Search } from "lucide-react";

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

export default function BacktestRunSelector({ runs, selected, onSelect }: Props) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set());
  const ref = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  // Group runs by strategy, sorted by timestamp desc within each group
  const groups = useMemo(() => {
    const map = new Map<string, BacktestRun[]>();
    for (const run of runs) {
      const key = run.strategy_name;
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(run);
    }
    // Sort runs within each group by run_id desc (newest first)
    for (const [, group] of map) {
      group.sort((a, b) => b.run_id.localeCompare(a.run_id));
    }
    // Sort groups alphabetically
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

  // Auto-expand group of selected run, and first group on initial open
  useEffect(() => {
    if (open && selected) {
      setExpandedGroups((prev) => {
        const next = new Set(prev);
        next.add(selected.strategy_name);
        return next;
      });
    }
  }, [open, selected]);

  const toggleGroup = (name: string) => {
    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  const handleSelect = (run: BacktestRun) => {
    onSelect(run);
    setOpen(false);
    setSearch("");
  };

  const selectedLabel = selected
    ? `${selected.strategy_name} — ${formatRunId(selected.run_id)}`
    : "Select a backtest run…";

  return (
    <div ref={ref} className="relative">
      {/* Trigger */}
      <button
        onClick={() => setOpen(!open)}
        className="flex h-8 min-w-[320px] items-center gap-2 rounded-md border border-gray-700 bg-gray-900 px-3 text-xs text-gray-300 hover:border-gray-600 hover:bg-gray-800/80 transition"
      >
        <Clock className="h-3.5 w-3.5 shrink-0 text-gray-500" />
        <span className="flex-1 truncate text-left">{selectedLabel}</span>
        {selected && (() => {
          const meta = selected.meta as Record<string, unknown>;
          const retPct = meta.total_return_pct as number | undefined;
          return retPct != null ? (
            <span className={`text-[10px] font-medium ${retPct >= 0 ? "text-green-400" : "text-red-400"}`}>
              {retPct >= 0 ? "+" : ""}{retPct.toFixed(1)}%
            </span>
          ) : null;
        })()}
        <ChevronDown className={`h-3.5 w-3.5 shrink-0 text-gray-500 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>

      {/* Dropdown */}
      {open && (
        <div className="absolute left-0 top-10 z-50 w-96 rounded-lg border border-gray-700 bg-gray-900 shadow-xl">
          {/* Search */}
          <div className="flex items-center gap-2 border-b border-gray-800 px-3 py-2">
            <Search className="h-3.5 w-3.5 text-gray-500" />
            <input
              autoFocus
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Filter runs…"
              className="flex-1 bg-transparent text-xs text-gray-300 placeholder:text-gray-600 outline-none"
            />
          </div>

          {/* Grouped list */}
          <div className="max-h-80 overflow-y-auto py-1">
            {filteredGroups.length === 0 ? (
              <p className="px-3 py-4 text-center text-xs text-gray-600">No matching runs</p>
            ) : (
              filteredGroups.map(([strategyName, groupRuns]) => {
                const isExpanded = expandedGroups.has(strategyName);
                return (
                  <div key={strategyName}>
                    {/* Strategy group header */}
                    <button
                      onClick={() => toggleGroup(strategyName)}
                      className="flex w-full items-center gap-2 px-3 py-1.5 text-left hover:bg-gray-800/50"
                    >
                      <ChevronRight
                        className={`h-3 w-3 shrink-0 text-gray-500 transition-transform ${isExpanded ? "rotate-90" : ""}`}
                      />
                      <span className="text-xs font-semibold text-gray-300">{strategyName}</span>
                      <span className="ml-auto text-[10px] text-gray-600">{groupRuns.length} runs</span>
                    </button>

                    {/* Runs within this strategy */}
                    {isExpanded && (
                      <div className="pb-1">
                        {groupRuns.map((run) => {
                          const meta = run.meta as Record<string, unknown>;
                          const retPct = meta.total_return_pct as number | undefined;
                          const trades = meta.total_trades as number | undefined;
                          const isActive =
                            selected?.run_id === run.run_id &&
                            selected?.strategy_name === run.strategy_name;

                          return (
                            <button
                              key={run.run_id}
                              onClick={() => handleSelect(run)}
                              className={`flex w-full items-center gap-3 px-3 py-1.5 pl-8 text-left transition ${
                                isActive
                                  ? "bg-blue-900/30 text-blue-300"
                                  : "text-gray-400 hover:bg-gray-800/50 hover:text-gray-200"
                              }`}
                            >
                              <span className="text-xs tabular-nums">
                                {formatRunId(run.run_id)}
                              </span>
                              <span className="flex-1" />
                              {trades != null && (
                                <span className="text-[10px] text-gray-600">{trades}t</span>
                              )}
                              {retPct != null && (
                                <span
                                  className={`min-w-[48px] text-right text-[10px] font-medium ${
                                    retPct >= 0 ? "text-green-400" : "text-red-400"
                                  }`}
                                >
                                  {retPct >= 0 ? "+" : ""}{retPct.toFixed(1)}%
                                </span>
                              )}
                            </button>
                          );
                        })}
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </div>
      )}
    </div>
  );
}
