import { useMemo } from "react";
import type { StrategyListItem } from "@/types/api";
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

interface Props {
  strategies: StrategyListItem[];
  selected: string | null;
  onSelect: (className: string) => void;
  /** Header title override. Defaults to "Select Strategy". */
  title?: string;
  /** Subtitle override. If omitted, shows strategy count. */
  subtitle?: string;
  /** If true, only show strategies with has_data_spec. */
  dataReadyOnly?: boolean;
}

// ── Compact top bar shown when a strategy is loaded ──

export function StrategyBar({
  selected,
  strategies,
  onChangeStrategy,
}: {
  selected: string;
  strategies: StrategyListItem[];
  onChangeStrategy: () => void;
}) {
  const strat = strategies.find((s) => s.class_name === selected);

  return (
    <div className="flex items-center gap-3 border-b border-gray-800 px-4 py-2">
      <button
        onClick={onChangeStrategy}
        className="flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs text-gray-400 hover:bg-gray-800 hover:text-gray-200 transition"
      >
        <ArrowLeft className="h-3.5 w-3.5" />
        Change
      </button>
      <div className="h-4 w-px bg-gray-800" />
      <FlaskConical className="h-3.5 w-3.5 text-gray-500" />
      <span className="text-xs font-semibold text-gray-300">{selected}</span>
      {strat && (
        <span className="text-[10px] text-gray-500">{strat.module}</span>
      )}
      {strat?.last_modified && (
        <span className="flex items-center gap-1 text-[10px] text-gray-600">
          <Clock className="h-2.5 w-2.5" />
          {timeAgo(strat.last_modified)}
        </span>
      )}
      {strat && (
        <span
          className={`inline-flex items-center gap-1 text-[10px] ${
            strat.has_data_spec ? "text-blue-400" : "text-gray-600"
          }`}
        >
          <span className={`inline-block h-1.5 w-1.5 rounded-full ${strat.has_data_spec ? "bg-blue-400" : "bg-gray-600"}`} />
          {strat.has_data_spec ? "Data ready" : "No data"}
        </span>
      )}
    </div>
  );
}

// ── Full-page strategy selector panel ──

export default function StrategySelector({
  strategies,
  selected,
  onSelect,
  title = "Select Strategy",
  subtitle,
  dataReadyOnly = false,
}: Props) {
  const filtered = useMemo(() => {
    const list = dataReadyOnly ? strategies.filter((s) => s.has_data_spec) : strategies;
    return [...list].sort((a, b) => {
      if (a.last_modified && b.last_modified) return b.last_modified.localeCompare(a.last_modified);
      if (a.last_modified) return -1;
      if (b.last_modified) return 1;
      return a.class_name.localeCompare(b.class_name);
    });
  }, [strategies, dataReadyOnly]);

  const countLabel = subtitle ??
    `${filtered.length} strateg${filtered.length !== 1 ? "ies" : "y"} available`;

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="border-b border-gray-800 px-6 py-4">
        <h1 className="text-sm font-semibold text-gray-200">{title}</h1>
        <p className="text-xs text-gray-500">{countLabel}</p>
      </div>

      {/* Strategy list */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        {filtered.length === 0 ? (
          <div className="py-12 text-center">
            <p className="text-sm text-gray-500">No strategies found.</p>
            <p className="mt-1 text-xs text-gray-600">
              Create a strategy class in <code className="text-gray-400">strategies/</code> that
              extends <code className="text-gray-400">SingleAssetStrategy</code>.
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            {filtered.map((s) => {
              const isActive = selected === s.class_name;
              return (
                <button
                  key={s.class_name}
                  onClick={() => onSelect(s.class_name)}
                  className={`flex w-full items-center gap-4 rounded-lg border px-5 py-3.5 text-left transition ${
                    isActive
                      ? "border-blue-800 bg-blue-900/20 text-blue-300"
                      : "border-gray-800 bg-gray-900/50 text-gray-300 hover:border-gray-700 hover:bg-gray-800/40"
                  }`}
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
                  <span
                    className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[10px] font-medium ${
                      s.has_data_spec
                        ? "bg-blue-900/30 text-blue-400"
                        : "bg-gray-800 text-gray-500"
                    }`}
                  >
                    <span className={`inline-block h-1.5 w-1.5 rounded-full ${s.has_data_spec ? "bg-blue-400" : "bg-gray-600"}`} />
                    {s.has_data_spec ? "Data ready" : "No data"}
                  </span>
                </button>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
