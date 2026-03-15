import type { StrategyListItem } from "@/types/api";
import { FlaskConical, ArrowLeft } from "lucide-react";

interface Props {
  strategies: StrategyListItem[];
  selected: string | null;
  onSelect: (className: string) => void;
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
      {strat && (
        <span
          className={`inline-flex items-center gap-1 text-[10px] ${
            strat.has_data_spec ? "text-blue-400" : "text-gray-600"
          }`}
        >
          <span className={`inline-block h-1.5 w-1.5 rounded-full ${strat.has_data_spec ? "bg-blue-400" : "bg-gray-600"}`} />
          {strat.has_data_spec ? "Data spec ready" : "No data spec"}
        </span>
      )}
    </div>
  );
}

// ── Full-page strategy selector panel ──

export default function StrategySelector({ strategies, selected, onSelect }: Props) {
  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="border-b border-gray-800 px-6 py-4">
        <h1 className="text-sm font-semibold text-gray-200">Select Strategy</h1>
        <p className="text-xs text-gray-500">
          {strategies.length} strateg{strategies.length !== 1 ? "ies" : "y"} available
        </p>
      </div>

      {/* Strategy list */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        {strategies.length === 0 ? (
          <div className="py-12 text-center">
            <p className="text-sm text-gray-500">No strategies found.</p>
            <p className="mt-1 text-xs text-gray-600">
              Create a strategy class in <code className="text-gray-400">strategies/</code> that
              extends <code className="text-gray-400">SingleAssetStrategy</code>.
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            {strategies.map((s) => {
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
                  <div className="flex-1">
                    <span className="text-sm font-semibold">{s.class_name}</span>
                    <span className="ml-2 text-xs text-gray-500">{s.module}</span>
                  </div>
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
