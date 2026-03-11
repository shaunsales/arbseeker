import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { getIndicatorRegistry } from "@/api/strategy";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Plus, X, ChevronDown, ChevronRight, Loader2 } from "lucide-react";
import type { IndicatorMeta } from "@/types/api";

export interface ActiveIndicator {
  id: string;
  name: string;
  params: Record<string, number>;
  meta: IndicatorMeta;
}

interface Props {
  activeIndicators: ActiveIndicator[];
  onAdd: (indicator: ActiveIndicator) => void;
  onRemove: (id: string) => void;
  onCompute: () => void;
  computing: boolean;
}

const CATEGORY_LABELS: Record<string, string> = {
  trend: "Trend / MA",
  momentum: "Momentum",
  volatility: "Volatility",
  volume: "Volume",
};

const CATEGORY_COLORS: Record<string, string> = {
  trend: "text-blue-400",
  momentum: "text-amber-400",
  volatility: "text-red-400",
  volume: "text-green-400",
};

let _nextId = 0;
function nextId() {
  return `ind_${++_nextId}`;
}

export default function IndicatorPicker({
  activeIndicators,
  onAdd,
  onRemove,
  onCompute,
  computing,
}: Props) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [expandedCat, setExpandedCat] = useState<string | null>(null);
  const [editingParams, setEditingParams] = useState<string | null>(null);
  const [paramValues, setParamValues] = useState<Record<string, number>>({});

  const { data: registry } = useQuery({
    queryKey: ["indicator-registry"],
    queryFn: getIndicatorRegistry,
    staleTime: Infinity,
  });

  const grouped = useMemo(() => {
    if (!registry) return {};
    const groups: Record<string, [string, IndicatorMeta][]> = {};
    for (const [key, meta] of Object.entries(registry)) {
      if (search && !meta.label.toLowerCase().includes(search.toLowerCase()) && !key.includes(search.toLowerCase())) {
        continue;
      }
      if (!groups[meta.category]) groups[meta.category] = [];
      groups[meta.category].push([key, meta]);
    }
    return groups;
  }, [registry, search]);

  function handleAddIndicator(key: string, meta: IndicatorMeta) {
    const defaults: Record<string, number> = {};
    for (const [pName, pMeta] of Object.entries(meta.params)) {
      defaults[pName] = pMeta.default;
    }
    onAdd({ id: nextId(), name: key, params: defaults, meta });
  }

  function toggleCategory(cat: string) {
    setExpandedCat(expandedCat === cat ? null : cat);
  }

  return (
    <div className="space-y-2">
      {/* Toggle button */}
      <div className="flex items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setOpen(!open)}
          className="h-7 gap-1.5 border-gray-700 bg-gray-800 text-xs text-gray-300 hover:bg-gray-700"
        >
          <Plus className="h-3 w-3" />
          Indicators
          {activeIndicators.length > 0 && (
            <Badge className="ml-1 h-4 bg-blue-600 px-1 text-[10px]">
              {activeIndicators.length}
            </Badge>
          )}
        </Button>

        {activeIndicators.length > 0 && (
          <Button
            size="sm"
            onClick={onCompute}
            disabled={computing}
            className="h-7 gap-1.5 bg-blue-600 text-xs hover:bg-blue-700"
          >
            {computing ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              "Apply"
            )}
          </Button>
        )}

        {/* Active indicator pills */}
        <div className="flex flex-wrap gap-1">
          {activeIndicators.map((ind) => (
            <span
              key={ind.id}
              className="inline-flex items-center gap-1 rounded bg-gray-800 px-1.5 py-0.5 text-[10px] text-gray-300 border border-gray-700"
            >
              <span className={CATEGORY_COLORS[ind.meta.category] ?? "text-gray-400"}>
                {ind.meta.label}
              </span>
              {Object.keys(ind.params).length > 0 && (
                <span className="text-gray-500">
                  ({Object.values(ind.params).join(",")})
                </span>
              )}
              <button
                onClick={() => onRemove(ind.id)}
                className="ml-0.5 text-gray-500 hover:text-red-400"
              >
                <X className="h-2.5 w-2.5" />
              </button>
            </span>
          ))}
        </div>
      </div>

      {/* Picker panel */}
      {open && (
        <div className="rounded-lg border border-gray-700 bg-gray-900 p-3">
          <Input
            placeholder="Search indicators..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="mb-2 h-7 border-gray-700 bg-gray-800 text-xs"
          />

          <div className="max-h-64 overflow-y-auto space-y-1">
            {Object.entries(CATEGORY_LABELS).map(([cat, label]) => {
              const items = grouped[cat];
              if (!items?.length) return null;
              const isExpanded = expandedCat === cat || !!search;

              return (
                <div key={cat}>
                  <button
                    onClick={() => toggleCategory(cat)}
                    className="flex w-full items-center gap-1.5 rounded px-2 py-1 text-xs font-medium hover:bg-gray-800"
                  >
                    {isExpanded ? (
                      <ChevronDown className="h-3 w-3 text-gray-500" />
                    ) : (
                      <ChevronRight className="h-3 w-3 text-gray-500" />
                    )}
                    <span className={CATEGORY_COLORS[cat]}>{label}</span>
                    <span className="text-gray-600">({items.length})</span>
                  </button>

                  {isExpanded && (
                    <div className="ml-4 space-y-0.5">
                      {items.map(([key, meta]) => {
                        const isEditing = editingParams === key;
                        return (
                          <div key={key}>
                            <div className="flex items-center gap-2 rounded px-2 py-1 text-xs hover:bg-gray-800/80">
                              <button
                                onClick={() => handleAddIndicator(key, meta)}
                                className="flex-1 text-left text-gray-300 hover:text-white"
                              >
                                {meta.label}
                              </button>
                              <Badge
                                className={`h-4 px-1 text-[9px] ${
                                  meta.display === "overlay"
                                    ? "bg-purple-900/50 text-purple-300"
                                    : "bg-gray-700 text-gray-400"
                                }`}
                              >
                                {meta.display}
                              </Badge>
                              {Object.keys(meta.params).length > 0 && (
                                <button
                                  onClick={() => {
                                    if (isEditing) {
                                      setEditingParams(null);
                                    } else {
                                      const defaults: Record<string, number> = {};
                                      for (const [pN, pM] of Object.entries(meta.params)) {
                                        defaults[pN] = pM.default;
                                      }
                                      setParamValues(defaults);
                                      setEditingParams(key);
                                    }
                                  }}
                                  className="text-gray-500 hover:text-gray-300 text-[10px]"
                                >
                                  ⚙
                                </button>
                              )}
                            </div>

                            {/* Param editor */}
                            {isEditing && (
                              <div className="ml-2 mb-1 flex flex-wrap items-center gap-2 rounded bg-gray-800/50 px-2 py-1.5">
                                {Object.entries(meta.params).map(([pName, pMeta]) => (
                                  <label key={pName} className="flex items-center gap-1 text-[10px] text-gray-400">
                                    {pName}:
                                    <input
                                      type="number"
                                      min={pMeta.min}
                                      max={pMeta.max}
                                      step={pMeta.type === "float" ? 0.01 : 1}
                                      value={paramValues[pName] ?? pMeta.default}
                                      onChange={(e) =>
                                        setParamValues((prev) => ({
                                          ...prev,
                                          [pName]: Number(e.target.value),
                                        }))
                                      }
                                      className="w-16 rounded border border-gray-600 bg-gray-900 px-1 py-0.5 text-[10px] text-gray-200"
                                    />
                                  </label>
                                ))}
                                <Button
                                  size="sm"
                                  onClick={() => {
                                    onAdd({
                                      id: nextId(),
                                      name: key,
                                      params: { ...paramValues },
                                      meta,
                                    });
                                    setEditingParams(null);
                                  }}
                                  className="h-5 px-2 text-[10px] bg-blue-600 hover:bg-blue-700"
                                >
                                  Add
                                </Button>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
