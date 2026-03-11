import { useState, useRef, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { listStrategies } from "@/api/strategy";
import { ChevronDown } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import StrategyDetail from "@/components/strategy/StrategyDetail";

export default function StrategyPage() {
  const [selected, setSelected] = useState<string | null>(null);
  const [pickerOpen, setPickerOpen] = useState(false);
  const pickerRef = useRef<HTMLDivElement>(null);

  const { data: strategies = [], isLoading } = useQuery({
    queryKey: ["strategies"],
    queryFn: listStrategies,
  });

  // Close picker on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (pickerRef.current && !pickerRef.current.contains(e.target as Node)) {
        setPickerOpen(false);
      }
    }
    if (pickerOpen) document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [pickerOpen]);

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      {/* Strategy picker bar */}
      <div className="flex-shrink-0 border-b border-gray-800 bg-gray-900/80 px-6 py-2.5">
        <div className="relative inline-block" ref={pickerRef}>
          <button
            onClick={() => setPickerOpen((v) => !v)}
            className="inline-flex items-center gap-2 rounded-md border border-gray-700 bg-gray-800 px-3 py-1.5 text-sm font-medium text-gray-200 transition hover:border-gray-600 hover:bg-gray-750"
          >
            {selected ?? "Select Strategy"}
            <ChevronDown className={`h-3.5 w-3.5 text-gray-400 transition ${pickerOpen ? "rotate-180" : ""}`} />
          </button>

          {pickerOpen && (
            <div className="absolute left-0 top-full z-50 mt-1 w-72 rounded-lg border border-gray-700 bg-gray-900 shadow-xl">
              <div className="px-3 py-2 text-[11px] uppercase tracking-wide text-gray-500">
                Select Strategy
              </div>
              <div className="max-h-64 overflow-y-auto px-1 pb-1">
                {isLoading ? (
                  <div className="space-y-1.5 px-2 py-1">
                    {[1, 2, 3].map((i) => (
                      <Skeleton key={i} className="h-8 w-full rounded-md" />
                    ))}
                  </div>
                ) : strategies.length === 0 ? (
                  <p className="px-3 py-3 text-xs text-gray-500">
                    No strategies found.
                  </p>
                ) : (
                  strategies.map((s) => (
                    <button
                      key={s.class_name}
                      onClick={() => {
                        setSelected(s.class_name);
                        setPickerOpen(false);
                      }}
                      className={`flex w-full items-center justify-between rounded-md px-3 py-1.5 text-left text-sm transition ${
                        selected === s.class_name
                          ? "bg-gray-800 text-white"
                          : "text-gray-300 hover:bg-gray-800/60 hover:text-white"
                      }`}
                    >
                      <div>
                        <span className="font-medium">{s.class_name}</span>
                        <span className="ml-1.5 text-xs text-gray-500">{s.module}</span>
                      </div>
                      <span
                        className={`h-2 w-2 flex-shrink-0 rounded-full ${
                          s.has_data_spec ? "bg-blue-400" : "bg-gray-600"
                        }`}
                      />
                    </button>
                  ))
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {selected ? (
          <StrategyDetail className={selected} />
        ) : (
          <div className="flex h-full items-center justify-center">
            <div className="text-center text-gray-500">
              <div className="mb-3 text-4xl">⚙</div>
              <p className="text-sm">
                Select a strategy to view its data requirements and build data files
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
