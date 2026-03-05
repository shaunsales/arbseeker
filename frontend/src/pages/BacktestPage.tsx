import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { listBacktests, viewBacktest, type BacktestRun } from "@/api/backtest";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import BacktestViewer from "@/components/backtest/BacktestViewer";

export default function BacktestPage() {
  const [selectedRun, setSelectedRun] = useState<BacktestRun | null>(null);

  const { data, isLoading } = useQuery({
    queryKey: ["backtest-runs"],
    queryFn: listBacktests,
  });

  const { data: viewData, isLoading: viewLoading } = useQuery({
    queryKey: [
      "backtest-view",
      selectedRun?.strategy_name,
      selectedRun?.run_id,
    ],
    queryFn: () =>
      viewBacktest(selectedRun!.strategy_name, selectedRun!.run_id),
    enabled: !!selectedRun,
  });

  return (
    <div className="flex h-full">
      {/* Sidebar */}
      <div className="flex w-72 flex-shrink-0 flex-col border-r border-gray-800">
        <div className="border-b border-gray-800 px-4 py-3">
          <h2 className="text-sm font-semibold text-gray-200">
            Backtest Runs
          </h2>
          <p className="text-xs text-gray-500">Previous backtest results</p>
        </div>

        <ScrollArea className="flex-1 p-2">
          {isLoading ? (
            <div className="space-y-2 p-2">
              {Array.from({ length: 4 }).map((_, i) => (
                <Skeleton key={i} className="h-10 w-full" />
              ))}
            </div>
          ) : data && data.runs.length > 0 ? (
            <div className="space-y-0.5">
              {data.runs.map((run) => {
                const isActive =
                  selectedRun?.run_id === run.run_id &&
                  selectedRun?.strategy_name === run.strategy_name;
                const meta = run.meta as Record<string, unknown>;
                return (
                  <button
                    key={`${run.strategy_name}-${run.run_id}`}
                    onClick={() => setSelectedRun(run)}
                    className={`flex w-full flex-col rounded px-3 py-2 text-left transition ${
                      isActive
                        ? "bg-blue-900/40 text-blue-300"
                        : "text-gray-400 hover:bg-gray-800/50 hover:text-gray-200"
                    }`}
                  >
                    <div className="flex items-center gap-2 text-xs">
                      <span className="font-medium">{run.strategy_name}</span>
                      <Badge
                        variant="secondary"
                        className="ml-auto h-4 px-1.5 text-[10px]"
                      >
                        {run.run_id.slice(0, 8)}
                      </Badge>
                    </div>
                    {meta.total_return_pct != null && (
                      <span
                        className={`mt-0.5 text-[10px] ${
                          (meta.total_return_pct as number) >= 0
                            ? "text-green-400"
                            : "text-red-400"
                        }`}
                      >
                        {(meta.total_return_pct as number).toFixed(2)}% return
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          ) : (
            <p className="px-2 py-8 text-center text-xs text-gray-600">
              No backtest runs yet. Build strategy data first, then run a
              backtest.
            </p>
          )}
        </ScrollArea>
      </div>

      {/* Main content */}
      <div className="flex-1 overflow-y-auto p-6">
        {!selectedRun ? (
          <div className="flex h-full items-center justify-center">
            <p className="text-sm text-gray-500">
              Select a backtest run from the sidebar to view results
            </p>
          </div>
        ) : viewLoading ? (
          <div className="flex h-60 items-center justify-center text-sm text-gray-500">
            Loading backtest results...
          </div>
        ) : !viewData || viewData.error ? (
          <div className="flex h-40 items-center justify-center text-sm text-red-400">
            {viewData?.error || "Failed to load backtest"}
          </div>
        ) : (
          <BacktestViewer data={viewData} />
        )}
      </div>
    </div>
  );
}
