import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { listBacktests, viewBacktest, type BacktestRun } from "@/api/backtest";
import { Skeleton } from "@/components/ui/skeleton";
import BacktestViewer from "@/components/backtest/BacktestViewer";
import BacktestRunSelector from "@/components/backtest/BacktestRunSelector";

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

  const runs = data?.runs ?? [];

  return (
    <div className="flex h-full flex-col">
      {/* Top bar: grouped run selector */}
      <div className="flex items-center gap-3 border-b border-gray-800 px-4 py-2">
        {isLoading ? (
          <Skeleton className="h-8 w-80" />
        ) : (
          <BacktestRunSelector
            runs={runs}
            selected={selectedRun}
            onSelect={setSelectedRun}
          />
        )}
      </div>

      {/* Main content — full width */}
      <div className="flex-1 overflow-y-auto p-4">
        {!selectedRun ? (
          <div className="flex h-full items-center justify-center">
            <p className="text-sm text-gray-500">
              Select a backtest run to view results
            </p>
          </div>
        ) : viewLoading ? (
          <div className="flex h-60 items-center justify-center text-sm text-gray-500">
            Loading backtest results…
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
