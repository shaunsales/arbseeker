import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { listBacktests, viewBacktest, type BacktestRun } from "@/api/backtest";
import { Skeleton } from "@/components/ui/skeleton";
import BacktestViewer from "@/components/backtest/BacktestViewer";
import BacktestRunSelector, { BacktestRunBar } from "@/components/backtest/BacktestRunSelector";

export default function BacktestPage() {
  const [selectedRun, setSelectedRun] = useState<BacktestRun | null>(null);
  const [showSelector, setShowSelector] = useState(true);

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

  const handleSelectRun = (run: BacktestRun) => {
    setSelectedRun(run);
    setShowSelector(false);
  };

  // Loading state
  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Skeleton className="h-8 w-80" />
      </div>
    );
  }

  // State 1: Full-page selector panel
  if (showSelector || !selectedRun) {
    return (
      <BacktestRunSelector
        runs={runs}
        selected={selectedRun}
        onSelect={handleSelectRun}
      />
    );
  }

  // State 2: Compact bar + viewer
  return (
    <div className="flex h-full flex-col">
      <BacktestRunBar
        selected={selectedRun}
        onChangeRun={() => setShowSelector(true)}
      />

      <div className="flex-1 overflow-y-auto p-4">
        {viewLoading ? (
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
