import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { listBacktests, viewBacktest, type BacktestRun } from "@/api/backtest";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import BacktestViewer from "@/components/backtest/BacktestViewer";
import RunBacktestForm from "@/components/backtest/RunBacktestForm";

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

  const handleRunComplete = (strategyName: string, runId: string) => {
    setSelectedRun({ strategy_name: strategyName, run_id: runId, meta: {}, folder: "" });
  };

  const runs = data?.runs ?? [];

  const handleSelectRun = (value: string) => {
    const run = runs.find(
      (r) => `${r.strategy_name}::${r.run_id}` === value,
    );
    if (run) setSelectedRun(run);
  };

  const selectedKey = selectedRun
    ? `${selectedRun.strategy_name}::${selectedRun.run_id}`
    : "";

  return (
    <div className="flex h-full flex-col">
      {/* Top bar: run selector + run backtest */}
      <div className="flex items-center gap-3 border-b border-gray-800 px-4 py-2">
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-gray-500">Run</span>
          {isLoading ? (
            <Skeleton className="h-8 w-64" />
          ) : (
            <Select value={selectedKey} onValueChange={handleSelectRun}>
              <SelectTrigger className="h-8 w-80 text-xs">
                <SelectValue placeholder="Select a backtest run…" />
              </SelectTrigger>
              <SelectContent>
                {runs.map((run) => {
                  const meta = run.meta as Record<string, unknown>;
                  const retPct = meta.total_return_pct as number | undefined;
                  return (
                    <SelectItem
                      key={`${run.strategy_name}::${run.run_id}`}
                      value={`${run.strategy_name}::${run.run_id}`}
                    >
                      <span className="font-medium">{run.strategy_name}</span>
                      <span className="ml-2 text-gray-500">{run.run_id.slice(0, 8)}</span>
                      {retPct != null && (
                        <span
                          className={`ml-2 ${retPct >= 0 ? "text-green-400" : "text-red-400"}`}
                        >
                          {retPct.toFixed(1)}%
                        </span>
                      )}
                    </SelectItem>
                  );
                })}
              </SelectContent>
            </Select>
          )}
        </div>

        <div className="ml-auto">
          <RunBacktestForm
            strategies={data?.strategies ?? []}
            onRunComplete={handleRunComplete}
          />
        </div>
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
