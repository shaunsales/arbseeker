import { useQuery } from "@tanstack/react-query";
import { listBacktests } from "@/api/backtest";
import RunBacktestForm from "@/components/backtest/RunBacktestForm";

export default function BacktestExecutionPage() {
  const { data } = useQuery({
    queryKey: ["backtest-runs"],
    queryFn: listBacktests,
  });

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-gray-800 px-6 py-4">
        <h1 className="text-sm font-semibold text-gray-200">Backtest Execution</h1>
        <p className="text-xs text-gray-500">Configure and run backtests. Optimizer coming soon.</p>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="mx-auto max-w-lg">
          <RunBacktestForm
            strategies={data?.strategies ?? []}
            variant="page"
          />
        </div>
      </div>
    </div>
  );
}
