import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { runBacktest } from "@/api/backtest";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Play, Loader2, ChevronDown, ChevronUp } from "lucide-react";

interface Strategy {
  class_name: string;
  module: string;
  has_data_spec: boolean;
  data_date_range?: { start: string; end: string } | null;
}

interface Props {
  strategies: Strategy[];
  onRunComplete?: (strategyName: string, runId: string) => void;
  /** "popover" = compact button+dropdown (default), "page" = full inline form */
  variant?: "popover" | "page";
}

export default function RunBacktestForm({ strategies, onRunComplete, variant = "popover" }: Props) {
  const queryClient = useQueryClient();
  const [expanded, setExpanded] = useState(variant === "page");
  const [className, setClassName] = useState("");
  const [capital, setCapital] = useState("100000");
  const [commissionBps, setCommissionBps] = useState("3.5");
  const [slippageBps, setSlippageBps] = useState("2.0");
  const [fundingBps, setFundingBps] = useState("5.0");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [error, setError] = useState("");

  const eligible = strategies.filter((s) => s.has_data_spec);

  // Get the selected strategy's data date range
  const selectedStrategy = eligible.find((s) => s.class_name === className);
  const dateRange = selectedStrategy?.data_date_range ?? null;

  // When strategy changes, auto-populate dates from available data range
  const handleStrategyChange = (name: string) => {
    setClassName(name);
    const strat = eligible.find((s) => s.class_name === name);
    if (strat?.data_date_range) {
      setStartDate(strat.data_date_range.start);
      setEndDate(strat.data_date_range.end);
    } else {
      setStartDate("");
      setEndDate("");
    }
  };

  const mutation = useMutation({
    mutationFn: runBacktest,
    onSuccess: (data) => {
      if (data.success && data.strategy_name && data.run_id) {
        queryClient.invalidateQueries({ queryKey: ["backtest-runs"] });
        onRunComplete?.(data.strategy_name, data.run_id);
        setError("");
      } else {
        setError(data.error || "Backtest failed");
      }
    },
    onError: (err) => {
      setError(String(err));
    },
  });

  const handleRun = () => {
    if (!className) {
      setError("Select a strategy");
      return;
    }
    setError("");
    mutation.mutate({
      class_name: className,
      capital: parseFloat(capital) || 100_000,
      commission_bps: parseFloat(commissionBps) || 0,
      slippage_bps: parseFloat(slippageBps) || 0,
      funding_daily_bps: parseFloat(fundingBps) || 0,
      start_date: startDate || undefined,
      end_date: endDate || undefined,
    });
  };

  const formFields = (
    <div className="space-y-3">
      {/* Strategy select */}
      <div>
        <label className="mb-1 block text-[11px] text-gray-500">Strategy</label>
        <Select value={className} onValueChange={handleStrategyChange}>
          <SelectTrigger className="h-8 text-xs">
            <SelectValue placeholder="Select strategy…" />
          </SelectTrigger>
          <SelectContent>
            {eligible.map((s) => (
              <SelectItem key={s.class_name} value={s.class_name}>
                {s.class_name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {eligible.length === 0 && (
          <p className="mt-1 text-[10px] text-yellow-500">
            No strategies with built data found
          </p>
        )}
      </div>

      {/* Capital */}
      <div>
        <label className="mb-1 block text-[11px] text-gray-500">Starting Capital ($)</label>
        <Input
          type="number"
          value={capital}
          onChange={(e) => setCapital(e.target.value)}
          className="h-8 text-xs"
        />
      </div>

      {/* Date range */}
      <div>
        <label className="mb-1 block text-[11px] text-gray-500">
          Date Range
          {dateRange && (
            <span className="ml-1 font-normal text-gray-600">
              (data: {dateRange.start} → {dateRange.end})
            </span>
          )}
        </label>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <span className="text-[10px] text-gray-600">Start month</span>
            <input
              type="month"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              min={dateRange?.start}
              max={dateRange?.end}
              className="h-7 w-full rounded border border-gray-700 bg-gray-800 px-2 text-xs text-gray-200 focus:border-blue-500 focus:outline-none"
            />
          </div>
          <div>
            <span className="text-[10px] text-gray-600">End month</span>
            <input
              type="month"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              min={dateRange?.start}
              max={dateRange?.end}
              className="h-7 w-full rounded border border-gray-700 bg-gray-800 px-2 text-xs text-gray-200 focus:border-blue-500 focus:outline-none"
            />
          </div>
        </div>
        {!dateRange && className && (
          <p className="mt-0.5 text-[10px] text-yellow-500">No built data found — build data first</p>
        )}
      </div>

      {/* Cost model — compact row */}
      <div>
        <label className="mb-1 block text-[11px] text-gray-500">Costs (bps)</label>
        <div className="grid grid-cols-3 gap-2">
          <div>
            <span className="text-[10px] text-gray-600">Commission</span>
            <Input type="number" step="0.5" value={commissionBps} onChange={(e) => setCommissionBps(e.target.value)} className="h-7 text-xs" />
          </div>
          <div>
            <span className="text-[10px] text-gray-600">Slippage</span>
            <Input type="number" step="0.5" value={slippageBps} onChange={(e) => setSlippageBps(e.target.value)} className="h-7 text-xs" />
          </div>
          <div>
            <span className="text-[10px] text-gray-600">Funding/day</span>
            <Input type="number" step="0.5" value={fundingBps} onChange={(e) => setFundingBps(e.target.value)} className="h-7 text-xs" />
          </div>
        </div>
      </div>

      {/* Error */}
      {error && <p className="text-xs text-red-400">{error}</p>}

      {/* Run button */}
      <Button onClick={handleRun} disabled={mutation.isPending || !className} className="h-8 w-full text-xs" size="sm">
        {mutation.isPending ? (
          <><Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />Running…</>
        ) : (
          <><Play className="mr-1.5 h-3.5 w-3.5" />Run Backtest</>
        )}
      </Button>

      {/* Success message */}
      {mutation.isSuccess && mutation.data?.success && (
        <p className="text-xs text-green-400">
          ✓ {mutation.data.total_trades} trades — {mutation.data.strategy_name}/{mutation.data.run_id?.slice(0, 8)}
        </p>
      )}
    </div>
  );

  // Page variant: render form fields directly
  if (variant === "page") {
    return formFields;
  }

  // Popover variant: button + absolute dropdown
  return (
    <div className="relative">
      <Button variant="outline" size="sm" onClick={() => setExpanded(!expanded)} className="h-8 gap-1.5 text-xs">
        {mutation.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
        Run Backtest
        {expanded ? <ChevronUp className="h-3 w-3 text-gray-500" /> : <ChevronDown className="h-3 w-3 text-gray-500" />}
      </Button>

      {expanded && (
        <div className="absolute right-0 top-10 z-50 w-80 rounded-lg border border-gray-700 bg-gray-900 p-4 shadow-xl">
          {formFields}
        </div>
      )}
    </div>
  );
}
