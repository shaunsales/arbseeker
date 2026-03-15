import { useState, useMemo } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { runBacktest } from "@/api/backtest";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import MonthRangePicker, { expandMonthRange } from "@/components/ui/month-range-picker";
import { Play, Loader2 } from "lucide-react";
import type { StrategyListItem } from "@/types/api";

interface Props {
  /** The pre-selected strategy (already chosen via selector). */
  strategy: StrategyListItem;
  onRunComplete?: (strategyName: string, runId: string) => void;
}

export default function RunBacktestForm({ strategy, onRunComplete }: Props) {
  const queryClient = useQueryClient();
  const [capital, setCapital] = useState("100000");
  const [commissionBps, setCommissionBps] = useState("3.5");
  const [slippageBps, setSlippageBps] = useState("2.0");
  const [fundingBps, setFundingBps] = useState("3.0");
  const [slEnabled, setSlEnabled] = useState(false);
  const [slPct, setSlPct] = useState("5.0");
  const [tslEnabled, setTslEnabled] = useState(false);
  const [tslPct, setTslPct] = useState("5.0");
  const [error, setError] = useState("");

  const dateRange = strategy.data_date_range ?? null;
  const availableMonths = useMemo(
    () => dateRange ? expandMonthRange(dateRange.start, dateRange.end) : [],
    [dateRange],
  );
  const [startDate, setStartDate] = useState(dateRange?.start ?? "");
  const [endDate, setEndDate] = useState(dateRange?.end ?? "");

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
    setError("");
    mutation.mutate({
      class_name: strategy.class_name,
      capital: parseFloat(capital) || 100_000,
      commission_bps: parseFloat(commissionBps) || 0,
      slippage_bps: parseFloat(slippageBps) || 0,
      funding_daily_bps: parseFloat(fundingBps) || 0,
      start_date: startDate || undefined,
      end_date: endDate || undefined,
      stop_loss_pct: slEnabled ? (parseFloat(slPct) || null) : null,
      trailing_stop_pct: tslEnabled ? (parseFloat(tslPct) || null) : null,
    });
  };

  return (
    <div className="space-y-4">
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
      {availableMonths.length > 0 ? (
        <MonthRangePicker
          months={availableMonths}
          startDate={startDate}
          endDate={endDate}
          onRangeChange={(s, e) => { setStartDate(s); setEndDate(e); }}
          showLegend={false}
        />
      ) : (
        <p className="text-[10px] text-yellow-500">No built data found — build data first</p>
      )}

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

      {/* Risk Management — SL / TSL */}
      <div>
        <label className="mb-1 block text-[11px] text-gray-500">Risk Management</label>
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setSlEnabled(!slEnabled)}
              className={`h-5 w-9 rounded-full transition-colors ${
                slEnabled ? "bg-blue-600" : "bg-gray-700"
              } relative`}
            >
              <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-transform ${
                slEnabled ? "left-[18px]" : "left-0.5"
              }`} />
            </button>
            <span className="text-[11px] text-gray-400 w-24">Stop Loss</span>
            {slEnabled && (
              <div className="flex items-center gap-1">
                <Input type="number" step="0.5" min="0.1" value={slPct} onChange={(e) => setSlPct(e.target.value)} className="h-7 w-20 text-xs" />
                <span className="text-[10px] text-gray-500">%</span>
              </div>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setTslEnabled(!tslEnabled)}
              className={`h-5 w-9 rounded-full transition-colors ${
                tslEnabled ? "bg-blue-600" : "bg-gray-700"
              } relative`}
            >
              <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-transform ${
                tslEnabled ? "left-[18px]" : "left-0.5"
              }`} />
            </button>
            <span className="text-[11px] text-gray-400 w-24">Trailing Stop</span>
            {tslEnabled && (
              <div className="flex items-center gap-1">
                <Input type="number" step="0.5" min="0.1" value={tslPct} onChange={(e) => setTslPct(e.target.value)} className="h-7 w-20 text-xs" />
                <span className="text-[10px] text-gray-500">%</span>
              </div>
            )}
          </div>
          {!slEnabled && !tslEnabled && (
            <p className="text-[10px] text-gray-600 italic">Using strategy defaults (if any)</p>
          )}
        </div>
      </div>

      {/* Error */}
      {error && <p className="text-xs text-red-400">{error}</p>}

      {/* Run button */}
      <Button onClick={handleRun} disabled={mutation.isPending || !startDate || !endDate} className="h-8 w-full text-xs" size="sm">
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
}
