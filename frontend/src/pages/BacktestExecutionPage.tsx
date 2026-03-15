import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { listBacktests } from "@/api/backtest";
import RunBacktestForm from "@/components/backtest/RunBacktestForm";
import StrategySelector, { StrategyBar } from "@/components/strategy/StrategySelector";
import type { StrategyListItem } from "@/types/api";

export default function BacktestExecutionPage() {
  const [selected, setSelected] = useState<string | null>(null);
  const [showSelector, setShowSelector] = useState(true);

  const { data } = useQuery({
    queryKey: ["backtest-runs"],
    queryFn: listBacktests,
  });

  const strategies: StrategyListItem[] = data?.strategies ?? [];
  const selectedStrat = strategies.find((s) => s.class_name === selected);

  const handleSelect = (className: string) => {
    setSelected(className);
    setShowSelector(false);
  };

  // Phase 1: Strategy selector
  if (showSelector || !selected || !selectedStrat) {
    return (
      <StrategySelector
        strategies={strategies}
        selected={selected}
        onSelect={handleSelect}
        title="Select Strategy"
        dataReadyOnly
      />
    );
  }

  // Phase 2: Compact bar + config form
  return (
    <div className="flex h-full flex-col">
      <StrategyBar
        selected={selected}
        strategies={strategies}
        onChangeStrategy={() => setShowSelector(true)}
      />
      <div className="flex-1 overflow-y-auto p-6">
        <div className="mx-auto max-w-lg">
          <RunBacktestForm
            key={selected}
            strategy={selectedStrat}
          />
        </div>
      </div>
    </div>
  );
}
