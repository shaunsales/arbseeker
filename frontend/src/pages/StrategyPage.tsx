import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { listStrategies } from "@/api/strategy";
import { Skeleton } from "@/components/ui/skeleton";
import StrategyDetail from "@/components/strategy/StrategyDetail";
import StrategySelector, { StrategyBar } from "@/components/strategy/StrategySelector";

export default function StrategyPage() {
  const [selected, setSelected] = useState<string | null>(null);
  const [showSelector, setShowSelector] = useState(true);

  const { data: strategies = [], isLoading } = useQuery({
    queryKey: ["strategies"],
    queryFn: listStrategies,
  });

  const handleSelect = (className: string) => {
    setSelected(className);
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

  // State 1: Full-page selector
  if (showSelector || !selected) {
    return (
      <StrategySelector
        strategies={strategies}
        selected={selected}
        onSelect={handleSelect}
      />
    );
  }

  // State 2: Compact bar + detail
  return (
    <div className="flex h-full flex-col">
      <StrategyBar
        selected={selected}
        strategies={strategies}
        onChangeStrategy={() => setShowSelector(true)}
      />
      <div className="flex-1 overflow-y-auto p-6">
        <StrategyDetail className={selected} />
      </div>
    </div>
  );
}
