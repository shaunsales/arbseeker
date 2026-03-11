import { useState, useCallback, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { getStrategySpec } from "@/api/strategy";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { useSidebar } from "@/components/ui/sidebar";
import StrategySpec from "./StrategySpec";
import CurrentData from "./CurrentData";
import BuildControls from "./BuildControls";

interface Props {
  className: string;
}

export default function StrategyDetail({ className }: Props) {
  const [chartExpanded, setChartExpanded] = useState(false);
  const { setOpen: setSidebarOpen } = useSidebar();
  const sidebarWasOpen = useRef(true);

  const handleToggleExpand = useCallback(() => {
    setChartExpanded((prev) => {
      const next = !prev;
      if (next) {
        // Collapsing sidebar when expanding chart
        sidebarWasOpen.current = true; // sidebar state before expand
        setSidebarOpen(false);
      } else {
        // Restore sidebar when collapsing chart
        setSidebarOpen(sidebarWasOpen.current);
      }
      return next;
    });
  }, [setSidebarOpen]);

  const {
    data: status,
    isLoading,
    refetch,
  } = useQuery({
    queryKey: ["strategy-spec", className],
    queryFn: () => getStrategySpec(className),
  });

  if (isLoading || !status) {
    return (
      <div className="mx-auto max-w-3xl space-y-4">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-40 w-full rounded-lg" />
        <Skeleton className="h-40 w-full rounded-lg" />
      </div>
    );
  }

  return (
    <div className={`mx-auto space-y-4 transition-all ${chartExpanded ? "max-w-7xl" : "max-w-3xl"}`}>
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-lg font-semibold text-gray-100">
            {status.class_name}
          </h2>
          <div className="mt-1 flex items-center gap-2 text-sm text-gray-400">
            <span>
              Venue <strong className="text-gray-200">{status.spec.venue}</strong>
            </span>
            <span className="text-gray-600">·</span>
            <span>
              Market <strong className="text-gray-200">{status.spec.market}</strong>
            </span>
            <span className="text-gray-600">·</span>
            <span>
              Ticker <strong className="text-gray-200">{status.spec.ticker}</strong>
            </span>
          </div>
        </div>
        <Badge
          variant={status.has_manifest ? "default" : "secondary"}
          className={
            status.has_manifest
              ? "bg-green-900/50 text-green-300 hover:bg-green-900/50"
              : ""
          }
        >
          {status.has_manifest ? "Data Ready" : "No Data"}
        </Badge>
      </div>

      {/* Data Spec */}
      <StrategySpec intervals={status.spec.intervals} />

      {/* Current Data or Build Controls */}
      {status.has_manifest && status.manifest ? (
        <CurrentData
          className={className}
          manifest={status.manifest}
          errors={status.errors}
          onDeleted={refetch}
          chartExpanded={chartExpanded}
          onToggleExpand={handleToggleExpand}
        />
      ) : (
        <BuildControls className={className} onBuilt={refetch} />
      )}
    </div>
  );
}
