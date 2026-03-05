import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getDataTree } from "@/api/data";
import DataTree from "@/components/data/DataTree";
import DataDownload from "@/components/data/DataDownload";
import DataPreview from "@/components/data/DataPreview";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";

interface Selection {
  venue: string;
  market: string;
  ticker: string;
  interval: string;
}

export default function DataPage() {
  const [selected, setSelected] = useState<Selection | null>(null);

  const { data, isLoading, refetch } = useQuery({
    queryKey: ["data-tree"],
    queryFn: getDataTree,
  });

  return (
    <div className="flex h-full">
      {/* Sidebar */}
      <div className="flex w-72 flex-shrink-0 flex-col border-r border-gray-800">
        <div className="border-b border-gray-800 px-4 py-3">
          <h2 className="text-sm font-semibold text-gray-200">Data Browser</h2>
          <p className="text-xs text-gray-500">Downloaded OHLCV data</p>
        </div>

        <ScrollArea className="flex-1 p-2">
          {isLoading ? (
            <div className="space-y-2 p-2">
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-6 w-full" />
              ))}
            </div>
          ) : data ? (
            <DataTree
              tree={data.data_tree}
              onSelect={(v, m, t, i) => setSelected({ venue: v, market: m, ticker: t, interval: i })}
              selected={selected}
            />
          ) : null}
        </ScrollArea>

        <div className="border-t border-gray-800 p-3">
          <DataDownload
            intervals={data?.intervals ?? []}
            onComplete={() => refetch()}
          />
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 overflow-y-auto p-6">
        {selected ? (
          <DataPreview
            venue={selected.venue}
            market={selected.market}
            ticker={selected.ticker}
            interval={selected.interval}
          />
        ) : (
          <div className="flex h-full items-center justify-center">
            <div className="text-center">
              <p className="text-sm text-gray-500">
                Select a dataset from the sidebar to preview
              </p>
              <p className="mt-1 text-xs text-gray-600">
                Or download new data using the panel below
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
