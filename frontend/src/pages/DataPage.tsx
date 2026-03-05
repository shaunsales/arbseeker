import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { getDataTree } from "@/api/data";
import DataTree from "@/components/data/DataTree";
import DataPreview from "@/components/data/DataPreview";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { Download } from "lucide-react";

interface Selection {
  venue: string;
  market: string;
  ticker: string;
  interval: string;
}

export default function DataPage() {
  const [selected, setSelected] = useState<Selection | null>(null);

  const { data, isLoading } = useQuery({
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
          <Link
            to="/download"
            className="flex w-full items-center justify-center gap-2 rounded-md border border-gray-700 bg-gray-800 px-3 py-2 text-xs font-medium text-gray-300 transition hover:bg-gray-700 hover:text-white"
          >
            <Download className="h-3.5 w-3.5" />
            Download New Data
          </Link>
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
                Or use the <Link to="/download" className="text-blue-400 hover:underline">Download</Link> page to get new data
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
