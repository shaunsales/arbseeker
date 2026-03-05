import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getBasisFilesList, getBasisPreview } from "@/api/basis";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import BasisChart from "@/components/basis/BasisChart";

export default function BasisPage() {
  const [selected, setSelected] = useState<{
    ticker: string;
    interval: string;
  } | null>(null);

  const { data: files, isLoading } = useQuery({
    queryKey: ["basis-files"],
    queryFn: getBasisFilesList,
  });

  const { data: preview, isLoading: previewLoading } = useQuery({
    queryKey: ["basis-preview", selected?.ticker, selected?.interval],
    queryFn: () => getBasisPreview(selected!.ticker, selected!.interval),
    enabled: !!selected,
  });

  return (
    <div className="flex h-full">
      {/* Sidebar */}
      <div className="flex w-64 flex-shrink-0 flex-col border-r border-gray-800">
        <div className="border-b border-gray-800 px-4 py-3">
          <h2 className="text-sm font-semibold text-gray-200">Basis Files</h2>
          <p className="text-xs text-gray-500">Spot vs futures basis data</p>
        </div>

        <ScrollArea className="flex-1 p-2">
          {isLoading ? (
            <div className="space-y-2 p-2">
              {Array.from({ length: 4 }).map((_, i) => (
                <Skeleton key={i} className="h-8 w-full" />
              ))}
            </div>
          ) : files && files.length > 0 ? (
            <div className="space-y-0.5">
              {files.map((f) => {
                const isActive =
                  selected?.ticker === f.ticker &&
                  selected?.interval === f.interval;
                return (
                  <button
                    key={`${f.ticker}-${f.interval}`}
                    onClick={() =>
                      setSelected({
                        ticker: f.ticker,
                        interval: f.interval,
                      })
                    }
                    className={`flex w-full items-center gap-2 rounded px-3 py-2 text-left text-xs transition ${
                      isActive
                        ? "bg-blue-900/40 text-blue-300"
                        : "text-gray-400 hover:bg-gray-800/50 hover:text-gray-200"
                    }`}
                  >
                    <span className="font-medium">{f.ticker}</span>
                    <Badge
                      variant="secondary"
                      className="ml-auto h-4 px-1.5 text-[10px]"
                    >
                      {f.interval}
                    </Badge>
                  </button>
                );
              })}
            </div>
          ) : (
            <p className="px-2 py-8 text-center text-xs text-gray-600">
              No basis files yet.
            </p>
          )}
        </ScrollArea>
      </div>

      {/* Main content */}
      <div className="flex-1 overflow-y-auto p-6">
        {!selected ? (
          <div className="flex h-full items-center justify-center">
            <p className="text-sm text-gray-500">
              Select a basis file from the sidebar to preview
            </p>
          </div>
        ) : previewLoading ? (
          <div className="flex h-60 items-center justify-center text-sm text-gray-500">
            Loading preview...
          </div>
        ) : !preview || preview.error ? (
          <div className="flex h-40 items-center justify-center text-sm text-red-400">
            {preview?.error || "Failed to load preview"}
          </div>
        ) : (
          <div className="space-y-4">
            {/* Header */}
            <div className="flex items-center gap-3">
              <h2 className="text-sm font-semibold text-gray-200">
                <span className="text-blue-400">{preview.ticker}</span> ·{" "}
                {preview.interval}
              </h2>
              <Badge className="bg-blue-900/50 text-blue-300 hover:bg-blue-900/50">
                {preview.stats.bars.toLocaleString()} bars
              </Badge>
              <span className="text-xs text-gray-500">
                {preview.stats.start} → {preview.stats.end}
              </span>
            </div>

            {/* Stats row */}
            <div className="grid grid-cols-4 gap-3">
              <div className="rounded border border-gray-800 bg-gray-900 p-3">
                <p className="text-xs text-gray-500">Quality</p>
                <p
                  className={`text-lg font-semibold ${
                    preview.stats.quality_pct >= 99
                      ? "text-green-400"
                      : preview.stats.quality_pct >= 90
                        ? "text-yellow-400"
                        : "text-red-400"
                  }`}
                >
                  {preview.stats.quality_pct.toFixed(1)}%
                </p>
              </div>
              {preview.quote_venues.map((venue) => {
                const vs = preview.venue_stats[venue] as Record<
                  string,
                  unknown
                >;
                return (
                  <div
                    key={venue}
                    className="rounded border border-gray-800 bg-gray-900 p-3"
                  >
                    <p className="text-xs text-gray-500">{venue} basis</p>
                    <p className="text-lg font-semibold text-amber-400">
                      {typeof vs?.mean === "number"
                        ? `${vs.mean.toFixed(1)} bps`
                        : "—"}
                    </p>
                    {typeof vs?.std === "number" && (
                      <p className="text-[10px] text-gray-500">
                        σ {vs.std.toFixed(1)} bps
                      </p>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Chart */}
            <BasisChart
              chartData={preview.chart_data}
              quoteVenues={preview.quote_venues}
            />
          </div>
        )}
      </div>
    </div>
  );
}
