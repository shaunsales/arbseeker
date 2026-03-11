import { useState, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { getPreview, computeIndicators } from "@/api/strategy";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { X } from "lucide-react";
import StrategyChart from "./StrategyChart";
import IndicatorPicker, { type ActiveIndicator } from "./IndicatorPicker";
import type { ComputeIndicatorsResponse } from "@/types/api";

interface Props {
  className: string;
  interval: string;
  onClose: () => void;
}

export default function DataPreview({ className, interval, onClose }: Props) {
  const [tab, setTab] = useState<"chart" | "table">("chart");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(100);
  const [activeIndicators, setActiveIndicators] = useState<ActiveIndicator[]>([]);
  const [adHocData, setAdHocData] = useState<ComputeIndicatorsResponse | null>(null);
  const [computing, setComputing] = useState(false);

  const { data, isLoading } = useQuery({
    queryKey: ["strategy-preview", className, interval, page, pageSize],
    queryFn: () => getPreview(className, interval, page, pageSize),
  });

  const handleAddIndicator = useCallback((ind: ActiveIndicator) => {
    setActiveIndicators((prev) => [...prev, ind]);
  }, []);

  const handleRemoveIndicator = useCallback((id: string) => {
    setActiveIndicators((prev) => prev.filter((i) => i.id !== id));
  }, []);

  const handleCompute = useCallback(async () => {
    if (activeIndicators.length === 0) return;
    setComputing(true);
    try {
      const result = await computeIndicators({
        class_name: className,
        interval,
        indicators: activeIndicators.map((i) => ({
          name: i.name,
          params: i.params,
        })),
      });
      setAdHocData(result);
    } catch (e) {
      console.error("Failed to compute indicators:", e);
    }
    setComputing(false);
  }, [activeIndicators, className, interval]);

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
      {/* Header */}
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-semibold text-gray-200">
            <span className="text-blue-400">{interval}</span>
            {data && <> · {data.strategy_name}</>}
          </h3>
          {data && (
            <>
              <Badge className="bg-blue-900/50 text-blue-300 hover:bg-blue-900/50">
                {data.pagination.total_rows.toLocaleString()} bars
              </Badge>
              {data.overlay_cols.length > 0 && (
                <Badge className="bg-purple-900/50 text-purple-300 hover:bg-purple-900/50">
                  {data.overlay_cols.length} overlay
                  {data.overlay_cols.length !== 1 ? "s" : ""}
                </Badge>
              )}
              {data.separate_cols.length > 0 && (
                <Badge className="bg-amber-900/50 text-amber-300 hover:bg-amber-900/50">
                  {data.separate_cols.length} indicator
                  {data.separate_cols.length !== 1 ? "s" : ""}
                </Badge>
              )}
            </>
          )}
        </div>
        <button
          onClick={onClose}
          className="text-gray-500 hover:text-gray-300 transition"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Tabs */}
      <div className="mb-3 flex border-b border-gray-800">
        {(["chart", "table"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`border-b-2 px-3 py-1.5 text-xs font-medium transition ${
              tab === t
                ? "border-blue-500 text-blue-400"
                : "border-transparent text-gray-500 hover:text-gray-300"
            }`}
          >
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {isLoading ? (
        <div className="flex h-40 items-center justify-center text-sm text-gray-500">
          Loading preview...
        </div>
      ) : !data ? (
        <div className="flex h-40 items-center justify-center text-sm text-red-400">
          Failed to load preview
        </div>
      ) : (
        <>
          {/* Chart */}
          {tab === "chart" && (
            <>
              <IndicatorPicker
                activeIndicators={activeIndicators}
                onAdd={handleAddIndicator}
                onRemove={handleRemoveIndicator}
                onCompute={handleCompute}
                computing={computing}
              />
              <StrategyChart
                chartData={data.chart_data}
                separateCols={data.separate_cols}
                adHocData={adHocData}
              />
            </>
          )}

          {/* Table */}
          {tab === "table" && (
            <div>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead className="bg-gray-800/50">
                    <tr>
                      <th className="px-2 py-1.5 text-left font-medium text-gray-400">
                        Timestamp
                      </th>
                      {data.all_cols.map((col) => (
                        <th
                          key={col}
                          className={`px-2 py-1.5 text-right font-medium ${
                            data.overlay_cols.includes(col)
                              ? "text-purple-400"
                              : data.separate_cols.includes(col)
                                ? "text-amber-400"
                                : "text-gray-400"
                          }`}
                        >
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800">
                    {data.table_data.map((row, i) => (
                      <tr key={i} className="hover:bg-gray-800/50">
                        <td className="px-2 py-1 font-mono text-gray-400">
                          {String(row.timestamp)}
                        </td>
                        {data.all_cols.map((col) => (
                          <td
                            key={col}
                            className={`px-2 py-1 text-right font-mono ${
                              data.overlay_cols.includes(col)
                                ? "text-purple-300"
                                : data.separate_cols.includes(col)
                                  ? "text-amber-300"
                                  : "text-gray-300"
                            }`}
                          >
                            {String(row[col] ?? "")}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              <div className="mt-3 flex items-center justify-between text-xs">
                <span className="text-gray-500">
                  {(data.pagination.page - 1) * data.pagination.page_size + 1}–
                  {Math.min(
                    data.pagination.page * data.pagination.page_size,
                    data.pagination.total_rows
                  )}{" "}
                  of {data.pagination.total_rows.toLocaleString()}
                </span>
                <div className="flex items-center gap-1.5">
                  <select
                    value={pageSize}
                    onChange={(e) => {
                      setPageSize(Number(e.target.value));
                      setPage(1);
                    }}
                    className="rounded border border-gray-700 bg-gray-800 px-1.5 py-0.5 text-xs text-gray-300"
                  >
                    <option value={100}>100</option>
                    <option value={500}>500</option>
                    <option value={1000}>1000</option>
                  </select>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => setPage(1)}
                    disabled={page <= 1}
                    className="h-6 px-1.5 text-xs"
                  >
                    «
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => setPage((p) => p - 1)}
                    disabled={page <= 1}
                    className="h-6 px-1.5 text-xs"
                  >
                    ‹
                  </Button>
                  <span className="px-2 text-gray-400">
                    {data.pagination.page}/{data.pagination.total_pages}
                  </span>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => setPage((p) => p + 1)}
                    disabled={page >= data.pagination.total_pages}
                    className="h-6 px-1.5 text-xs"
                  >
                    ›
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => setPage(data.pagination.total_pages)}
                    disabled={page >= data.pagination.total_pages}
                    className="h-6 px-1.5 text-xs"
                  >
                    »
                  </Button>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
