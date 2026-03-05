import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getDataPreview } from "@/api/data";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import OhlcvChart from "./OhlcvChart";

interface Props {
  venue: string;
  market: string;
  ticker: string;
  interval: string;
}

export default function DataPreview({ venue, market, ticker, interval }: Props) {
  const [tab, setTab] = useState<"chart" | "table">("chart");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(100);

  const { data, isLoading } = useQuery({
    queryKey: ["data-preview", venue, market, ticker, interval, page, pageSize],
    queryFn: () => getDataPreview(venue, market, ticker, interval, page, pageSize),
  });

  if (isLoading) {
    return (
      <div className="flex h-60 items-center justify-center text-sm text-gray-500">
        Loading preview...
      </div>
    );
  }

  if (!data || data.error) {
    return (
      <div className="flex h-40 items-center justify-center text-sm text-red-400">
        {data?.error || "Failed to load preview"}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center gap-3">
        <h2 className="text-sm font-semibold text-gray-200">
          <span className="text-blue-400">{ticker}</span> · {interval}
        </h2>
        <Badge className="bg-blue-900/50 text-blue-300 hover:bg-blue-900/50">
          {data.pagination.total_rows.toLocaleString()} bars
        </Badge>
        <span className="text-xs text-gray-500">
          {data.date_range.min} → {data.date_range.max}
        </span>
        <Badge variant="secondary" className="text-[10px]">
          {data.available_periods.length} months
        </Badge>
      </div>

      {/* Summary stats */}
      {data.summary && (
        <div className="flex gap-4 text-xs text-gray-400">
          {Object.entries(data.summary).map(([key, val]) => (
            <span key={key}>
              <span className="text-gray-500">{key}:</span>{" "}
              {typeof val === "number" ? val.toLocaleString() : String(val)}
            </span>
          ))}
        </div>
      )}

      {/* Tabs */}
      <div className="flex border-b border-gray-800">
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

      {/* Chart */}
      {tab === "chart" && <OhlcvChart chartData={data.chart_data} />}

      {/* Table */}
      {tab === "table" && (
        <div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead className="bg-gray-800/50">
                <tr>
                  {["timestamp", "open", "high", "low", "close", "volume"].map(
                    (col) => (
                      <th
                        key={col}
                        className={`px-2 py-1.5 font-medium text-gray-400 ${
                          col === "timestamp" ? "text-left" : "text-right"
                        }`}
                      >
                        {col}
                      </th>
                    )
                  )}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800">
                {data.table_data.map((row, i) => (
                  <tr key={i} className="hover:bg-gray-800/50">
                    <td className="px-2 py-1 font-mono text-gray-400">
                      {row.timestamp}
                    </td>
                    <td className="px-2 py-1 text-right font-mono text-gray-300">
                      {row.open}
                    </td>
                    <td className="px-2 py-1 text-right font-mono text-green-400">
                      {row.high}
                    </td>
                    <td className="px-2 py-1 text-right font-mono text-red-400">
                      {row.low}
                    </td>
                    <td className="px-2 py-1 text-right font-mono text-gray-200">
                      {row.close}
                    </td>
                    <td className="px-2 py-1 text-right font-mono text-gray-500">
                      {row.volume}
                    </td>
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
    </div>
  );
}
