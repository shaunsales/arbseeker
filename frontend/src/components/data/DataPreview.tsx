import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
} from "@tanstack/react-table";
import { getDataPreview } from "@/api/data";
import type { OhlcvTableRow } from "@/types/api";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ArrowUpDown, CheckCircle2, AlertTriangle } from "lucide-react";
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
        Loading preview…
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

  const r = data.report;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center gap-3">
        <h2 className="text-sm font-semibold text-gray-200">
          <span className="text-blue-400">{ticker}</span>{" "}
          <span className="text-gray-500">·</span> {interval}
        </h2>
        <Badge className="bg-blue-900/50 text-blue-300 hover:bg-blue-900/50">
          {data.pagination.total_rows.toLocaleString()} bars
        </Badge>
        <span className="text-xs text-gray-500">
          {data.date_range.min} → {data.date_range.max}
        </span>
      </div>

      {/* Quality stats */}
      <div className="flex flex-wrap gap-4 text-xs">
        <span className="flex items-center gap-1">
          {r.is_valid ? (
            <CheckCircle2 className="h-3.5 w-3.5 text-green-400" />
          ) : (
            <AlertTriangle className="h-3.5 w-3.5 text-yellow-400" />
          )}
          <span className="text-gray-400">Coverage</span>{" "}
          <span className={r.coverage_pct >= 99 ? "text-green-400" : "text-yellow-400"}>
            {r.coverage_pct}%
          </span>
        </span>
        <span className="text-gray-500">
          {r.total_bars.toLocaleString()} / {r.expected_bars.toLocaleString()} expected
        </span>
        {r.gap_count > 0 && (
          <span className="text-yellow-400">
            {r.gap_count} gaps ({r.total_missing_bars} bars missing)
          </span>
        )}
        {r.null_count > 0 && <span className="text-red-400">{r.null_count} nulls</span>}
        {r.zero_volume_count > 0 && (
          <span className="text-gray-500">{r.zero_volume_count} zero-vol bars</span>
        )}
      </div>

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
            {t === "chart" ? "Chart" : "Table"}
          </button>
        ))}
      </div>

      {/* Chart */}
      {tab === "chart" && (
        <div className="space-y-2">
          {data.chart_data.resampled && (
            <div className="flex items-center gap-2 rounded border border-amber-900/50 bg-amber-950/30 px-3 py-1.5 text-xs text-amber-300">
              <span>
                Showing <span className="font-semibold">{data.chart_data.chart_interval}</span> aggregation
                of {data.chart_data.original_bars.toLocaleString()} bars
                ({data.chart_data.ohlcv.length.toLocaleString()} chart bars)
              </span>
            </div>
          )}
          <OhlcvChart chartData={data.chart_data} />
        </div>
      )}

      {/* Table */}
      {tab === "table" && (
        <OhlcvTable
          rows={data.table_data}
          pagination={data.pagination}
          page={page}
          pageSize={pageSize}
          onPageChange={setPage}
          onPageSizeChange={(s) => { setPageSize(s); setPage(1); }}
        />
      )}
    </div>
  );
}


// ── OHLCV Table with TanStack Table ──

function fmtPrice(v: number) {
  return v >= 1000 ? v.toFixed(2) : v >= 1 ? v.toFixed(4) : v.toFixed(6);
}

function fmtVolume(v: number) {
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(2)}M`;
  if (v >= 1_000) return `${(v / 1_000).toFixed(1)}K`;
  return v.toFixed(1);
}

function fmtTimestamp(iso: string) {
  return iso.replace("T", " ").replace("+00:00", "").slice(0, 16);
}

const columns: ColumnDef<OhlcvTableRow>[] = [
  {
    accessorKey: "timestamp",
    header: "Time",
    cell: ({ getValue }) => (
      <span className="text-gray-400">{fmtTimestamp(getValue<string>())}</span>
    ),
    enableSorting: false,
  },
  {
    accessorKey: "open",
    header: "Open",
    cell: ({ getValue }) => fmtPrice(getValue<number>()),
    meta: { align: "right" },
  },
  {
    accessorKey: "high",
    header: "High",
    cell: ({ getValue }) => (
      <span className="text-green-400">{fmtPrice(getValue<number>())}</span>
    ),
    meta: { align: "right" },
  },
  {
    accessorKey: "low",
    header: "Low",
    cell: ({ getValue }) => (
      <span className="text-red-400">{fmtPrice(getValue<number>())}</span>
    ),
    meta: { align: "right" },
  },
  {
    accessorKey: "close",
    header: "Close",
    cell: ({ getValue }) => (
      <span className="text-gray-100">{fmtPrice(getValue<number>())}</span>
    ),
    meta: { align: "right" },
  },
  {
    accessorKey: "volume",
    header: "Volume",
    cell: ({ getValue }) => (
      <span className="text-gray-500">{fmtVolume(getValue<number>())}</span>
    ),
    meta: { align: "right" },
  },
];

interface TableProps {
  rows: OhlcvTableRow[];
  pagination: { page: number; page_size: number; total_rows: number; total_pages: number };
  page: number;
  pageSize: number;
  onPageChange: (p: number) => void;
  onPageSizeChange: (s: number) => void;
}

function OhlcvTable({ rows, pagination, page, pageSize, onPageChange, onPageSizeChange }: TableProps) {
  const [sorting, setSorting] = useState<SortingState>([]);

  const table = useReactTable({
    data: rows,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  return (
    <div>
      <div className="overflow-x-auto rounded border border-gray-800">
        <table className="w-full text-xs">
          <thead className="bg-gray-800/60">
            {table.getHeaderGroups().map((hg) => (
              <tr key={hg.id}>
                {hg.headers.map((header) => {
                  const align = (header.column.columnDef.meta as { align?: string })?.align;
                  return (
                    <th
                      key={header.id}
                      onClick={header.column.getToggleSortingHandler()}
                      className={`cursor-pointer select-none px-3 py-2 font-medium text-gray-400 ${
                        align === "right" ? "text-right" : "text-left"
                      }`}
                    >
                      <span className="inline-flex items-center gap-1">
                        {flexRender(header.column.columnDef.header, header.getContext())}
                        {header.column.getCanSort() && (
                          <ArrowUpDown className="h-3 w-3 text-gray-600" />
                        )}
                      </span>
                    </th>
                  );
                })}
              </tr>
            ))}
          </thead>
          <tbody className="divide-y divide-gray-800/50">
            {table.getRowModel().rows.map((row) => (
              <tr key={row.id} className="font-mono hover:bg-gray-800/30">
                {row.getVisibleCells().map((cell) => {
                  const align = (cell.column.columnDef.meta as { align?: string })?.align;
                  return (
                    <td
                      key={cell.id}
                      className={`px-3 py-1.5 ${align === "right" ? "text-right" : ""}`}
                    >
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination controls */}
      <div className="mt-3 flex items-center justify-between text-xs">
        <span className="text-gray-500">
          {((page - 1) * pageSize + 1).toLocaleString()}–
          {Math.min(page * pageSize, pagination.total_rows).toLocaleString()} of{" "}
          {pagination.total_rows.toLocaleString()}
        </span>
        <div className="flex items-center gap-1.5">
          <select
            value={pageSize}
            onChange={(e) => onPageSizeChange(Number(e.target.value))}
            className="rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-gray-300"
          >
            {[50, 100, 500, 1000].map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
          <Button variant="secondary" size="sm" onClick={() => onPageChange(1)} disabled={page <= 1} className="h-6 px-2 text-xs">
            «
          </Button>
          <Button variant="secondary" size="sm" onClick={() => onPageChange(page - 1)} disabled={page <= 1} className="h-6 px-2 text-xs">
            ‹
          </Button>
          <span className="px-2 text-gray-400">
            {page} / {pagination.total_pages}
          </span>
          <Button variant="secondary" size="sm" onClick={() => onPageChange(page + 1)} disabled={page >= pagination.total_pages} className="h-6 px-2 text-xs">
            ›
          </Button>
          <Button variant="secondary" size="sm" onClick={() => onPageChange(pagination.total_pages)} disabled={page >= pagination.total_pages} className="h-6 px-2 text-xs">
            »
          </Button>
        </div>
      </div>
    </div>
  );
}
