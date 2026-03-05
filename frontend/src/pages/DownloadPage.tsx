import { useState, useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { getDataTree, downloadData, getDownloadStatus } from "@/api/data";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Download, ExternalLink } from "lucide-react";

const INTERVALS = [
  "1m", "3m", "5m", "15m", "30m",
  "1h", "2h", "4h", "6h", "8h", "12h",
  "1d",
];

export default function DownloadPage() {
  const [market, setMarket] = useState("futures");
  const [ticker, setTicker] = useState("");
  const [interval, setSelectedInterval] = useState("1h");
  const [startMonth, setStartMonth] = useState("");
  const [endMonth, setEndMonth] = useState("");
  const [downloading, setDownloading] = useState(false);
  const [progress, setProgress] = useState<{ pct: number; msg: string } | null>(null);
  const [completed, setCompleted] = useState<string[]>([]);
  const pollRef = useRef<ReturnType<typeof globalThis.setInterval> | null>(null);

  const { data: treeData } = useQuery({
    queryKey: ["data-tree"],
    queryFn: getDataTree,
  });

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  async function handleDownload() {
    if (!ticker || !startMonth || !endMonth) return;
    setDownloading(true);
    setProgress({ pct: 0, msg: "Starting download..." });

    try {
      const res = await downloadData({
        venue: "binance",
        market,
        ticker: ticker.toUpperCase(),
        interval,
        start_month: startMonth,
        end_month: endMonth,
      });

      if (res.status === "already_running") {
        setProgress({ pct: 50, msg: "Download already in progress..." });
      }

      pollRef.current = globalThis.setInterval(async () => {
        try {
          const status = await getDownloadStatus(res.job_id);
          setProgress({ pct: status.progress, msg: status.message });

          if (status.status === "complete" || status.status === "error") {
            if (pollRef.current) clearInterval(pollRef.current);
            pollRef.current = null;
            setDownloading(false);
            if (status.status === "complete") {
              setCompleted((prev) => [
                `${ticker.toUpperCase()} ${interval} (${startMonth} → ${endMonth})`,
                ...prev,
              ]);
              setTimeout(() => setProgress(null), 2000);
            }
          }
        } catch (_e) {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;
          setDownloading(false);
        }
      }, 1000);
    } catch (e) {
      setProgress({
        pct: 0,
        msg: e instanceof Error ? e.message : "Download failed",
      });
      setDownloading(false);
    }
  }

  // Count existing data
  const existingCount = treeData
    ? Object.values(treeData.data_tree.binance ?? {}).reduce(
        (acc, markets) => acc + Object.keys(markets).length,
        0,
      )
    : 0;

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="mx-auto max-w-2xl space-y-6">
        {/* Header */}
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-lg font-semibold text-gray-200">Download Data</h1>
            <Badge className="bg-amber-900/50 text-amber-300 hover:bg-amber-900/50">
              Binance
            </Badge>
          </div>
          <p className="mt-1 text-sm text-gray-500">
            Download historical OHLCV kline data from the Binance API.
            Data is saved locally as Parquet files organized by month.
          </p>
        </div>

        {/* Form card */}
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-6 space-y-5">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="mb-1.5 block text-xs font-medium text-gray-400">
                Market
              </label>
              <select
                value={market}
                onChange={(e) => setMarket(e.target.value)}
                className="w-full rounded-md border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-200 focus:border-blue-500 focus:outline-none"
              >
                <option value="futures">Futures (USD-M)</option>
                <option value="spot">Spot</option>
              </select>
            </div>
            <div>
              <label className="mb-1.5 block text-xs font-medium text-gray-400">
                Interval
              </label>
              <select
                value={interval}
                onChange={(e) => setSelectedInterval(e.target.value)}
                className="w-full rounded-md border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-200 focus:border-blue-500 focus:outline-none"
              >
                {INTERVALS.map((iv) => (
                  <option key={iv} value={iv}>{iv}</option>
                ))}
              </select>
            </div>
          </div>

          <div>
            <label className="mb-1.5 block text-xs font-medium text-gray-400">
              Ticker Symbol
            </label>
            <input
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              placeholder="e.g. BTCUSDT, ETHUSDT, SOLUSDT"
              className="w-full rounded-md border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-200 placeholder:text-gray-600 focus:border-blue-500 focus:outline-none"
            />
            <p className="mt-1 text-[11px] text-gray-600">
              Must match the exact Binance symbol name
            </p>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="mb-1.5 block text-xs font-medium text-gray-400">
                Start Month
              </label>
              <input
                type="month"
                value={startMonth}
                onChange={(e) => setStartMonth(e.target.value)}
                className="w-full rounded-md border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-200 focus:border-blue-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="mb-1.5 block text-xs font-medium text-gray-400">
                End Month
              </label>
              <input
                type="month"
                value={endMonth}
                onChange={(e) => setEndMonth(e.target.value)}
                className="w-full rounded-md border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-200 focus:border-blue-500 focus:outline-none"
              />
            </div>
          </div>

          <Button
            onClick={handleDownload}
            disabled={downloading || !ticker || !startMonth || !endMonth}
            className="w-full"
          >
            <Download className="mr-2 h-4 w-4" />
            {downloading ? "Downloading..." : "Download from Binance"}
          </Button>

          {progress && (
            <div className="space-y-1.5">
              <div className="h-2 w-full overflow-hidden rounded-full bg-gray-800">
                <div
                  className={`h-full rounded-full transition-all duration-300 ${
                    progress.pct >= 100 ? "bg-green-500" : "bg-blue-500"
                  }`}
                  style={{ width: `${Math.max(progress.pct, 2)}%` }}
                />
              </div>
              <p className="text-xs text-gray-500">{progress.msg}</p>
            </div>
          )}
        </div>

        {/* Completed downloads */}
        {completed.length > 0 && (
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <h3 className="mb-2 text-xs font-medium text-gray-400">
              Completed this session
            </h3>
            <div className="space-y-1">
              {completed.map((item, i) => (
                <div
                  key={i}
                  className="flex items-center gap-2 text-xs text-green-400"
                >
                  <span className="h-1.5 w-1.5 rounded-full bg-green-500" />
                  {item}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Info */}
        <div className="rounded-lg border border-gray-800 bg-gray-900/50 p-4 text-xs text-gray-500 space-y-2">
          <p>
            <strong className="text-gray-400">Data source:</strong> Binance REST API
            (klines endpoint). Data is downloaded month-by-month and saved as Parquet files.
          </p>
          <p>
            <strong className="text-gray-400">Storage:</strong> <code className="text-gray-400">data/</code> directory,
            organized as <code className="text-gray-400">venue/market/ticker/interval/YYYY-MM.parquet</code>
          </p>
          {existingCount > 0 && (
            <p>
              <strong className="text-gray-400">Currently stored:</strong>{" "}
              {existingCount} Binance ticker(s) in the data directory.
            </p>
          )}
          <a
            href="https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-blue-400 hover:underline"
          >
            Binance Kline API docs <ExternalLink className="h-3 w-3" />
          </a>
        </div>
      </div>
    </div>
  );
}
