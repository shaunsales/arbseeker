import { useState, useEffect, useRef } from "react";
import { downloadData, getDownloadStatus } from "@/api/data";
import { Button } from "@/components/ui/button";
import { Download } from "lucide-react";

interface Props {
  intervals: string[];
  onComplete: () => void;
}

export default function DataDownload({ intervals, onComplete }: Props) {
  const [venue] = useState("binance");
  const [market, setMarket] = useState("futures");
  const [ticker, setTicker] = useState("");
  const [interval, setInterval] = useState("1h");
  const [startMonth, setStartMonth] = useState("");
  const [endMonth, setEndMonth] = useState("");
  const [downloading, setDownloading] = useState(false);
  const [progress, setProgress] = useState<{ pct: number; msg: string } | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  async function handleDownload() {
    if (!ticker || !startMonth || !endMonth) return;
    setDownloading(true);
    setProgress({ pct: 0, msg: "Starting..." });

    try {
      const res = await downloadData({
        venue,
        market,
        ticker: ticker.toUpperCase(),
        interval,
        start_month: startMonth,
        end_month: endMonth,
      });

      if (res.status === "already_running") {
        setProgress({ pct: 50, msg: "Download already in progress..." });
      }

      // Poll for status
      pollRef.current = setInterval(async () => {
        try {
          const status = await getDownloadStatus(res.job_id);
          setProgress({ pct: status.progress, msg: status.message });

          if (status.status === "complete" || status.status === "error") {
            if (pollRef.current) clearInterval(pollRef.current);
            pollRef.current = null;
            setDownloading(false);
            if (status.status === "complete") {
              setTimeout(() => {
                setProgress(null);
                onComplete();
              }, 1500);
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

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-4 space-y-3">
      <h3 className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-gray-400">
        <Download className="h-3.5 w-3.5" />
        Download Data
      </h3>

      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="mb-1 block text-xs text-gray-500">Market</label>
          <select
            value={market}
            onChange={(e) => setMarket(e.target.value)}
            className="w-full rounded border border-gray-700 bg-gray-800 px-2.5 py-1.5 text-xs text-gray-200"
          >
            <option value="futures">Futures</option>
            <option value="spot">Spot</option>
          </select>
        </div>
        <div>
          <label className="mb-1 block text-xs text-gray-500">Interval</label>
          <select
            value={interval}
            onChange={(e) => setInterval(e.target.value)}
            className="w-full rounded border border-gray-700 bg-gray-800 px-2.5 py-1.5 text-xs text-gray-200"
          >
            {intervals.map((iv) => (
              <option key={iv} value={iv}>
                {iv}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div>
        <label className="mb-1 block text-xs text-gray-500">Ticker</label>
        <input
          type="text"
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          placeholder="e.g. BTCUSDT"
          className="w-full rounded border border-gray-700 bg-gray-800 px-2.5 py-1.5 text-xs text-gray-200 placeholder:text-gray-600 focus:border-blue-500 focus:outline-none"
        />
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="mb-1 block text-xs text-gray-500">Start Month</label>
          <input
            type="month"
            value={startMonth}
            onChange={(e) => setStartMonth(e.target.value)}
            className="w-full rounded border border-gray-700 bg-gray-800 px-2.5 py-1.5 text-xs text-gray-200 focus:border-blue-500 focus:outline-none"
          />
        </div>
        <div>
          <label className="mb-1 block text-xs text-gray-500">End Month</label>
          <input
            type="month"
            value={endMonth}
            onChange={(e) => setEndMonth(e.target.value)}
            className="w-full rounded border border-gray-700 bg-gray-800 px-2.5 py-1.5 text-xs text-gray-200 focus:border-blue-500 focus:outline-none"
          />
        </div>
      </div>

      <Button
        onClick={handleDownload}
        disabled={downloading || !ticker || !startMonth || !endMonth}
        className="w-full"
        size="sm"
      >
        {downloading ? "Downloading..." : "Download"}
      </Button>

      {progress && (
        <div className="space-y-1">
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-gray-800">
            <div
              className={`h-full rounded-full transition-all ${
                progress.pct >= 100 ? "bg-green-500" : "bg-blue-500"
              }`}
              style={{ width: `${Math.max(progress.pct, 2)}%` }}
            />
          </div>
          <p className="text-[10px] text-gray-500">{progress.msg}</p>
        </div>
      )}
    </div>
  );
}
