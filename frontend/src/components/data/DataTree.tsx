import { useState } from "react";
import type { DataTree } from "@/types/api";
import { ChevronRight, ChevronDown, Database, FolderOpen } from "lucide-react";
import { Badge } from "@/components/ui/badge";

interface Props {
  tree: DataTree;
  onSelect: (venue: string, market: string, ticker: string, interval: string) => void;
  selected?: { venue: string; market: string; ticker: string; interval: string } | null;
}

export default function DataTree({ tree, onSelect, selected }: Props) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  function toggle(key: string) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  const isSelected = (v: string, m: string, t: string, i: string) =>
    selected?.venue === v &&
    selected?.market === m &&
    selected?.ticker === t &&
    selected?.interval === i;

  return (
    <div className="space-y-0.5 text-sm">
      {Object.entries(tree).sort(([a], [b]) => a.localeCompare(b)).map(([venue, markets]) => {
        const venueKey = venue;
        return (
          <div key={venueKey}>
            <button
              onClick={() => toggle(venueKey)}
              className="flex w-full items-center gap-1.5 rounded px-2 py-1 text-left text-gray-400 hover:bg-gray-800/50 hover:text-gray-200"
            >
              {expanded.has(venueKey) ? (
                <ChevronDown className="h-3.5 w-3.5" />
              ) : (
                <ChevronRight className="h-3.5 w-3.5" />
              )}
              <FolderOpen className="h-3.5 w-3.5 text-blue-400" />
              <span className="font-medium">{venue}</span>
            </button>

            {expanded.has(venueKey) &&
              Object.entries(markets).sort(([a], [b]) => a.localeCompare(b)).map(([market, tickers]) => {
                const marketKey = `${venue}/${market}`;
                return (
                  <div key={marketKey} className="ml-4">
                    <button
                      onClick={() => toggle(marketKey)}
                      className="flex w-full items-center gap-1.5 rounded px-2 py-1 text-left text-gray-400 hover:bg-gray-800/50 hover:text-gray-200"
                    >
                      {expanded.has(marketKey) ? (
                        <ChevronDown className="h-3.5 w-3.5" />
                      ) : (
                        <ChevronRight className="h-3.5 w-3.5" />
                      )}
                      <span className="text-xs">{market}</span>
                    </button>

                    {expanded.has(marketKey) &&
                      Object.entries(tickers).sort(([a], [b]) => a.localeCompare(b)).map(([ticker, intervals]) => {
                        const tickerKey = `${venue}/${market}/${ticker}`;
                        return (
                          <div key={tickerKey} className="ml-4">
                            <button
                              onClick={() => toggle(tickerKey)}
                              className="flex w-full items-center gap-1.5 rounded px-2 py-1 text-left text-gray-300 hover:bg-gray-800/50 hover:text-gray-100"
                            >
                              {expanded.has(tickerKey) ? (
                                <ChevronDown className="h-3.5 w-3.5" />
                              ) : (
                                <ChevronRight className="h-3.5 w-3.5" />
                              )}
                              <span className="font-medium text-xs">
                                {ticker}
                              </span>
                              <span className="ml-auto text-xs text-gray-600">
                                {Object.keys(intervals).length} intervals
                              </span>
                            </button>

                            {expanded.has(tickerKey) && (
                              <div className="ml-6 space-y-0.5 py-0.5">
                                {Object.entries(intervals).sort(([a], [b]) => a.localeCompare(b)).map(
                                  ([interval, periods]) => (
                                    <button
                                      key={interval}
                                      onClick={() =>
                                        onSelect(venue, market, ticker, interval)
                                      }
                                      className={`flex w-full items-center gap-2 rounded px-2 py-1 text-left text-xs transition ${
                                        isSelected(venue, market, ticker, interval)
                                          ? "bg-blue-900/40 text-blue-300"
                                          : "text-gray-400 hover:bg-gray-800/50 hover:text-gray-200"
                                      }`}
                                    >
                                      <Database className="h-3 w-3" />
                                      <span className="font-mono">
                                        {interval}
                                      </span>
                                      <Badge
                                        variant="secondary"
                                        className="ml-auto h-4 px-1.5 text-[10px]"
                                      >
                                        {periods.length} mo
                                      </Badge>
                                    </button>
                                  )
                                )}
                              </div>
                            )}
                          </div>
                        );
                      })}
                  </div>
                );
              })}
          </div>
        );
      })}

      {Object.keys(tree).length === 0 && (
        <p className="px-2 py-4 text-center text-xs text-gray-600">
          No data downloaded yet. Use the download panel to get started.
        </p>
      )}
    </div>
  );
}
