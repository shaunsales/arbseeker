import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  connect,
  getExchanges,
  searchMultiExchangePairs,
  cbboWsUrl,
  type ConnectResult,
  type Exchange,
  type MultiPair,
  type CBBOMessage,
  type CBBOLevel,
} from "@/api/coinroutes";
import { Badge } from "@/components/ui/badge";
import {
  Plug, Loader2, Search, Building2, Radio, X, Check,
  ChevronDown, ChevronUp, RefreshCw,
} from "lucide-react";

// ── Types ──

interface CBBOState {
  data: CBBOMessage | null;
  status: "connecting" | "streaming" | "error";
  error: string | null;
  lastUpdateMs: number;
  updateCount: number;
}

// ── CBBO WebSocket hook ──

function useCBBO(currencyPair: string, delayMs = 0): CBBOState {
  const [data, setData] = useState<CBBOMessage | null>(null);
  const [status, setStatus] = useState<"connecting" | "streaming" | "error">("connecting");
  const [error, setError] = useState<string | null>(null);
  const [lastUpdateMs, setLastUpdateMs] = useState(0);
  const [updateCount, setUpdateCount] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);
  const retryRef = useRef(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let cancelled = false;

    function connectWs() {
      if (cancelled) return;
      setStatus("connecting");
      setError(null);

      const ws = new WebSocket(cbboWsUrl(currencyPair, 5));
      wsRef.current = ws;

      ws.onopen = () => {
        retryRef.current = 0;
        setStatus("streaming");
      };
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data) as CBBOMessage;
          if (msg.type === "error") {
            setError(msg.error_text || msg.error || "Unknown error");
            setStatus("error");
          } else {
            setData(msg);
            setLastUpdateMs(Date.now());
            setUpdateCount((c) => c + 1);
            setStatus("streaming");
          }
        } catch {
          // ignore non-JSON
        }
      };
      ws.onerror = () => {
        // handled by onclose
      };
      ws.onclose = () => {
        if (cancelled) return;
        const delay = Math.min(1000 * 2 ** retryRef.current, 15000);
        retryRef.current++;
        setError(`Connection lost — retrying in ${Math.round(delay / 1000)}s…`);
        setStatus("error");
        timerRef.current = setTimeout(connectWs, delay);
      };
    }

    const initialTimer = setTimeout(connectWs, delayMs);

    return () => {
      cancelled = true;
      clearTimeout(initialTimer);
      if (timerRef.current) clearTimeout(timerRef.current);
      if (wsRef.current) wsRef.current.close();
      wsRef.current = null;
    };
  }, [currencyPair, delayMs]);

  return { data, status, error, lastUpdateMs, updateCount };
}

// ── Refresh pulse: shows age since last update ──

function UpdatePulse({ lastUpdateMs, updateCount }: { lastUpdateMs: number; updateCount: number }) {
  const [age, setAge] = useState(0);

  useEffect(() => {
    if (!lastUpdateMs) return;
    setAge(0);
    const iv = setInterval(() => setAge(Date.now() - lastUpdateMs), 200);
    return () => clearInterval(iv);
  }, [lastUpdateMs]);

  if (!lastUpdateMs) return null;

  const secs = Math.floor(age / 1000);
  const fresh = age < 1500;

  return (
    <span className="inline-flex items-center gap-1 text-[10px]">
      <RefreshCw
        key={updateCount}
        className={`h-3 w-3 ${fresh ? "text-green-400 animate-spin" : "text-gray-600"}`}
        style={{ animationDuration: "1s", animationIterationCount: fresh ? 1 : 0 }}
      />
      <span className={fresh ? "text-green-400" : "text-gray-600"}>
        {secs < 1 ? "now" : `${secs}s`}
      </span>
    </span>
  );
}

// ── Page ──

const EXCHANGE_TYPE_LABELS: Record<string, string> = {
  ALL: "All",
  CEX: "CEX",
  DEX: "DEX",
  MM: "Market Makers",
  OTC: "OTC",
  SIM: "Sim",
};

export default function CoinRoutesPage() {
  const [connected, setConnected] = useState(false);
  const [exchangeTypeFilter, setExchangeTypeFilter] = useState("ALL");
  const [selectedExchanges, setSelectedExchanges] = useState<Set<string>>(new Set());
  const [tokenSearch, setTokenSearch] = useState("");
  const [submittedSearch, setSubmittedSearch] = useState("");
  const [streamingPairs, setStreamingPairs] = useState<Set<string>>(new Set());
  // Aggregated CBBO data for summary panel
  const [cbboMap, setCbboMap] = useState<Map<string, CBBOMessage>>(new Map());

  const connectMut = useMutation({
    mutationFn: connect,
    onSuccess: (data) => setConnected(data.connected),
  });

  const exchangesQuery = useQuery({
    queryKey: ["cr-exchanges"],
    queryFn: getExchanges,
    enabled: connected,
  });

  const pairSearchQuery = useQuery({
    queryKey: ["cr-multi-pairs", [...selectedExchanges].sort().join(","), submittedSearch],
    queryFn: () => searchMultiExchangePairs([...selectedExchanges], submittedSearch),
    enabled: selectedExchanges.size > 0 && submittedSearch.length > 0,
  });

  const result = connectMut.data as ConnectResult | undefined;

  const filteredExchanges = useMemo(() => {
    const all = exchangesQuery.data ?? [];
    const sorted = [...all].sort((a, b) => a.name.localeCompare(b.name));
    if (exchangeTypeFilter === "ALL") return sorted;
    return sorted.filter((ex) => ex.type === exchangeTypeFilter);
  }, [exchangesQuery.data, exchangeTypeFilter]);

  const toggleExchange = useCallback((slug: string) => {
    setSelectedExchanges((prev) => {
      const next = new Set(prev);
      if (next.has(slug)) next.delete(slug);
      else next.add(slug);
      return next;
    });
  }, []);

  const togglePair = useCallback((slug: string) => {
    setStreamingPairs((prev) => {
      const next = new Set(prev);
      if (next.has(slug)) {
        next.delete(slug);
        setCbboMap((m) => { const n = new Map(m); n.delete(slug); return n; });
      } else {
        next.add(slug);
      }
      return next;
    });
  }, []);

  const handleCbboData = useCallback((pair: string, msg: CBBOMessage) => {
    setCbboMap((prev) => new Map(prev).set(pair, msg));
  }, []);

  const handleSearch = () => {
    if (tokenSearch.trim()) setSubmittedSearch(tokenSearch.trim().toUpperCase());
  };

  return (
    <div className="flex flex-col gap-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold text-gray-100">CoinRoutes</h1>
          <p className="text-sm text-gray-500">
            Smart Order Router — connect, explore venues, and trade
          </p>
        </div>
        <div className="flex items-center gap-3">
          {result && (
            <Badge variant={connected ? "default" : "destructive"}>
              {connected ? `Connected · ${result.url}` : "Disconnected"}
            </Badge>
          )}
          <button
            onClick={() => connectMut.mutate()}
            disabled={connectMut.isPending}
            className="inline-flex items-center gap-2 rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50"
          >
            {connectMut.isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Plug className="h-4 w-4" />
            )}
            {connected ? "Reconnect" : "Connect"}
          </button>
        </div>
      </div>

      {result && !connected && (
        <div className="rounded-md border border-red-800 bg-red-950/50 p-3 text-sm text-red-300">
          {result.error}
        </div>
      )}

      {connected && (
        <>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            {/* Column 1: Exchange multi-select */}
            <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
              <div className="flex items-center gap-2 text-sm font-medium text-gray-300">
                <Building2 className="h-4 w-4 text-gray-500" />
                Exchanges
                {selectedExchanges.size > 0 && (
                  <Badge variant="outline" className="text-[10px]">
                    {selectedExchanges.size} selected
                  </Badge>
                )}
              </div>

              <div className="mt-2 flex flex-wrap gap-1">
                {Object.entries(EXCHANGE_TYPE_LABELS).map(([key, label]) => (
                  <button
                    key={key}
                    onClick={() => setExchangeTypeFilter(key)}
                    className={`rounded px-2 py-0.5 text-[10px] font-medium transition-colors ${
                      exchangeTypeFilter === key
                        ? "bg-blue-600/20 text-blue-400"
                        : "text-gray-500 hover:bg-gray-800 hover:text-gray-300"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>

              {exchangesQuery.isLoading && (
                <div className="mt-3 flex items-center gap-2 text-xs text-gray-500">
                  <Loader2 className="h-3 w-3 animate-spin" /> Loading…
                </div>
              )}
              {exchangesQuery.data && (
                <div className="mt-2 max-h-[40vh] space-y-0.5 overflow-auto">
                  {filteredExchanges.map((ex: Exchange) => {
                    const selected = selectedExchanges.has(ex.slug);
                    return (
                      <button
                        key={ex.slug}
                        onClick={() => toggleExchange(ex.slug)}
                        className={`flex w-full items-center gap-2 rounded px-2.5 py-1.5 text-left text-xs transition-colors ${
                          selected
                            ? "bg-blue-600/20 text-blue-400"
                            : "text-gray-400 hover:bg-gray-800 hover:text-gray-200"
                        }`}
                      >
                        <span className={`flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded border ${
                          selected ? "border-blue-500 bg-blue-600" : "border-gray-600"
                        }`}>
                          {selected && <Check className="h-2.5 w-2.5 text-white" />}
                        </span>
                        <span className="flex-1">{ex.name}</span>
                        <span className="text-[9px] text-gray-600">{ex.type}</span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Column 2–3: Token search + pair results */}
            <div className="lg:col-span-2 rounded-lg border border-gray-800 bg-gray-900 p-4">
              <p className="text-sm font-medium text-gray-300">Search Trading Pairs</p>
              {selectedExchanges.size === 0 ? (
                <p className="mt-2 text-xs text-gray-600">Select one or more exchanges to search pairs</p>
              ) : (
                <>
                  <div className="mt-2 flex gap-2">
                    <input
                      value={tokenSearch}
                      onChange={(e) => setTokenSearch(e.target.value.toUpperCase())}
                      onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                      placeholder="Base token e.g. BTC, ETH, SOL"
                      className="w-full rounded border border-gray-700 bg-gray-950 px-3 py-1.5 text-sm text-gray-200 placeholder-gray-600 outline-none focus:border-blue-500"
                    />
                    <button
                      onClick={handleSearch}
                      disabled={pairSearchQuery.isLoading || !tokenSearch.trim()}
                      className="inline-flex items-center gap-1.5 rounded bg-gray-800 px-3 py-1.5 text-xs font-medium text-gray-300 hover:bg-gray-700 disabled:opacity-50"
                    >
                      {pairSearchQuery.isLoading ? (
                        <Loader2 className="h-3 w-3 animate-spin" />
                      ) : (
                        <Search className="h-3 w-3" />
                      )}
                      Search
                    </button>
                  </div>

                  <div className="mt-2 flex flex-wrap gap-1">
                    {[...selectedExchanges].sort().map((slug) => (
                      <span
                        key={slug}
                        className="inline-flex items-center gap-1 rounded-full bg-blue-600/10 px-2 py-0.5 text-[10px] text-blue-400"
                      >
                        {slug}
                        <button onClick={() => toggleExchange(slug)} className="hover:text-blue-200">
                          <X className="h-2.5 w-2.5" />
                        </button>
                      </span>
                    ))}
                  </div>

                  {pairSearchQuery.error && (
                    <p className="mt-2 text-xs text-red-400">
                      {pairSearchQuery.error instanceof Error ? pairSearchQuery.error.message : "Search failed"}
                    </p>
                  )}

                  {pairSearchQuery.data && (
                    <div className="mt-3">
                      <p className="text-[10px] text-gray-500 mb-2">
                        {pairSearchQuery.data.count} pairs found — click to stream CBBO
                      </p>
                      <div className="flex max-h-[40vh] flex-wrap gap-1.5 overflow-auto">
                        {pairSearchQuery.data.pairs.map((p: MultiPair) => {
                          const active = streamingPairs.has(p.slug);
                          return (
                            <button
                              key={p.slug}
                              onClick={() => togglePair(p.slug)}
                              className={`inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs font-medium transition-colors ${
                                active
                                  ? "border-green-600 bg-green-600/10 text-green-400"
                                  : "border-gray-700 text-gray-400 hover:border-gray-600 hover:text-gray-200"
                              }`}
                            >
                              {active && <Radio className="h-3 w-3 animate-pulse" />}
                              <span>{p.slug}</span>
                              <span className="text-gray-600">{p.product_type}</span>
                              <span className="text-[9px] text-gray-600">
                                {p.exchanges.join(", ")}
                              </span>
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>

          {/* Live summary panel */}
          {cbboMap.size >= 2 && (
            <SummaryPanel cbboMap={cbboMap} />
          )}

          {/* CBBO panels */}
          {streamingPairs.size > 0 && (
            <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
              {[...streamingPairs].sort().map((pair, idx) => (
                <CBBOPanel
                  key={pair}
                  pair={pair}
                  delayMs={idx * 1500}
                  selectedExchanges={selectedExchanges}
                  onData={handleCbboData}
                  onClose={() => togglePair(pair)}
                />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ── Summary: live pricing comparison + basis ──

function SummaryPanel({ cbboMap }: { cbboMap: Map<string, CBBOMessage> }) {
  const rows = useMemo(() => {
    const entries: { pair: string; bestBid: number; bestAsk: number; mid: number }[] = [];
    for (const [pair, msg] of cbboMap) {
      const bestBid = msg.bids.length > 0 ? parseFloat(msg.bids[0].price) : 0;
      const bestAsk = msg.asks.length > 0 ? parseFloat(msg.asks[0].price) : 0;
      const mid = bestBid && bestAsk ? (bestBid + bestAsk) / 2 : bestBid || bestAsk;
      entries.push({ pair, bestBid, bestAsk, mid });
    }
    entries.sort((a, b) => a.pair.localeCompare(b.pair));
    return entries;
  }, [cbboMap]);

  // Use the first pair as the reference for basis
  const refMid = rows.length > 0 ? rows[0].mid : 0;

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
      <p className="text-sm font-medium text-gray-300 mb-3">
        Live Summary &amp; Basis
      </p>
      <div className="overflow-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-gray-800 text-left text-gray-500">
              <th className="pb-1.5 font-medium">Pair</th>
              <th className="pb-1.5 text-right font-medium">Best Bid</th>
              <th className="pb-1.5 text-right font-medium">Best Ask</th>
              <th className="pb-1.5 text-right font-medium">Spread</th>
              <th className="pb-1.5 text-right font-medium">Mid</th>
              <th className="pb-1.5 text-right font-medium">
                Basis vs {rows[0]?.pair ?? "—"}
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => {
              const spread = r.bestAsk && r.bestBid ? r.bestAsk - r.bestBid : 0;
              const spreadPct = r.mid ? (spread / r.mid) * 100 : 0;
              const basisAbs = r.mid - refMid;
              const basisPct = refMid ? (basisAbs / refMid) * 100 : 0;
              const isRef = i === 0;

              return (
                <tr key={r.pair} className="border-b border-gray-800/30">
                  <td className="py-1.5 font-medium text-gray-200">{r.pair}</td>
                  <td className="py-1.5 text-right font-mono text-green-400">
                    {r.bestBid ? r.bestBid.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : "—"}
                  </td>
                  <td className="py-1.5 text-right font-mono text-red-400">
                    {r.bestAsk ? r.bestAsk.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : "—"}
                  </td>
                  <td className="py-1.5 text-right font-mono text-gray-400">
                    {spread ? `${spread.toFixed(2)} (${spreadPct.toFixed(3)}%)` : "—"}
                  </td>
                  <td className="py-1.5 text-right font-mono text-gray-300">
                    {r.mid ? r.mid.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : "—"}
                  </td>
                  <td className={`py-1.5 text-right font-mono ${
                    isRef ? "text-gray-600" : basisAbs > 0 ? "text-green-400" : basisAbs < 0 ? "text-red-400" : "text-gray-400"
                  }`}>
                    {isRef ? "ref" : `${basisAbs >= 0 ? "+" : ""}${basisAbs.toFixed(2)} (${basisPct >= 0 ? "+" : ""}${basisPct.toFixed(3)}%)`}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Self-contained CBBO Panel ──

function CBBOPanel({
  pair,
  delayMs = 0,
  selectedExchanges,
  onData,
  onClose,
}: {
  pair: string;
  delayMs?: number;
  selectedExchanges: Set<string>;
  onData: (pair: string, data: CBBOMessage) => void;
  onClose: () => void;
}) {
  const cbbo = useCBBO(pair, delayMs);
  const onDataRef = useRef(onData);
  onDataRef.current = onData;

  // Report data up whenever it changes
  useEffect(() => {
    if (cbbo.data) onDataRef.current(pair, cbbo.data);
  }, [pair, cbbo.data]);

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Radio className={`h-4 w-4 ${cbbo.status === "streaming" ? "text-green-400 animate-pulse" : "text-gray-600"}`} />
          <span className="text-sm font-semibold text-gray-100">{pair}</span>
          <Badge variant="outline" className="text-[10px]">
            {cbbo.status === "streaming" ? "LIVE" : cbbo.status}
          </Badge>
          <UpdatePulse lastUpdateMs={cbbo.lastUpdateMs} updateCount={cbbo.updateCount} />
        </div>
        <div className="flex items-center gap-2">
          {cbbo.data?.markets && (
            <span className="text-[10px] text-gray-600">
              {cbbo.data.markets.length} venues
            </span>
          )}
          <button onClick={onClose} className="text-gray-600 hover:text-gray-300">
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {cbbo.error && (
        <div className="mt-2 rounded border border-red-800 bg-red-950/50 p-2 text-xs text-red-300">
          {cbbo.error}
        </div>
      )}

      {cbbo.status === "connecting" && (
        <div className="mt-4 flex items-center gap-2 text-xs text-gray-500">
          <Loader2 className="h-3 w-3 animate-spin" /> Connecting…
        </div>
      )}

      {cbbo.data && (
        <div className="mt-3 grid grid-cols-2 gap-4">
          <OrderSide label="Bids" levels={cbbo.data.bids} color="green" selectedExchanges={selectedExchanges} />
          <OrderSide label="Asks" levels={cbbo.data.asks} color="red" selectedExchanges={selectedExchanges} />
        </div>
      )}
    </div>
  );
}

// ── Order side table with exchange filtering ──

function OrderSide({
  label,
  levels,
  color,
  selectedExchanges,
}: {
  label: string;
  levels: CBBOLevel[];
  color: "green" | "red";
  selectedExchanges: Set<string>;
}) {
  const [showAll, setShowAll] = useState(false);
  const textColor = color === "green" ? "text-green-400" : "text-red-400";
  const dimColor = color === "green" ? "text-green-400/40" : "text-red-400/40";
  const labelColor = color === "green" ? "text-green-500" : "text-red-500";

  const primary = levels.filter((l) => selectedExchanges.has(l.exchange));
  const others = levels.filter((l) => !selectedExchanges.has(l.exchange));
  const hasOthers = others.length > 0;

  return (
    <div>
      <p className={`mb-1.5 text-xs font-medium ${labelColor}`}>
        {label} ({primary.length}{hasOthers ? ` + ${others.length}` : ""})
      </p>
      <div className="max-h-[30vh] overflow-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-gray-800 text-left text-gray-500">
              <th className="pb-1 font-medium">Exchange</th>
              <th className="pb-1 text-right font-medium">Price</th>
              <th className="pb-1 text-right font-medium">Qty</th>
            </tr>
          </thead>
          <tbody>
            {primary.map((lvl, i) => (
              <tr key={`p-${lvl.exchange}-${i}`} className="border-b border-gray-800/30">
                <td className="py-1 text-gray-300 font-medium">{lvl.exchange}</td>
                <td className={`py-1 text-right font-mono ${textColor}`}>
                  {parseFloat(lvl.price).toLocaleString()}
                </td>
                <td className="py-1 text-right font-mono text-gray-400">
                  {parseFloat(lvl.qty)}
                </td>
              </tr>
            ))}

            {hasOthers && !showAll && (
              <tr>
                <td colSpan={3} className="py-1">
                  <button
                    onClick={() => setShowAll(true)}
                    className="flex w-full items-center justify-center gap-1 text-[10px] text-gray-500 hover:text-gray-300"
                  >
                    <ChevronDown className="h-3 w-3" />
                    Show {others.length} more
                  </button>
                </td>
              </tr>
            )}

            {showAll && others.map((lvl, i) => (
              <tr key={`o-${lvl.exchange}-${i}`} className="border-b border-gray-800/20">
                <td className="py-1 text-gray-600">{lvl.exchange}</td>
                <td className={`py-1 text-right font-mono ${dimColor}`}>
                  {parseFloat(lvl.price).toLocaleString()}
                </td>
                <td className="py-1 text-right font-mono text-gray-600">
                  {parseFloat(lvl.qty)}
                </td>
              </tr>
            ))}

            {showAll && hasOthers && (
              <tr>
                <td colSpan={3} className="py-1">
                  <button
                    onClick={() => setShowAll(false)}
                    className="flex w-full items-center justify-center gap-1 text-[10px] text-gray-500 hover:text-gray-300"
                  >
                    <ChevronUp className="h-3 w-3" />
                    Hide
                  </button>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
