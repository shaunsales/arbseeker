import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  connect,
  getStrategies,
  getExchanges,
  getCurrencyPairs,
  getBalances,
  getPositions,
  getOrders,
  type ConnectResult,
} from "@/api/coinroutes";
import { Badge } from "@/components/ui/badge";
import {
  Plug,
  Loader2,
  RefreshCw,
  Building2,
  Coins,
  Wallet,
  BarChart3,
  ScrollText,
} from "lucide-react";

export default function CoinRoutesPage() {
  const [connected, setConnected] = useState(false);

  const connectMut = useMutation({
    mutationFn: connect,
    onSuccess: (data) => setConnected(data.connected),
  });

  const result = connectMut.data as ConnectResult | undefined;

  return (
    <div className="flex flex-col gap-6 p-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold text-gray-100">CoinRoutes</h1>
          <p className="text-sm text-gray-500">
            Smart Order Router — connect, explore venues, and manage orders
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
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <DiscoveryPanel
            title="Strategies"
            icon={BarChart3}
            queryKey="cr-strategies"
            queryFn={getStrategies}
          />
          <DiscoveryPanel
            title="Exchanges"
            icon={Building2}
            queryKey="cr-exchanges"
            queryFn={getExchanges}
          />
          <DiscoveryPanel
            title="Currency Pairs"
            icon={Coins}
            queryKey="cr-pairs"
            queryFn={getCurrencyPairs}
          />
          <DiscoveryPanel
            title="Balances"
            icon={Wallet}
            queryKey="cr-balances"
            queryFn={getBalances}
          />
          <DiscoveryPanel
            title="Positions"
            icon={BarChart3}
            queryKey="cr-positions"
            queryFn={getPositions}
          />
          <DiscoveryPanel
            title="Recent Orders"
            icon={ScrollText}
            queryKey="cr-orders"
            queryFn={getOrders}
          />
        </div>
      )}
    </div>
  );
}

function DiscoveryPanel({
  title,
  icon: Icon,
  queryKey,
  queryFn,
}: {
  title: string;
  icon: React.ComponentType<{ className?: string }>;
  queryKey: string;
  queryFn: () => Promise<unknown>;
}) {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: [queryKey],
    queryFn,
    enabled: false,
  });

  const hasData = data !== undefined;

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm font-medium text-gray-300">
          <Icon className="h-4 w-4 text-gray-500" />
          {title}
        </div>
        <button
          onClick={() => refetch()}
          disabled={isLoading}
          className="inline-flex items-center gap-1.5 rounded px-2.5 py-1 text-xs font-medium text-gray-400 hover:bg-gray-800 hover:text-gray-200 disabled:opacity-50"
        >
          {isLoading ? (
            <Loader2 className="h-3 w-3 animate-spin" />
          ) : (
            <RefreshCw className="h-3 w-3" />
          )}
          {hasData ? "Refresh" : "Fetch"}
        </button>
      </div>

      {error && (
        <p className="mt-2 text-xs text-red-400">
          {error instanceof Error ? error.message : "Request failed"}
        </p>
      )}

      {hasData && (
        <pre className="mt-3 max-h-64 overflow-auto rounded bg-gray-950 p-3 text-xs text-gray-400">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  );
}
