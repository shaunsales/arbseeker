import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { TooltipProvider } from "@/components/ui/tooltip";
import AppLayout from "@/components/layout/AppLayout";
import DataPage from "@/pages/DataPage";
import DownloadPage from "@/pages/DownloadPage";
import StrategyPage from "@/pages/StrategyPage";
import BasisPage from "@/pages/BasisPage";
import BacktestPage from "@/pages/BacktestPage";
import CoinRoutesPage from "@/pages/CoinRoutesPage";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 1,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <BrowserRouter>
          <Routes>
            <Route element={<AppLayout />}>
              {/* Data */}
              <Route path="/data" element={<DataPage />} />
              <Route path="/download" element={<DownloadPage />} />

              {/* Strategies */}
              <Route path="/strategies/single-asset" element={<StrategyPage />} />
              <Route path="/strategies/basis" element={<BasisPage />} />
              <Route path="/strategies/multi-leg" element={<MultiLegPlaceholder />} />

              {/* Backtest */}
              <Route path="/backtest" element={<BacktestPage />} />

              {/* Execution */}
              <Route path="/coinroutes" element={<CoinRoutesPage />} />

              <Route path="*" element={<Navigate to="/data" replace />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  );
}

function MultiLegPlaceholder() {
  return (
    <div className="flex h-full items-center justify-center">
      <div className="text-center">
        <p className="text-sm text-gray-500">Multi-Leg Strategy</p>
        <p className="mt-1 text-xs text-gray-600">
          Coming soon — build strategies using MultiLeggedStrategy base class
        </p>
      </div>
    </div>
  );
}
