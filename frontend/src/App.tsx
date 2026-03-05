import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import AppLayout from "@/components/layout/AppLayout";
import DataPage from "@/pages/DataPage";
import DownloadPage from "@/pages/DownloadPage";
import BasisPage from "@/pages/BasisPage";
import StrategyPage from "@/pages/StrategyPage";
import BacktestPage from "@/pages/BacktestPage";

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
      <BrowserRouter>
        <Routes>
          <Route element={<AppLayout />}>
            <Route path="/data" element={<DataPage />} />
            <Route path="/download" element={<DownloadPage />} />
            <Route path="/basis" element={<BasisPage />} />
            <Route path="/strategy" element={<StrategyPage />} />
            <Route path="/backtest" element={<BacktestPage />} />
            <Route path="*" element={<Navigate to="/strategy" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
