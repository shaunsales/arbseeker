import { get } from "./client";
import type { BasisPreviewResponse } from "@/types/api";

export function getBasisList(): Promise<{
  data_tree: Record<string, unknown>;
  basis_files: { ticker: string; interval: string; periods: string[]; path: string }[];
}> {
  return get("/basis/");
}

export function getBasisFilesList(): Promise<
  { ticker: string; interval: string; periods: string[]; path: string }[]
> {
  return get("/basis/list");
}

export function checkOverlap(params: {
  base_venue: string;
  base_market: string;
  base_ticker: string;
  quote_venue: string;
  quote_market: string;
  quote_ticker: string;
  interval: string;
}): Promise<Record<string, unknown>> {
  const qs = new URLSearchParams(params).toString();
  return get(`/basis/check-overlap?${qs}`);
}

export function createBasis(body: Record<string, unknown>): Promise<{
  success: boolean;
  path?: string;
  bars?: number;
  coverage_pct?: number;
  error?: string;
}> {
  return import("./client").then((c) => c.post("/basis/create", body));
}

export function getBasisPreview(
  ticker: string,
  interval: string,
  period?: string,
): Promise<BasisPreviewResponse> {
  const qs = period ? `?period=${period}` : "";
  return get<BasisPreviewResponse>(`/basis/preview/${ticker}/${interval}${qs}`);
}
