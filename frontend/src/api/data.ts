import { get, post } from "./client";
import type { DataTreeResponse, DataPreviewResponse, DownloadResponse } from "@/types/api";

export function getDataTree(): Promise<DataTreeResponse> {
  return get<DataTreeResponse>("/data/");
}

export function getDataPreview(
  venue: string,
  market: string,
  ticker: string,
  interval: string,
  page = 1,
  pageSize = 100,
  startDate?: string,
  endDate?: string,
): Promise<DataPreviewResponse> {
  const params = new URLSearchParams({
    page: String(page),
    page_size: String(pageSize),
  });
  if (startDate) params.set("start_date", startDate);
  if (endDate) params.set("end_date", endDate);
  return get<DataPreviewResponse>(
    `/data/preview/${venue}/${market}/${ticker}/${interval}?${params}`
  );
}

export function downloadData(req: {
  venue: string;
  market: string;
  ticker: string;
  interval: string;
  start_month: string;
  end_month: string;
}): Promise<DownloadResponse> {
  return post<DownloadResponse>("/data/download", req);
}

export function getDownloadStatus(jobId: string): Promise<{
  status: string;
  progress: number;
  message: string;
}> {
  return get(`/data/download/status/${jobId}`);
}

export function getSymbols(market = "futures"): Promise<{ symbols: string[] }> {
  return get(`/data/symbols/${market}`);
}
