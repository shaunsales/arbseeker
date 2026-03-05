// ── Data page types ──

// data_tree: { venue: { market: { ticker: { interval: period[] } } } }
export type DataTree = Record<string, Record<string, Record<string, Record<string, string[]>>>>;

export interface DataTreeResponse {
  data_tree: DataTree;
  intervals: string[];
}

export interface OhlcvBar {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface VolumeBar {
  time: number;
  value: number;
  color: string;
}

export interface OhlcvChartData {
  ohlcv: OhlcvBar[];
  volume: VolumeBar[];
}

export interface OhlcvTableRow {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface ValidationReport {
  total_bars: number;
  expected_bars: number;
  coverage_pct: number;
  is_valid: boolean;
  gap_count: number;
  total_missing_bars: number;
  null_count: number;
  zero_volume_count: number;
}

export interface DataPreviewResponse {
  venue: string;
  market: string;
  ticker: string;
  interval: string;
  available_periods: string[];
  selected_periods: string[];
  summary: Record<string, unknown>;
  report: ValidationReport;
  chart_data: OhlcvChartData;
  table_data: OhlcvTableRow[];
  pagination: Pagination;
  date_range: {
    min: string;
    max: string;
    start: string;
    end: string;
  };
  error?: string;
}

export interface DownloadResponse {
  job_id: string;
  status: string;
}

export interface Pagination {
  page: number;
  page_size: number;
  total_rows: number;
  total_pages: number;
}

// ── Basis page types ──

export interface BasisFileInfo {
  ticker: string;
  interval: string;
  periods: string[];
  path: string;
}

export interface BasisPreviewResponse {
  ticker: string;
  interval: string;
  chart_interval: string;
  quote_venues: string[];
  stats: {
    bars: number;
    start: string;
    end: string;
    quality_ok: number;
    quality_pct: number;
    quality_breakdown: Record<string, number>;
  };
  venue_stats: Record<string, Record<string, unknown>>;
  chart_data: Record<string, { time: number; value: number; color?: string }[]>;
  error?: string;
}

// ── Strategy page types ──

export interface StrategyListItem {
  class_name: string;
  module: string;
  has_data_spec: boolean;
}

export interface IndicatorSpec {
  name: string;
  params: Record<string, number>;
  warmup_bars: number;
}

export interface IntervalSpec {
  interval: string;
  indicators: IndicatorSpec[];
  is_price_only: boolean;
}

export interface DateRange {
  start: string;
  end: string;
}

export interface QualityMetric {
  bars: number;
  coverage_pct: number;
  null_indicator_bars?: number;
}

export interface StrategyManifest {
  date_range: DateRange;
  built_at: string;
  quality: Record<string, QualityMetric>;
}

export interface StrategyStatus {
  class_name: string;
  spec: {
    venue: string;
    market: string;
    ticker: string;
    intervals: IntervalSpec[];
  };
  has_manifest: boolean;
  manifest: StrategyManifest | null;
  errors: string[];
}

export interface AvailableDates {
  months: string[];
  raw_earliest: string;
  earliest_start: string;
  latest_end: string;
  warmup_months: number;
  per_interval: Record<string, {
    months: string[];
    start: string;
    end: string;
    count: number;
  }>;
}

export interface BuildRequest {
  class_name: string;
  start_date: string;
  end_date: string;
}

export interface PreviewData {
  strategy_name: string;
  class_name: string;
  interval: string;
  chart_data: string; // JSON string for chart
  overlay_cols: string[];
  separate_cols: string[];
  all_cols: string[];
  table_data: Record<string, string | number>[];
  pagination: Pagination;
}
