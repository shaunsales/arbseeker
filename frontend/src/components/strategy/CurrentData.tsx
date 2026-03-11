import { useState } from "react";
import type { StrategyManifest } from "@/types/api";
import { deleteStrategy } from "@/api/strategy";
import { Button } from "@/components/ui/button";
import DataPreview from "./DataPreview";

interface Props {
  className: string;
  manifest: StrategyManifest;
  errors: string[];
  onDeleted: () => void;
  chartExpanded: boolean;
  onToggleExpand: () => void;
}

export default function CurrentData({
  className,
  manifest,
  errors,
  onDeleted,
  chartExpanded,
  onToggleExpand,
}: Props) {
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [previewInterval, setPreviewInterval] = useState<string | null>(null);

  async function handleDelete() {
    setDeleting(true);
    try {
      const res = await deleteStrategy(className);
      if (res.success) onDeleted();
    } catch (e) {
      console.error("Delete failed:", e);
    }
    setDeleting(false);
    setConfirmDelete(false);
  }

  return (
    <>
      <div className="rounded-lg border border-gray-800 bg-gray-900 p-4 space-y-3">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-400">
          Current Data
        </h3>

        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <span className="text-xs text-gray-500">Tradable Range</span>
            <p className="font-mono text-xs text-gray-200">
              {manifest.date_range.start} → {manifest.date_range.end}
            </p>
          </div>
          <div>
            <span className="text-xs text-gray-500">Built At</span>
            <p className="font-mono text-xs text-gray-200">
              {manifest.built_at.slice(0, 19)}
            </p>
          </div>
        </div>

        {/* File rows */}
        <div className="space-y-1">
          {Object.entries(manifest.quality).map(([interval, q]) => (
            <button
              key={interval}
              onClick={() =>
                setPreviewInterval(
                  previewInterval === interval ? null : interval
                )
              }
              className={`flex w-full items-center gap-3 rounded border px-2.5 py-1.5 text-left text-xs transition ${
                previewInterval === interval
                  ? "border-blue-600/50 bg-gray-800"
                  : "border-gray-800 hover:border-blue-600/30 hover:bg-gray-800/50"
              }`}
            >
              <span className="w-8 flex-shrink-0 font-medium text-blue-400">
                {interval}
              </span>
              <span className="font-mono text-gray-500">
                {interval}.parquet
              </span>
              <span className="ml-auto text-gray-400">
                {q.bars.toLocaleString()} bars
              </span>
              <span
                className={
                  q.coverage_pct >= 99
                    ? "text-green-400"
                    : q.coverage_pct >= 90
                      ? "text-yellow-400"
                      : "text-red-400"
                }
              >
                {q.coverage_pct}%
              </span>
              {q.null_indicator_bars != null && q.null_indicator_bars > 0 && (
                <span className="text-yellow-500">
                  {q.null_indicator_bars} NaN
                </span>
              )}
              <span
                className={`transition ${
                  previewInterval === interval
                    ? "text-blue-400"
                    : "text-gray-600"
                }`}
              >
                →
              </span>
            </button>
          ))}
        </div>

        {/* Errors */}
        {errors.length > 0 && (
          <div className="rounded border border-red-800 bg-red-900/30 p-2.5 text-xs text-red-300">
            <p className="mb-1 font-semibold">Validation errors:</p>
            {errors.map((err, i) => (
              <p key={i}>{err}</p>
            ))}
          </div>
        )}

        {/* Delete button */}
        <div className="pt-1">
          {!confirmDelete ? (
            <Button
              variant="outline"
              onClick={() => setConfirmDelete(true)}
              className="w-full border-red-800 bg-red-900/40 text-red-300 hover:bg-red-900/60 hover:text-red-200"
            >
              Delete Data
            </Button>
          ) : (
            <div className="flex gap-2">
              <Button
                variant="destructive"
                onClick={handleDelete}
                disabled={deleting}
                className="flex-1"
              >
                {deleting ? "Deleting..." : "Confirm Delete"}
              </Button>
              <Button
                variant="secondary"
                onClick={() => setConfirmDelete(false)}
                disabled={deleting}
              >
                Cancel
              </Button>
            </div>
          )}
        </div>
      </div>

      {/* Preview panel */}
      {previewInterval && (
        <DataPreview
          className={className}
          interval={previewInterval}
          onClose={() => setPreviewInterval(null)}
          expanded={chartExpanded}
          onToggleExpand={onToggleExpand}
        />
      )}
    </>
  );
}
