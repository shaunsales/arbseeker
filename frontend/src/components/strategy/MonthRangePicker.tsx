import { useMemo } from "react";
import SharedMonthRangePicker from "@/components/ui/month-range-picker";
import type { AvailableDates } from "@/types/api";

interface Props {
  dates: AvailableDates;
  startDate: string;
  endDate: string;
  onRangeChange: (start: string, end: string) => void;
}

export default function MonthRangePicker({
  dates,
  startDate,
  endDate,
  onRangeChange,
}: Props) {
  const warmupMonths = useMemo(
    () => dates.months.filter((m) => m < dates.earliest_start),
    [dates.months, dates.earliest_start],
  );

  return (
    <SharedMonthRangePicker
      months={dates.months}
      startDate={startDate}
      endDate={endDate}
      onRangeChange={onRangeChange}
      warmupMonths={warmupMonths}
      warmupLabel={`Warmup (${dates.warmup_months}mo)`}
      perInterval={dates.per_interval}
    />
  );
}
