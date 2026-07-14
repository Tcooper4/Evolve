/** Align daily news/strategy markers with whatever candle timeline is plotted. */

import type { Candle } from "./api";

/** Calendar day key for UTC/unix/ISO candle times and YYYY-MM-DD markers. */
export function chartDayKey(time: string | number): string {
  if (typeof time === "number" || /^\d+(\.\d+)?$/.test(String(time))) {
    const sec = typeof time === "number" ? time : Number(time);
    if (Number.isFinite(sec) && sec > 1e8) {
      return new Date(sec * 1000).toISOString().slice(0, 10);
    }
  }
  const s = String(time);
  if (s.includes("T") || s.length > 10) {
    const raw = s.endsWith("Z") || /[+-]\d{2}:?\d{2}$/.test(s) ? s : `${s}Z`;
    const ms = Date.parse(raw);
    if (Number.isFinite(ms)) return new Date(ms).toISOString().slice(0, 10);
  }
  return s.slice(0, 10);
}

/** Last candle time string per calendar day in the plotted series. */
export function lastCandleTimeByDay(candles: Candle[]): Map<string, string> {
  const map = new Map<string, string>();
  for (const c of candles) {
    map.set(chartDayKey(c.time), c.time);
  }
  return map;
}

export function candleDaySet(candles: Candle[]): Set<string> {
  return new Set(candles.map((c) => chartDayKey(c.time)));
}

/** Keep only markers whose calendar day appears in the plotted candles. */
export function filterMarkersToCandles<T extends { time: string }>(
  markers: T[],
  candles: Candle[],
): T[] {
  if (!markers.length || !candles.length) return [];
  const days = candleDaySet(candles);
  return markers.filter((m) => days.has(chartDayKey(m.time)));
}

export type OverlaySeriesLike = {
  id: string;
  label?: string;
  color: string;
  style?: "solid" | "dashed" | "dotted";
  points: Array<{ time: string; value: number }>;
};

/**
 * Remap daily overlay points onto real plotted bar times (last bar of that day).
 * Drops series that cannot draw (≥2 points) after filtering.
 */
export function alignOverlaySeriesToCandles(
  series: OverlaySeriesLike[],
  candles: Candle[],
): OverlaySeriesLike[] {
  if (!series.length || !candles.length) return [];
  const lastByDay = lastCandleTimeByDay(candles);
  const out: OverlaySeriesLike[] = [];
  for (const s of series) {
    const points: Array<{ time: string; value: number }> = [];
    for (const p of s.points) {
      const ct = lastByDay.get(chartDayKey(p.time));
      if (!ct || !Number.isFinite(p.value)) continue;
      points.push({ time: ct, value: p.value });
    }
    if (points.length >= 2) {
      out.push({ ...s, points });
    }
  }
  return out;
}
