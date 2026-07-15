/** Align daily news/strategy markers with whatever candle timeline is plotted. */

import type { Candle, ChartEvent } from "./api";

/** Letter + color meaning for volume/event marks (shown in chart hover note). */
export function describeEventMark(e: ChartEvent): string {
  const tier = String(e.tier || "significant");
  const letter = String(
    e.text
      ?? (tier === "notable" ? "n" : tier === "event_move" ? "E" : tier === "provisional" ? "LIVE" : "N"),
  );
  const up = (e.price_change_pct ?? 0) >= 0
    || (e.color || "").toLowerCase().includes("00ff")
    || (e.color || "").toLowerCase().includes("7eb6")
    || (e.color || "").toLowerCase().includes("f0c7");

  let name = "Full volume spike";
  let meaning =
    "hit the full spike bar (≥~2× volume with ≥~2% move, or ≥~3× volume alone)";
  let colorMeaning = up ? "green = up-day" : "red = down-day";

  if (tier === "notable") {
    name = "Notable volume";
    meaning = "elevated volume below the full 2×/2% or 3× spike bar";
    colorMeaning = up ? "gold = up-day" : "orange = down-day";
  } else if (tier === "event_move") {
    name = "Large session move";
    meaning = "big price change (≥~1.2%) without extreme volume";
    colorMeaning = up ? "blue = up-day" : "purple = down-day";
  } else if (tier === "provisional") {
    name = "Live volume spike";
    meaning = "provisional intraday spike (may change by close)";
    colorMeaning = "amber = live";
  }

  const vol = e.volume_ratio != null ? ` · ${Number(e.volume_ratio).toFixed(1)}× vol` : "";
  const move = e.price_change_pct != null
    ? ` · ${(Number(e.price_change_pct) * 100).toFixed(1)}%`
    : "";
  return `[${letter}] ${name}${vol}${move} — ${meaning}. Color: ${colorMeaning}.`;
}

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
