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

  let name = "Busy day";
  let meaning = "unusually high volume (and often a big move)";
  let color = up ? "green up" : "red down";

  if (tier === "notable") {
    name = "Above-average volume";
    meaning = "louder than usual, but not a full spike";
    color = up ? "gold up" : "orange down";
  } else if (tier === "event_move") {
    name = "Big price move";
    meaning = "large % change without extreme volume";
    color = up ? "blue up" : "purple down";
  } else if (tier === "provisional") {
    name = "Live volume spike";
    meaning = "intraday alert — may change by close";
    color = "amber live";
  }

  const vol = e.volume_ratio != null ? ` · ${Number(e.volume_ratio).toFixed(1)}× vol` : "";
  const move = e.price_change_pct != null
    ? ` · ${(Number(e.price_change_pct) * 100).toFixed(1)}%`
    : "";
  return `[${letter}] ${name}${vol}${move} — ${meaning} (${color}).`;
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
  points?: Array<{ time: string; value: number }>;
  /** Fixed $ level (gamma flip / wing guides) — drawn as a candle price line. */
  priceLevel?: number;
};

function isFlatValues(vals: number[]): boolean {
  if (!vals.length) return false;
  const first = vals[0];
  return vals.every((v) => Math.abs(v - first) < 1e-9);
}

/**
 * Remap daily overlay points onto real plotted bar times (last bar of that day).
 * Constant guides (gamma flip, wings) become priceLevel so they draw on 1D too.
 * Time-varying series still need ≥2 points after filtering.
 */
export function alignOverlaySeriesToCandles(
  series: OverlaySeriesLike[],
  candles: Candle[],
): OverlaySeriesLike[] {
  if (!series.length || !candles.length) return [];
  const lastByDay = lastCandleTimeByDay(candles);
  const out: OverlaySeriesLike[] = [];
  for (const s of series) {
    if (s.priceLevel != null && Number.isFinite(s.priceLevel)) {
      out.push({ ...s, points: [] });
      continue;
    }
    const points: Array<{ time: string; value: number }> = [];
    for (const p of s.points ?? []) {
      const ct = lastByDay.get(chartDayKey(p.time));
      if (!ct || !Number.isFinite(p.value)) continue;
      points.push({ time: ct, value: p.value });
    }
    if (!points.length) continue;
    const vals = points.map((p) => p.value);
    if (isFlatValues(vals)) {
      out.push({ ...s, points: [], priceLevel: vals[0] });
      continue;
    }
    if (points.length >= 2) {
      out.push({ ...s, points });
    }
  }
  return out;
}
