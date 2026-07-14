/** Chart time helpers — UTC candles displayed in a user-chosen timezone. */

export const CHART_TIMEZONES = [
  { id: "local", label: "Browser local" },
  { id: "America/New_York", label: "US Eastern (NYSE)" },
  { id: "America/Chicago", label: "US Central" },
  { id: "America/Denver", label: "US Mountain" },
  { id: "America/Los_Angeles", label: "US Pacific" },
  { id: "UTC", label: "UTC" },
  { id: "Europe/London", label: "London" },
  { id: "Europe/Berlin", label: "Berlin / Paris" },
  { id: "Asia/Tokyo", label: "Tokyo" },
  { id: "Asia/Hong_Kong", label: "Hong Kong" },
  { id: "Australia/Sydney", label: "Sydney" },
] as const;

export type ChartTimezoneId = (typeof CHART_TIMEZONES)[number]["id"];

const PREF_KEY = "evolve_chart_timezone";

export function resolveChartTimeZone(pref: string | null | undefined): string | undefined {
  const id = (pref || "").trim();
  if (!id || id === "local") return undefined; // browser default
  return id;
}

export function loadCachedChartTimezone(): string {
  try {
    return sessionStorage.getItem(PREF_KEY) || "America/New_York";
  } catch {
    return "America/New_York";
  }
}

export function cacheChartTimezone(id: string): void {
  try {
    sessionStorage.setItem(PREF_KEY, id);
  } catch {
    /* ignore */
  }
}

/** Parse chart/candle time to epoch ms (UTC). */
export function parseChartTimeMs(t: string | number): number {
  if (typeof t === "number") {
    return t < 1e12 ? t * 1000 : t;
  }
  if (/^\d+(\.\d+)?$/.test(t)) {
    const n = Number(t);
    return n < 1e12 ? n * 1000 : n;
  }
  return Date.parse(t);
}

/**
 * Shift a UTC unix-seconds bar into "display seconds" so Lightweight Charts
 * (which always treats unix as UTC) shows wall-clock time for ``timeZone``.
 * Noon ET (16:00Z) becomes 12:00 on the axis instead of 16:00.
 */
export function utcSecToDisplaySec(utcSec: number, timeZone?: string): number {
  if (!timeZone || timeZone === "UTC") return utcSec;
  const d = new Date(utcSec * 1000);
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hourCycle: "h23",
  }).formatToParts(d);
  const g = (type: string) => parts.find((p) => p.type === type)?.value ?? "0";
  const asUtcMs = Date.UTC(
    Number(g("year")),
    Number(g("month")) - 1,
    Number(g("day")),
    Number(g("hour")),
    Number(g("minute")),
    Number(g("second")),
  );
  return Math.floor(asUtcMs / 1000);
}

/** 12-hour clock labels — never military / 24h. */
export function formatChartTime(
  t: string | number,
  intraday: boolean,
  timeZone?: string,
  /** When true, ``t`` is already shifted display-unix (wall clock encoded as UTC). */
  displayShifted = false,
): string {
  if (!intraday) {
    const s = String(t);
    if (typeof t === "number" || /^\d+$/.test(s)) {
      const ms = parseChartTimeMs(t);
      if (!Number.isFinite(ms)) return s;
      const opts: Intl.DateTimeFormatOptions = {
        year: "numeric",
        month: "short",
        day: "numeric",
        timeZone: displayShifted ? "UTC" : (timeZone || undefined),
      };
      return new Date(ms).toLocaleDateString("en-US", opts);
    }
    return s.length >= 10 ? s.slice(0, 10) : s;
  }
  const ms = parseChartTimeMs(t);
  if (!Number.isFinite(ms)) return String(t);
  const opts: Intl.DateTimeFormatOptions = {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  };
  // Shifted display times: format as UTC so noon stays noon.
  // Raw UTC data: format in the chosen zone.
  if (displayShifted || !timeZone) {
    opts.timeZone = displayShifted ? "UTC" : undefined;
  } else {
    opts.timeZone = timeZone;
  }
  return new Date(ms).toLocaleString("en-US", opts);
}

/**
 * Short tick-mark labels for the time axis (≤ ~8–10 chars).
 * TickMarkType: 0 Year, 1 Month, 2 DayOfMonth, 3 Time, 4 TimeWithSeconds.
 */
export function formatTickMark(
  time: string | number | { year: number; month: number; day: number },
  tickMarkType: number,
  timeZone?: string,
  displayShifted = false,
): string {
  let ms: number;
  if (typeof time === "object" && time !== null && "year" in time) {
    ms = Date.UTC(time.year, time.month - 1, time.day);
  } else {
    ms = parseChartTimeMs(time as string | number);
  }
  if (!Number.isFinite(ms)) return "";

  const zone = displayShifted ? "UTC" : (timeZone || undefined);
  const base: Intl.DateTimeFormatOptions = { timeZone: zone };

  // Prefer date ticks over clock when the library asks for day/month/year
  if (tickMarkType <= 2) {
    if (tickMarkType === 0) {
      return new Date(ms).toLocaleDateString("en-US", { ...base, year: "numeric" });
    }
    if (tickMarkType === 1) {
      return new Date(ms).toLocaleDateString("en-US", { ...base, month: "short" });
    }
    return new Date(ms).toLocaleDateString("en-US", {
      ...base, month: "short", day: "numeric",
    });
  }

  // Time ticks — always 12-hour (e.g. "9:30 AM")
  return new Date(ms).toLocaleTimeString("en-US", {
    ...base,
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });
}
