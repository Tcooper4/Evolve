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

export function formatChartTime(
  t: string | number,
  intraday: boolean,
  timeZone?: string,
): string {
  if (!intraday) {
    const s = String(t);
    return s.length >= 10 && !/^\d+$/.test(s) ? s.slice(0, 10) : s;
  }
  let ms: number;
  if (typeof t === "number") {
    ms = t < 1e12 ? t * 1000 : t;
  } else if (/^\d+$/.test(t)) {
    const n = Number(t);
    ms = n < 1e12 ? n * 1000 : n;
  } else {
    ms = Date.parse(t);
  }
  if (!Number.isFinite(ms)) return String(t);
  const opts: Intl.DateTimeFormatOptions = {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZoneName: "short",
  };
  if (timeZone) opts.timeZone = timeZone;
  return new Date(ms).toLocaleString(undefined, opts);
}
