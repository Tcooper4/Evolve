import { useEffect, useRef, useState } from "react";
import {
  createChart,
  ColorType,
  CrosshairMode,
  LineStyle,
  type IChartApi,
  type ISeriesApi,
  type CandlestickData,
  type HistogramData,
  type MouseEventParams,
  type SeriesMarker,
  type Time,
} from "lightweight-charts";
import type { Candle } from "./api";
import {
  formatChartTime,
  formatTickMark,
  resolveChartTimeZone,
  utcSecToDisplaySec,
} from "./chartTime";
import { chartDayKey } from "./chartMarkers";

export interface ChartMarker {
  time: string;
  position?: "aboveBar" | "belowBar";
  color?: string;
  shape?: "circle" | "square" | "arrowUp" | "arrowDown";
  text?: string;
  title?: string;
}

export interface ChartOverlaySeries {
  id: string;
  label?: string;
  color: string;
  style?: "solid" | "dashed" | "dotted";
  /** Time series on the candle price scale (SMA, Bollinger, etc.). */
  points?: Array<{ time: string; value: number }>;
  /**
   * Fixed dollar level (gamma flip, wing guides). Drawn with createPriceLine
   * so it stays on the candle scale — not the volume pane at the bottom.
   */
  priceLevel?: number;
}

interface Hover {
  o: number; h: number; l: number; c: number; v?: number; t: string;
  note?: string;
}

function toChartTime(t: string, timeZone?: string): Time {
  if (t.includes("T") || t.length > 10) {
    const ms = Date.parse(t);
    if (Number.isFinite(ms)) {
      const utcSec = Math.floor(ms / 1000);
      return utcSecToDisplaySec(utcSec, timeZone) as Time;
    }
  }
  return t.slice(0, 10) as Time;
}

function toCandleData(candles: Candle[], timeZone?: string): CandlestickData[] {
  return candles.map((c) => ({
    time: toChartTime(c.time, timeZone),
    open: c.open,
    high: c.high,
    low: c.low,
    close: c.close,
  })) as CandlestickData[];
}

function toVolumeData(candles: Candle[], timeZone?: string): HistogramData[] {
  return candles.map((c) => ({
    time: toChartTime(c.time, timeZone),
    // Per-bar volume only — never session totals
    value: Number(c.volume) || 0,
    color: c.close >= c.open
      ? "rgba(46,234,139,0.35)"
      : "rgba(255,84,112,0.35)",
  }));
}

function formatVolume(v: number): string {
  if (!Number.isFinite(v) || v <= 0) return "—";
  if (v >= 1e9) return `${(v / 1e9).toFixed(2)}B`;
  if (v >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(1)}K`;
  return v.toFixed(0);
}

function findCandle(
  candles: Candle[],
  chartTime: string | number,
  timeZone?: string,
): Candle | undefined {
  // Prefer exact match against the same display time we fed the series
  if (typeof chartTime === "number" || /^\d+$/.test(String(chartTime))) {
    const target = typeof chartTime === "number" ? chartTime : Number(chartTime);
    const hit = candles.find((c) => {
      const ct = toChartTime(c.time, timeZone);
      return typeof ct === "number" && ct === target;
    });
    if (hit) return hit;
  }
  const t = String(chartTime);
  return candles.find((c) => {
    if (c.time === t) return true;
    if (c.time.slice(0, 10) === t.slice(0, 10) && !c.time.includes("T")) return true;
    return false;
  });
}

function lineStyleOf(style?: ChartOverlaySeries["style"]): LineStyle {
  if (style === "solid") return LineStyle.Solid;
  if (style === "dashed") return LineStyle.Dashed;
  return LineStyle.Dotted;
}

export default function Chart({
  candles,
  markers = [],
  overlays = [],
  live = false,
  timeZone,
}: {
  candles: Candle[];
  markers?: ChartMarker[];
  overlays?: ChartOverlaySeries[];
  live?: boolean;
  timeZone?: string;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const overlaySeriesRef = useRef<Map<string, ISeriesApi<"Line">>>(new Map());
  const priceLinesRef = useRef<Map<string, ReturnType<ISeriesApi<"Candlestick">["createPriceLine"]>>>(new Map());
  const candlesRef = useRef(candles);
  const markersRef = useRef(markers);
  const fittedOnceRef = useRef(false);
  const modeRef = useRef("");
  const [hover, setHover] = useState<Hover | null>(null);
  const [activeNote, setActiveNote] = useState<string | null>(null);
  const intraday = candles.some((c) => c.time.includes("T") || c.time.length > 10);
  const tz = resolveChartTimeZone(timeZone);
  // Prefer America/New_York for US equity charts when prefs fall through
  const displayTz = tz || (intraday ? "America/New_York" : undefined);
  // Data is shifted so LWC axis (UTC) = wall clock in displayTz
  const shiftDisplay = Boolean(intraday && displayTz);

  candlesRef.current = candles;
  markersRef.current = markers;

  useEffect(() => {
    if (!containerRef.current || candles.length === 0) return;

    const mode = `${intraday ? "i" : "d"}|${displayTz || "local"}`;
    if (chartRef.current && modeRef.current && modeRef.current !== mode) {
      try {
        const chart = chartRef.current;
        const onMove = (chart as unknown as { __onMove?: (p: MouseEventParams) => void }).__onMove;
        if (onMove) chart.unsubscribeCrosshairMove(onMove);
        chart.remove();
      } catch { /* skip */ }
      chartRef.current = null;
      candleSeriesRef.current = null;
      volumeSeriesRef.current = null;
      overlaySeriesRef.current = new Map();
      priceLinesRef.current = new Map();
      fittedOnceRef.current = false;
    }
    modeRef.current = mode;

    if (!chartRef.current) {
      const chart = createChart(containerRef.current, {
        layout: {
          background: { type: ColorType.Solid, color: "transparent" },
          textColor: "#5d6b84",
          fontFamily: "Inter, system-ui, sans-serif",
        },
        grid: {
          vertLines: { color: "rgba(27,36,56,0.6)" },
          horzLines: { color: "rgba(27,36,56,0.6)" },
        },
        crosshair: {
          mode: CrosshairMode.Magnet,
          vertLine: { color: "#00d4ff", width: 1, style: 3, labelBackgroundColor: "#0f1522" },
          horzLine: { color: "#00d4ff", width: 1, style: 3, labelBackgroundColor: "#0f1522" },
        },
        rightPriceScale: { borderColor: "#1b2438" },
        timeScale: {
          borderColor: "#1b2438",
          fixLeftEdge: true,
          fixRightEdge: true,
          lockVisibleTimeRangeOnResize: true,
          timeVisible: intraday,
          secondsVisible: false,
          barSpacing: Math.max(3, Math.min(8, Math.floor(720 / Math.max(candles.length, 1)))),
          minBarSpacing: 2,
          tickMarkFormatter: (time: Time, tickMarkType: number) => formatTickMark(
            time as string | number | { year: number; month: number; day: number },
            tickMarkType,
            displayTz,
            shiftDisplay,
          ),
        },
        handleScroll: {
          mouseWheel: true,
          pressedMouseMove: true,
          horzTouchDrag: true,
          vertTouchDrag: false,
        },
        handleScale: {
          axisPressedMouseMove: { time: false, price: true },
          mouseWheel: true,
          pinch: true,
        },
        autoSize: true,
        localization: {
          timeFormatter: (time: Time) => formatChartTime(
            typeof time === "number" ? time : String(time),
            intraday,
            displayTz,
            shiftDisplay,
          ),
          locale: "en-US",
        },
      });

      const series = chart.addCandlestickSeries({
        upColor: "#2eea8b",
        downColor: "#ff5470",
        borderVisible: false,
        wickUpColor: "#2eea8b",
        wickDownColor: "#ff5470",
      });
      series.priceScale().applyOptions({
        scaleMargins: { top: 0.05, bottom: 0.22 },
      });

      const volume = chart.addHistogramSeries({
        priceFormat: { type: "volume" },
        priceScaleId: "vol",
        lastValueVisible: false,
        priceLineVisible: false,
      });
      chart.priceScale("vol").applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
        visible: false,
        borderVisible: false,
      });

      chartRef.current = chart;
      candleSeriesRef.current = series;
      volumeSeriesRef.current = volume;

      const onMove = (p: MouseEventParams) => {
        const candleSeries = candleSeriesRef.current;
        if (!candleSeries) return;
        const d = p.seriesData.get(candleSeries) as CandlestickData | undefined;
        if (d && p.time != null) {
          const match = findCandle(
            candlesRef.current,
            p.time as string | number,
            displayTz,
          );
          const day = match
            ? chartDayKey(match.time)
            : chartDayKey(p.time as string | number);
          const noteByDay = new Map<string, string>();
          for (const m of markersRef.current) {
            const key = chartDayKey(m.time);
            const bit = m.title || m.text || "";
            if (!bit) continue;
            const prev = noteByDay.get(key);
            noteByDay.set(key, prev ? `${prev} · ${bit}` : bit);
          }
          const note = noteByDay.get(day) || undefined;
          // Prefer per-candle volume from our data (histogram can lag/mismatch)
          const v = Number(match?.volume ?? 0);
          setHover({
            o: d.open, h: d.high, l: d.low, c: d.close,
            v: Number.isFinite(v) && v > 0 ? v : undefined,
            t: match?.time ?? String(p.time), note,
          });
          setActiveNote(note ?? null);
        } else {
          setHover(null);
          setActiveNote(null);
        }
      };
      chart.subscribeCrosshairMove(onMove);
      (chart as unknown as { __onMove?: typeof onMove }).__onMove = onMove;
    } else {
      // Keep formatters current without remount when only candles change
      chartRef.current.applyOptions({
        localization: {
          timeFormatter: (time: Time) => formatChartTime(
            typeof time === "number" ? time : String(time),
            intraday,
            displayTz,
            shiftDisplay,
          ),
          locale: "en-US",
        },
        timeScale: {
          timeVisible: intraday,
          tickMarkFormatter: (time: Time, tickMarkType: number) => formatTickMark(
            time as string | number | { year: number; month: number; day: number },
            tickMarkType,
            displayTz,
            shiftDisplay,
          ),
        },
      });
    }

    const series = candleSeriesRef.current;
    const volume = volumeSeriesRef.current;
    const chart = chartRef.current;
    if (!series || !volume || !chart) return;

    series.setData(toCandleData(candles, displayTz));
    volume.setData(toVolumeData(candles, displayTz));

    if (markers.length) {
      // Keep every same-day mark (news + strategy + options) — don't let
      // the last overlay wipe earlier ones.
      const byDay = new Map<string, ChartMarker[]>();
      for (const m of markers) {
        const day = chartDayKey(m.time);
        const list = byDay.get(day) ?? [];
        list.push(m);
        byDay.set(day, list);
      }
      const lastCandleByDay = new Map<string, Candle>();
      for (const c of candles) {
        lastCandleByDay.set(chartDayKey(c.time), c);
      }
      const mk: SeriesMarker<Time>[] = [];
      for (const [day, list] of byDay) {
        const c = lastCandleByDay.get(day);
        if (!c) continue;
        // One LWC marker slot per day: combine letters, prefer first color/shape.
        const letters = list.map((m) => m.text || "?").join("");
        const primary = list[list.length - 1];
        mk.push({
          time: toChartTime(c.time, displayTz),
          position: primary.position ?? (c.close >= c.open ? "aboveBar" : "belowBar"),
          color: primary.color ?? (c.close >= c.open ? "#2eea8b" : "#ff5470"),
          shape: primary.shape ?? "circle",
          text: letters.slice(0, 6),
        });
      }
      series.setMarkers(mk);
    } else {
      series.setMarkers([]);
    }

    if (!fittedOnceRef.current) {
      chart.timeScale().fitContent();
      fittedOnceRef.current = true;
    }
  }, [candles, markers, intraday, displayTz, shiftDisplay]);

  useEffect(() => {
    const chart = chartRef.current;
    const candleSeries = candleSeriesRef.current;
    if (!chart || !candleSeries) return;
    const existing = overlaySeriesRef.current;
    const priceLines = priceLinesRef.current;
    const nextIds = new Set(overlays.map((o) => o.id));

    for (const [id, s] of existing) {
      if (!nextIds.has(id)) {
        try { chart.removeSeries(s); } catch { /* skip */ }
        existing.delete(id);
      }
    }
    for (const [id, line] of priceLines) {
      if (!nextIds.has(id)) {
        try { candleSeries.removePriceLine(line); } catch { /* skip */ }
        priceLines.delete(id);
      }
    }

    for (const ov of overlays) {
      const level = ov.priceLevel != null && Number.isFinite(ov.priceLevel)
        ? ov.priceLevel
        : null;

      if (level != null) {
        const oldSeries = existing.get(ov.id);
        if (oldSeries) {
          try { chart.removeSeries(oldSeries); } catch { /* skip */ }
          existing.delete(ov.id);
        }
        const prev = priceLines.get(ov.id);
        if (prev) {
          try { candleSeries.removePriceLine(prev); } catch { /* skip */ }
        }
        const line = candleSeries.createPriceLine({
          price: level,
          color: ov.color,
          lineWidth: 1,
          lineStyle: lineStyleOf(ov.style),
          axisLabelVisible: true,
          title: ov.label ?? ov.id,
        });
        priceLines.set(ov.id, line);
        continue;
      }

      const prevLine = priceLines.get(ov.id);
      if (prevLine) {
        try { candleSeries.removePriceLine(prevLine); } catch { /* skip */ }
        priceLines.delete(ov.id);
      }

      let s = existing.get(ov.id);
      if (!s) {
        s = chart.addLineSeries({
          color: ov.color,
          lineWidth: 1,
          lineStyle: lineStyleOf(ov.style),
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
          title: ov.label ?? ov.id,
        });
        existing.set(ov.id, s);
      } else {
        s.applyOptions({
          color: ov.color,
          lineStyle: lineStyleOf(ov.style),
          title: ov.label ?? ov.id,
        });
      }
      const lastByDay = new Map<string, string>();
      for (const c of candles) {
        lastByDay.set(chartDayKey(c.time), c.time);
      }
      const data = (ov.points ?? [])
        .map((p) => {
          if (!Number.isFinite(p.value)) return null;
          const aligned = lastByDay.get(chartDayKey(p.time));
          if (!aligned) return null;
          return { time: toChartTime(aligned, displayTz), value: p.value };
        })
        .filter((x): x is { time: Time; value: number } => x != null);
      s.setData(data);
    }
  }, [overlays, candles, displayTz]);

  useEffect(() => {
    setHover((h) => {
      if (!h) return h;
      const match = findCandle(candles, h.t, displayTz)
        || candles.find((c) => c.time === h.t);
      if (!match) return h;
      const v = Number(match.volume) || 0;
      if (
        match.open === h.o && match.high === h.h && match.low === h.l
        && match.close === h.c && (h.v ?? 0) === v
      ) {
        return h;
      }
      return {
        ...h,
        o: match.open,
        h: match.high,
        l: match.low,
        c: match.close,
        v: v > 0 ? v : undefined,
        t: match.time,
      };
    });
  }, [candles, displayTz]);

  useEffect(() => () => {
    const chart = chartRef.current;
    if (chart) {
      const onMove = (chart as unknown as { __onMove?: (p: MouseEventParams) => void }).__onMove;
      if (onMove) chart.unsubscribeCrosshairMove(onMove);
      chart.remove();
    }
    chartRef.current = null;
    candleSeriesRef.current = null;
    volumeSeriesRef.current = null;
    overlaySeriesRef.current = new Map();
    priceLinesRef.current = new Map();
    fittedOnceRef.current = false;
  }, []);

  const last = candles[candles.length - 1];
  let shown: Hover | null = hover;
  if (!shown && last) {
    shown = {
      o: last.open, h: last.high, l: last.low, c: last.close,
      v: last.volume, t: last.time,
    };
  } else if (shown) {
    const match = findCandle(candles, shown.t, displayTz)
      || candles.find((c) => c.time === shown!.t);
    if (match && Number(match.volume) > 0) {
      shown = { ...shown, v: match.volume, t: match.time };
    }
  }

  return (
    <div className={live ? "chart-live" : undefined}>
      <div className="legend num" style={{ padding: "0 18px 8px", minHeight: 22 }}>
        {shown ? (
          <>
            <span>{formatChartTime(shown.t, intraday, displayTz, false)}</span>
            <span>Open <b>{shown.o.toFixed(2)}</b></span>
            <span>High <b>{shown.h.toFixed(2)}</b></span>
            <span>Low <b>{shown.l.toFixed(2)}</b></span>
            <span>Close <b style={{ color: shown.c >= shown.o ? "var(--up)" : "var(--down)" }}>{shown.c.toFixed(2)}</b></span>
            <span>Volume <b>{formatVolume(Number(shown.v) || 0)}</b></span>
            {shown.note && (
              <span className="dim" title={shown.note} style={{ maxWidth: 280, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {shown.note.startsWith("[") ? shown.note.split(" — ")[0] : "Event"}
              </span>
            )}
            {live && <span className="live-dot" title="Price updating live on the latest candle">LIVE</span>}
          </>
        ) : (
          <span className="dim">—</span>
        )}
      </div>
      <div
        style={{
          padding: "0 18px 8px",
          fontSize: 12.5,
          lineHeight: 1.35,
          minHeight: 48,
          maxHeight: 48,
          overflow: "hidden",
          color: activeNote ? "var(--text-2, #c5d0e0)" : "var(--muted, #6b7c93)",
        }}
        title={activeNote ?? "Big volume day · Above-average volume · Large price move · hover a dot for details"}
      >
        {activeNote || "Hover a marked day — big volume · above average · large move (colors = up vs down day)"}
      </div>
      <div id="chart" className="chart-pane" ref={containerRef} />
    </div>
  );
}
