import { useEffect, useRef, useState } from "react";
import {
  createChart,
  ColorType,
  CrosshairMode,
  type IChartApi,
  type ISeriesApi,
  type CandlestickData,
  type HistogramData,
  type MouseEventParams,
  type SeriesMarker,
  type Time,
} from "lightweight-charts";
import type { Candle } from "./api";
import { formatChartTime, resolveChartTimeZone } from "./chartTime";

export interface ChartMarker {
  time: string;
  position?: "aboveBar" | "belowBar";
  color?: string;
  shape?: "circle" | "square" | "arrowUp" | "arrowDown";
  text?: string;
  title?: string;
}

interface Hover {
  o: number; h: number; l: number; c: number; v?: number; t: string;
  note?: string;
}

function toChartTime(t: string): Time {
  if (t.includes("T") || t.length > 10) {
    const ms = Date.parse(t);
    if (Number.isFinite(ms)) return Math.floor(ms / 1000) as Time;
  }
  return t.slice(0, 10) as Time;
}

function toCandleData(candles: Candle[]): CandlestickData[] {
  return candles.map((c) => ({
    time: toChartTime(c.time),
    open: c.open,
    high: c.high,
    low: c.low,
    close: c.close,
  })) as CandlestickData[];
}

function toVolumeData(candles: Candle[]): HistogramData[] {
  return candles.map((c) => ({
    time: toChartTime(c.time),
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

function timesEqual(chartTime: string | number, candleTime: string): boolean {
  const a = String(chartTime);
  const b = String(toChartTime(candleTime));
  if (a === b) return true;
  // Daily bars: chart may be YYYY-MM-DD
  if (a.length >= 10 && candleTime.slice(0, 10) === a.slice(0, 10)) return true;
  return false;
}

function findCandle(
  candles: Candle[],
  chartTime: string | number,
): Candle | undefined {
  const t = String(chartTime);
  return candles.find((c) => timesEqual(t, c.time));
}

export default function Chart({
  candles,
  markers = [],
  live = false,
  timeZone,
}: {
  candles: Candle[];
  markers?: ChartMarker[];
  /** Soft CSS cue when the last bar is being updated live */
  live?: boolean;
  /** IANA timezone id, or omit / "local" for browser local */
  timeZone?: string;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const candlesRef = useRef(candles);
  const markersRef = useRef(markers);
  const fittedOnceRef = useRef(false);
  const [hover, setHover] = useState<Hover | null>(null);
  const [activeNote, setActiveNote] = useState<string | null>(null);
  const intraday = candles.some((c) => c.time.includes("T") || c.time.length > 10);
  const tz = resolveChartTimeZone(timeZone);

  candlesRef.current = candles;
  markersRef.current = markers;

  // Create chart once; update data in place so live ticks / soft polls don't remount
  useEffect(() => {
    if (!containerRef.current || candles.length === 0) return;

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
            true,
            tz,
          ),
        },
      });

      const series = chart.addCandlestickSeries({
        upColor: "#2eea8b",
        downColor: "#ff5470",
        borderVisible: false,
        wickUpColor: "#2eea8b",
        wickDownColor: "#ff5470",
      });
      // Leave room at the bottom so the volume pane is visible
      series.priceScale().applyOptions({
        scaleMargins: { top: 0.05, bottom: 0.22 },
      });

      const volume = chart.addHistogramSeries({
        priceFormat: { type: "volume" },
        priceScaleId: "vol",
        lastValueVisible: false,
        priceLineVisible: false,
      });
      // Hide the vol axis (avoids a stray "0" in the bottom-right)
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
        const volSeries = volumeSeriesRef.current;
        if (!candleSeries) return;
        const d = p.seriesData.get(candleSeries) as CandlestickData | undefined;
        if (d && p.time != null) {
          const t = String(p.time);
          const day = t.slice(0, 10);
          const noteByDay = new Map(
            markersRef.current.map((m) => [m.time.slice(0, 10), m.title || m.text || ""]),
          );
          const note = noteByDay.get(day) || undefined;
          const volPoint = volSeries
            ? (p.seriesData.get(volSeries) as HistogramData | undefined)
            : undefined;
          const match = findCandle(candlesRef.current, p.time as string | number);
          const v = Number(volPoint?.value ?? match?.volume ?? 0);
          setHover({
            o: d.open, h: d.high, l: d.low, c: d.close,
            v: Number.isFinite(v) && v > 0 ? v : undefined,
            t, note,
          });
          setActiveNote(note ?? null);
        } else {
          setHover(null);
          setActiveNote(null);
        }
      };
      chart.subscribeCrosshairMove(onMove);
      (chart as unknown as { __onMove?: typeof onMove }).__onMove = onMove;
    }

    const chart = chartRef.current;
    const series = candleSeriesRef.current;
    const volume = volumeSeriesRef.current;
    if (!chart || !series || !volume) return;

    chart.timeScale().applyOptions({
      timeVisible: intraday,
      barSpacing: Math.max(3, Math.min(8, Math.floor(720 / Math.max(candles.length, 1)))),
    });
    chart.applyOptions({
      localization: {
        timeFormatter: (time: Time) => {
          const raw = typeof time === "number" || typeof time === "string"
            ? time
            : String(time);
          return formatChartTime(raw as string | number, intraday, tz);
        },
      },
    });
    series.setData(toCandleData(candles));
    volume.setData(toVolumeData(candles));

    if (markers.length) {
      const byTime = new Map(markers.map((m) => [m.time.slice(0, 10), m]));
      const mk: SeriesMarker<Time>[] = [];
      for (const c of candles) {
        const key = c.time.slice(0, 10);
        const m = byTime.get(key);
        if (!m) continue;
        mk.push({
          time: toChartTime(c.time),
          position: m.position ?? (c.close >= c.open ? "aboveBar" : "belowBar"),
          color: m.color ?? (c.close >= c.open ? "#2eea8b" : "#ff5470"),
          shape: m.shape ?? (c.close >= c.open ? "arrowUp" : "arrowDown"),
          text: m.text ?? "N",
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
  }, [candles, markers, intraday, tz]);

  // Keep legend volume in sync when soft-poll / live updates refresh candle data
  useEffect(() => {
    setHover((h) => {
      if (!h) return h;
      const match = findCandle(candles, h.t);
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
      };
    });
  }, [candles]);

  // Tear down only on unmount (or when series empties → remount next load)
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
    // Prefer latest volume from candle array (soft poll) over stale hover
    const match = findCandle(candles, shown.t);
    if (match && Number(match.volume) > 0) {
      shown = { ...shown, v: match.volume };
    }
  }

  return (
    <div className={live ? "chart-live" : undefined}>
      {shown && (
        <div className="legend num" style={{ padding: "0 18px 8px" }}>
          <span>{formatChartTime(shown.t, intraday, tz)}</span>
          <span>O <b>{shown.o.toFixed(2)}</b></span>
          <span>H <b>{shown.h.toFixed(2)}</b></span>
          <span>L <b>{shown.l.toFixed(2)}</b></span>
          <span>C <b style={{ color: shown.c >= shown.o ? "var(--up)" : "var(--down)" }}>{shown.c.toFixed(2)}</b></span>
          <span>V <b>{formatVolume(Number(shown.v) || 0)}</b></span>
          {live && <span className="live-dot" title="Live last bar">LIVE</span>}
        </div>
      )}
      {activeNote && (
        <div className="dim" style={{ padding: "0 18px 8px", fontSize: 12.5, lineHeight: 1.4 }}>
          {activeNote}
        </div>
      )}
      <div id="chart" ref={containerRef} />
    </div>
  );
}
