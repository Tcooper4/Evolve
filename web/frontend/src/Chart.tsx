import { useEffect, useRef, useState } from "react";
import {
  createChart,
  ColorType,
  CrosshairMode,
  type IChartApi,
  type CandlestickData,
  type HistogramData,
  type MouseEventParams,
  type SeriesMarker,
  type Time,
} from "lightweight-charts";
import type { Candle } from "./api";

export interface ChartMarker {
  time: string;
  position?: "aboveBar" | "belowBar";
  color?: string;
  shape?: "circle" | "square" | "arrowUp" | "arrowDown";
  text?: string;
  title?: string;
}

interface Hover {
  o: number; h: number; l: number; c: number; t: string;
  note?: string;
}

function toChartTime(t: string): Time {
  // Daily: YYYY-MM-DD · Intraday: ISO → unix seconds
  if (t.includes("T") || t.length > 10) {
    const ms = Date.parse(t);
    if (Number.isFinite(ms)) return Math.floor(ms / 1000) as Time;
  }
  return t.slice(0, 10) as Time;
}

function formatHoverTime(t: string | number, intraday: boolean): string {
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
  return new Date(ms).toLocaleString(undefined, {
    month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
  });
}

export default function Chart({
  candles,
  markers = [],
}: {
  candles: Candle[];
  markers?: ChartMarker[];
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [hover, setHover] = useState<Hover | null>(null);
  const [activeNote, setActiveNote] = useState<string | null>(null);
  const intraday = candles.some((c) => c.time.includes("T") || c.time.length > 10);

  useEffect(() => {
    if (!containerRef.current || candles.length === 0) return;
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
        // Keep bar width similar across periods (3M hourly ≈ 6M daily density)
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
    });

    const series = chart.addCandlestickSeries({
      upColor: "#2eea8b",
      downColor: "#ff5470",
      borderVisible: false,
      wickUpColor: "#2eea8b",
      wickDownColor: "#ff5470",
    });

    const candleData = candles.map((c) => ({
      time: toChartTime(c.time),
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
    })) as CandlestickData[];
    series.setData(candleData);

    const volume = chart.addHistogramSeries({
      priceFormat: { type: "volume" },
      priceScaleId: "vol",
    });
    chart.priceScale("vol").applyOptions({
      scaleMargins: { top: 0.82, bottom: 0 },
    });
    volume.setData(
      candles.map(
        (c): HistogramData => ({
          time: toChartTime(c.time),
          value: c.volume,
          color: c.close >= c.open
            ? "rgba(46,234,139,0.28)"
            : "rgba(255,84,112,0.28)",
        }),
      ),
    );

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
      if (mk.length) series.setMarkers(mk);
    }

    const noteByDay = new Map(
      markers.map((m) => [m.time.slice(0, 10), m.title || m.text || ""]),
    );

    const onMove = (p: MouseEventParams) => {
      const d = p.seriesData.get(series) as CandlestickData | undefined;
      if (d && p.time) {
        const t = String(p.time);
        const day = t.includes("T") ? t.slice(0, 10) : t.slice(0, 10);
        const note = noteByDay.get(day) || undefined;
        setHover({ o: d.open, h: d.high, l: d.low, c: d.close, t, note });
        setActiveNote(note ?? null);
      } else {
        setHover(null);
        setActiveNote(null);
      }
    };
    chart.subscribeCrosshairMove(onMove);
    chart.timeScale().fitContent();
    chartRef.current = chart;
    return () => {
      chart.unsubscribeCrosshairMove(onMove);
      chart.remove();
    };
  }, [candles, markers, intraday]);

  const last = candles[candles.length - 1];
  const shown = hover ?? (last
    ? { o: last.open, h: last.high, l: last.low, c: last.close, t: last.time }
    : null);

  return (
    <div>
      {shown && (
        <div className="legend num" style={{ padding: "0 18px 8px" }}>
          <span>{formatHoverTime(shown.t, intraday)}</span>
          <span>O <b>{shown.o.toFixed(2)}</b></span>
          <span>H <b>{shown.h.toFixed(2)}</b></span>
          <span>L <b>{shown.l.toFixed(2)}</b></span>
          <span>C <b style={{ color: shown.c >= shown.o ? "var(--up)" : "var(--down)" }}>{shown.c.toFixed(2)}</b></span>
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
