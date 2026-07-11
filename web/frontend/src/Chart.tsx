import { useEffect, useRef, useState } from "react";
import {
  createChart,
  ColorType,
  CrosshairMode,
  type IChartApi,
  type CandlestickData,
  type HistogramData,
  type MouseEventParams,
} from "lightweight-charts";
import type { Candle } from "./api";

interface Hover {
  o: number; h: number; l: number; c: number; t: string;
}

export default function Chart({ candles }: { candles: Candle[] }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [hover, setHover] = useState<Hover | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
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
      timeScale: { borderColor: "#1b2438" },
      autoSize: true,
    });

    const series = chart.addCandlestickSeries({
      upColor: "#2eea8b",
      downColor: "#ff5470",
      borderVisible: false,
      wickUpColor: "#2eea8b",
      wickDownColor: "#ff5470",
    });
    series.setData(candles as CandlestickData[]);

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
          time: c.time,
          value: c.volume,
          color: c.close >= c.open
            ? "rgba(46,234,139,0.28)"
            : "rgba(255,84,112,0.28)",
        }),
      ),
    );

    const onMove = (p: MouseEventParams) => {
      const d = p.seriesData.get(series) as CandlestickData | undefined;
      if (d && p.time) {
        setHover({ o: d.open, h: d.high, l: d.low, c: d.close, t: String(p.time) });
      } else setHover(null);
    };
    chart.subscribeCrosshairMove(onMove);
    chart.timeScale().fitContent();
    chartRef.current = chart;
    return () => {
      chart.unsubscribeCrosshairMove(onMove);
      chart.remove();
    };
  }, [candles]);

  const last = candles[candles.length - 1];
  const shown = hover ?? (last
    ? { o: last.open, h: last.high, l: last.low, c: last.close, t: last.time }
    : null);

  return (
    <div>
      {shown && (
        <div className="legend num" style={{ padding: "0 18px 8px" }}>
          <span>{shown.t}</span>
          <span>O <b>{shown.o.toFixed(2)}</b></span>
          <span>H <b>{shown.h.toFixed(2)}</b></span>
          <span>L <b>{shown.l.toFixed(2)}</b></span>
          <span>C <b style={{ color: shown.c >= shown.o ? "var(--up)" : "var(--down)" }}>{shown.c.toFixed(2)}</b></span>
        </div>
      )}
      <div id="chart" ref={containerRef} />
    </div>
  );
}
