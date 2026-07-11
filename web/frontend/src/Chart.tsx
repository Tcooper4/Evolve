import { useEffect, useRef } from "react";
import {
  createChart,
  ColorType,
  type IChartApi,
  type CandlestickData,
} from "lightweight-charts";
import type { Candle } from "./api";

export default function Chart({ candles }: { candles: Candle[] }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#0d1420" },
        textColor: "#8494ad",
      },
      grid: {
        vertLines: { color: "#1c2940" },
        horzLines: { color: "#1c2940" },
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
    series.setData(candles as CandlestickData[]);
    chart.timeScale().fitContent();
    chartRef.current = chart;
    return () => chart.remove();
  }, [candles]);

  return <div id="chart" ref={containerRef} />;
}
