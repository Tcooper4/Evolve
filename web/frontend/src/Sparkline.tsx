// Chart sparkline with optional compare series, position shading, and hover.
import { useState } from "react";
import type { MouseEvent } from "react";

export default function Sparkline({
  values,
  compare,
  positions,
  times,
  width = 72,
  height = 30,
  baseline = "start",
  color,
  compareColor = "var(--accent)",
  strokeWidth = 1.6,
  compareStrokeWidth = 1.8,
  compareDash = "5 4",
}: {
  values: Array<number | null | undefined>;
  compare?: Array<number | null | undefined>;
  /** 0/1 (or abs position) — shaded when in market */
  positions?: Array<number | null | undefined>;
  times?: Array<string | null | undefined>;
  width?: number;
  height?: number;
  baseline?: "start" | "zero" | number | false;
  color?: string;
  compareColor?: string;
  strokeWidth?: number;
  compareStrokeWidth?: number;
  compareDash?: string | false;
}) {
  const [hover, setHover] = useState<{
    i: number; x: number; y: number;
  } | null>(null);

  const finite = (v: unknown): v is number =>
    typeof v === "number" && Number.isFinite(v);

  const nums = values.filter(finite);
  if (nums.length < 2) return <svg width={width} height={height} />;

  const compareNums = (compare ?? []).filter(finite);
  const all = compareNums.length ? [...nums, ...compareNums] : nums;
  const min = Math.min(...all);
  const max = Math.max(...all);
  const span = max - min || 1;
  const padTop = 4;
  const padBot = times && times.length ? 14 : 4;
  const n = Math.max(values.length, compare?.length ?? 0, 2);
  const plotH = height - padTop - padBot;
  const yOf = (v: number) => padTop + plotH - ((v - min) / span) * plotH;
  const step = width / (n - 1);
  const xOf = (i: number) => i * step;

  const segmentsOf = (series: Array<number | null | undefined>) => {
    const segs: string[] = [];
    let cur: string[] = [];
    const flush = () => {
      if (cur.length >= 2) segs.push(cur.join(" "));
      cur = [];
    };
    for (let i = 0; i < n; i++) {
      const v = series[i];
      if (finite(v)) cur.push(`${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`);
      else flush();
    }
    flush();
    return segs;
  };

  const first = nums[0];
  const last = nums[nums.length - 1];
  const up = last >= first;
  const stroke = color ?? (up ? "var(--up)" : "var(--down)");

  let baseY: number | null = null;
  if (baseline === "start") baseY = yOf(first);
  else if (baseline === "zero" && min <= 0 && max >= 0) baseY = yOf(0);
  else if (typeof baseline === "number" && Number.isFinite(baseline)) {
    baseY = yOf(Math.min(max, Math.max(min, baseline)));
  }

  const valueSegs = segmentsOf(values);
  const compareSegs = compare ? segmentsOf(compare) : [];

  // Position bands as rects spanning contiguous in-market runs
  const bands: { x: number; w: number }[] = [];
  if (positions && positions.length) {
    let start: number | null = null;
    for (let i = 0; i < n; i++) {
      const on = Math.abs(Number(positions[i] ?? 0)) > 1e-9;
      if (on && start == null) start = i;
      if ((!on || i === n - 1) && start != null) {
        const end = on && i === n - 1 ? i : i - 1;
        if (end >= start) {
          const x0 = xOf(start);
          const x1 = xOf(Math.max(start, end));
          bands.push({ x: x0, w: Math.max(2, x1 - x0) });
        }
        start = null;
      }
    }
  }

  const tickIdx = n > 2 ? [0, Math.floor((n - 1) / 2), n - 1] : [0, n - 1];

  function onMove(e: MouseEvent<SVGSVGElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const i = Math.max(0, Math.min(n - 1, Math.round(x / step)));
    const v = values[i];
    if (!finite(v)) {
      setHover(null);
      return;
    }
    setHover({ i, x: xOf(i), y: yOf(v) });
  }

  const hi = hover?.i ?? -1;
  const tipParts: string[] = [];
  if (hi >= 0) {
    if (times?.[hi]) tipParts.push(String(times[hi]));
    if (finite(values[hi])) tipParts.push(`A ${Number(values[hi]).toFixed(3)}`);
    if (compare && finite(compare[hi])) tipParts.push(`B ${Number(compare[hi]).toFixed(3)}`);
    if (positions && Math.abs(Number(positions[hi] ?? 0)) > 1e-9) tipParts.push("in");
  }

  return (
    <div style={{ position: "relative", width, maxWidth: "100%" }}>
      <svg
        width={width}
        height={height}
        className="wl-spark"
        overflow="visible"
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
        style={{ display: "block", cursor: "crosshair" }}
      >
        {bands.map((b, i) => (
          <rect
            key={`b-${i}`}
            x={b.x}
            y={padTop}
            width={b.w}
            height={plotH}
            fill="var(--accent)"
            opacity="0.10"
          />
        ))}
        {baseY != null && (
          <line
            x1={0} y1={baseY} x2={width} y2={baseY}
            stroke="var(--text-3)" strokeWidth="1" strokeDasharray="3 3" opacity="0.55"
          />
        )}
        {valueSegs.map((pts, i) => (
          <polyline
            key={`v-${i}`} points={pts} fill="none" stroke={stroke}
            strokeWidth={strokeWidth} strokeLinejoin="round" strokeLinecap="round" opacity="1"
          />
        ))}
        {compareSegs.map((pts, i) => (
          <polyline
            key={`c-${i}`} points={pts} fill="none" stroke={compareColor}
            strokeWidth={compareStrokeWidth}
            strokeDasharray={compareDash === false ? undefined : compareDash}
            strokeLinejoin="round" strokeLinecap="butt" opacity="0.95"
          />
        ))}
        {times && tickIdx.map((i) => (
          <text
            key={`t-${i}`}
            x={xOf(i)}
            y={height - 2}
            textAnchor={i === 0 ? "start" : i === n - 1 ? "end" : "middle"}
            fill="var(--text-3)"
            fontSize="9"
          >
            {String(times[i] ?? "").slice(0, 10)}
          </text>
        ))}
        {hover && (
          <>
            <line x1={hover.x} y1={padTop} x2={hover.x} y2={padTop + plotH}
              stroke="var(--text-3)" strokeWidth="1" strokeDasharray="2 2" opacity="0.7" />
            <circle cx={hover.x} cy={hover.y} r="3.5" fill={stroke} />
          </>
        )}
      </svg>
      {hover && tipParts.length > 0 && (
        <div
          className="dim"
          style={{
            position: "absolute",
            left: Math.min(width - 120, Math.max(0, hover.x - 40)),
            top: 0,
            fontSize: 11,
            background: "var(--surface-2)",
            border: "1px solid var(--border)",
            borderRadius: 6,
            padding: "2px 6px",
            pointerEvents: "none",
            whiteSpace: "nowrap",
          }}
        >
          {tipParts.join(" · ")}
        </div>
      )}
    </div>
  );
}
