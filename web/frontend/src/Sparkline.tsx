// Tiny inline SVG sparkline - the "mini-graph in the card" pattern.
export default function Sparkline({
  values,
  width = 72,
  height = 30,
}: {
  values: number[];
  width?: number;
  height?: number;
}) {
  if (values.length < 2) return <svg width={width} height={height} />;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const step = width / (values.length - 1);
  const pts = values.map(
    (v, i) => `${(i * step).toFixed(1)},${(height - 2 - ((v - min) / span) * (height - 4)).toFixed(1)}`,
  );
  const up = values[values.length - 1] >= values[0];
  const color = up ? "var(--up)" : "var(--down)";
  return (
    <svg width={width} height={height} className="wl-spark">
      <polyline
        points={pts.join(" ")}
        fill="none"
        stroke={color}
        strokeWidth="1.6"
        strokeLinejoin="round"
        strokeLinecap="round"
        opacity="0.9"
      />
    </svg>
  );
}
