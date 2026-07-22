/** Survivorship / current-membership bias caveat for index universes. */

import { useState } from "react";

const STRONGER = /russell|3000|1000|small/i;

export default function UniverseBiasNote({
  universeId,
  context = "scan",
}: {
  universeId?: string;
  /** scan = index screen; backtest = single-name historical run */
  context?: "scan" | "backtest";
}) {
  const [hidden, setHidden] = useState(false);
  if (hidden) return null;

  const id = universeId || "";
  const stronger = STRONGER.test(id);
  const body =
    context === "backtest"
      ? "Backtests on names that still trade today can look better than live results would have — failed companies are usually missing from the sample. Treat returns as directional, not guaranteed."
      : stronger
        ? "Results use today's index membership. For faster-turnover lists, that bias is typically larger — past research finds returns can look several percent per year too good and risk scores about 10% too optimistic. This screen has not measured Evolve's own gap."
        : "Results use today's index membership, not who was in the list back then. That usually makes historical screens look somewhat better than they would have been live. Direction of the bias is well documented; Evolve has not measured its own size here.";

  return (
    <div
      className="dim universe-bias-note"
      style={{
        marginTop: 10,
        fontSize: 12.5,
        lineHeight: 1.45,
        display: "flex",
        gap: 10,
        alignItems: "flex-start",
        justifyContent: "space-between",
      }}
    >
      <span>{body}</span>
      <button
        type="button"
        className="ghost"
        style={{ fontSize: 11, flexShrink: 0, padding: "2px 8px" }}
        onClick={() => setHidden(true)}
        aria-label="Dismiss bias note"
      >
        Dismiss
      </button>
    </div>
  );
}
