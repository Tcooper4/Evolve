import { useEffect, useState } from "react";
import { getPulse, refreshPulseSymbol } from "./api";

export interface TapeItem {
  symbol: string;
  label: string;
  price: number | null;
  change_pct: number | null;
}

/**
 * Seamless CSS marquee tape. Quotes refresh in place — no DOM recycle/reorder,
 * which was causing the brief glitch when a pulse item rolled off-screen.
 */
export default function MarketTicker({
  onSelect,
}: {
  onSelect?: (symbol: string) => void;
}) {
  const [items, setItems] = useState<TapeItem[]>([]);

  useEffect(() => {
    getPulse()
      .then((p) => setItems(p.items ?? []))
      .catch(() => {});
  }, []);

  // Quiet background refresh — update prices by symbol key only
  useEffect(() => {
    if (items.length === 0) return;
    let cancelled = false;
    let idx = 0;
    const tick = async () => {
      if (cancelled || items.length === 0) return;
      const target = items[idx % items.length];
      idx += 1;
      try {
        const fresh = await refreshPulseSymbol(target.symbol);
        if (cancelled) return;
        setItems((prev) =>
          prev.map((it) =>
            it.symbol === target.symbol
              ? {
                  ...it,
                  price: fresh.price ?? it.price,
                  change_pct: fresh.change_pct ?? it.change_pct,
                }
              : it,
          ),
        );
      } catch {
        /* keep last quote */
      }
    };
    const id = window.setInterval(tick, 4500);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [items.length]); // eslint-disable-line react-hooks/exhaustive-deps

  if (items.length === 0) return null;

  const renderRow = (keyPrefix: string) =>
    items.map((p) => {
      const pup = (p.change_pct ?? 0) >= 0;
      const navSym = p.symbol.replace("^", "").replace("=F", "").replace("-USD", "");
      return (
        <button
          key={`${keyPrefix}-${p.symbol}`}
          type="button"
          className="ticker-item"
          title="Click to analyze"
          onClick={() => onSelect?.(navSym || p.symbol)}
        >
          <span className="sym">{p.label}</span>
          <span className="px num">
            {p.price != null
              ? Math.abs(p.price) >= 1000
                ? p.price.toFixed(1)
                : p.price.toFixed(2)
              : "—"}
          </span>
          {p.change_pct != null && (
            <span className={`delta num ${pup ? "up" : "down"}`}>
              {pup ? "▲" : "▼"} {Math.abs(p.change_pct).toFixed(2)}%
            </span>
          )}
        </button>
      );
    });

  return (
    <div className="ticker-tape" aria-label="Live market tape">
      <div className="ticker-track ticker-marquee">
        {renderRow("a")}
        {renderRow("b")}
      </div>
    </div>
  );
}
