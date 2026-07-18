/**
 * Per-page spotlight tour. Fail-open: if tour-state fetch fails, the page
 * still works and no tour runs.
 *
 * Library: react-joyride — standard spotlight pattern; ~small dep tree vs
 * hand-rolling overlays. Skip is available on every step.
 *
 * Mobile: tooltip is capped to the viewport and Floater is told to flip /
 * shift inside the screen so steps are not clipped off the left/right edge
 * (Phase 1 audit at 375px).
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import Joyride, { ACTIONS, EVENTS, STATUS, type CallBackProps } from "react-joyride";
import { getTours, markTourSeen, type TourPageId } from "./api";
import { TOUR_STEPS } from "./tourSteps";

function useNarrowTour(): boolean {
  const [narrow, setNarrow] = useState(() =>
    typeof window !== "undefined" ? window.matchMedia("(max-width: 520px)").matches : false,
  );
  useEffect(() => {
    const mq = window.matchMedia("(max-width: 520px)");
    const onChange = () => setNarrow(mq.matches);
    onChange();
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return narrow;
}

export default function PageTour({ pageId }: { pageId: TourPageId }) {
  const [run, setRun] = useState(false);
  const [ready, setReady] = useState(false);
  const narrow = useNarrowTour();

  useEffect(() => {
    let cancelled = false;
    setRun(false);
    setReady(false);
    getTours()
      .then((r) => {
        if (cancelled) return;
        const seen = r.tours_seen?.[pageId] === true;
        setReady(true);
        if (!seen) setRun(true);
      })
      .catch(() => {
        // Fail open — page stays usable; tour simply does not fire.
        if (!cancelled) setReady(true);
      });
    return () => {
      cancelled = true;
    };
  }, [pageId]);

  const persistSeen = useCallback(() => {
    void markTourSeen(pageId).catch(() => {
      /* fail open — local skip still stops the tour */
    });
  }, [pageId]);

  const onCallback = useCallback(
    (data: CallBackProps) => {
      const { status, action, type } = data;
      if (status === STATUS.FINISHED || status === STATUS.SKIPPED) {
        setRun(false);
        persistSeen();
        return;
      }
      // Skip on any step (joyride fires skip action)
      if (action === ACTIONS.SKIP) {
        setRun(false);
        persistSeen();
        return;
      }
      if (type === EVENTS.TOUR_END) {
        setRun(false);
      }
    },
    [persistSeen],
  );

  const tourStyles = useMemo(
    () => ({
      options: {
        zIndex: 10000,
        primaryColor: "#5b8def",
        textColor: "#e8eef8",
        backgroundColor: "#1a2230",
        arrowColor: "#1a2230",
        overlayColor: "rgba(0, 0, 0, 0.55)",
        width: narrow ? Math.min(360, (typeof window !== "undefined" ? window.innerWidth : 360) - 28) : 380,
      },
      buttonSkip: {
        color: "#9aa8bc",
        fontSize: 13,
        minHeight: 44,
        padding: "10px 12px",
      },
      buttonNext: {
        minHeight: 44,
        padding: "10px 14px",
        fontSize: 13,
        whiteSpace: "normal" as const,
        lineHeight: 1.3,
      },
      buttonBack: {
        minHeight: 44,
        padding: "10px 12px",
      },
      buttonClose: {
        minHeight: 44,
        minWidth: 44,
        padding: 8,
      },
      tooltip: {
        borderRadius: 10,
        fontSize: 13.5,
        maxWidth: "calc(100vw - 28px)",
        padding: narrow ? "14px 14px 12px" : undefined,
      },
      tooltipContainer: {
        textAlign: "left" as const,
      },
      tooltipContent: {
        padding: "4px 0 8px",
      },
      tooltipFooter: {
        flexWrap: "wrap" as const,
        gap: 8,
        alignItems: "center",
      },
    }),
    [narrow],
  );

  if (!ready) return null;

  return (
    <Joyride
      steps={TOUR_STEPS[pageId]}
      run={run}
      continuous
      showSkipButton
      showProgress
      scrollToFirstStep
      disableOverlayClose
      // Keep the spotlight in view on phones without fighting nested scrollers.
      disableScrolling={false}
      callback={onCallback}
      locale={{
        back: "Back",
        close: "Close",
        last: "Done",
        next: "Next",
        open: "Open",
        skip: "Skip",
      }}
      floaterProps={{
        disableAnimation: narrow,
        hideArrow: false,
        // Stay inside the viewport — Phase 1 showed tooltips clipped left/right.
        placement: "auto",
        offset: 12,
        styles: {
          floater: {
            maxWidth: "calc(100vw - 24px)",
            filter: "none",
          },
        },
      }}
      styles={tourStyles}
    />
  );
}
