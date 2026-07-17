/**
 * Per-page spotlight tour. Fail-open: if tour-state fetch fails, the page
 * still works and no tour runs.
 *
 * Library: react-joyride — standard spotlight pattern; ~small dep tree vs
 * hand-rolling overlays. Skip is available on every step.
 */

import { useCallback, useEffect, useState } from "react";
import Joyride, { ACTIONS, EVENTS, STATUS, type CallBackProps } from "react-joyride";
import { getTours, markTourSeen, type TourPageId } from "./api";
import { TOUR_STEPS } from "./tourSteps";

export default function PageTour({ pageId }: { pageId: TourPageId }) {
  const [run, setRun] = useState(false);
  const [ready, setReady] = useState(false);

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
      callback={onCallback}
      locale={{
        back: "Back",
        close: "Close",
        last: "Done",
        next: "Next",
        open: "Open",
        skip: "Skip tour",
      }}
      styles={{
        options: {
          zIndex: 10000,
          primaryColor: "#5b8def",
          textColor: "#e8eef8",
          backgroundColor: "#1a2230",
          arrowColor: "#1a2230",
          overlayColor: "rgba(0, 0, 0, 0.55)",
        },
        buttonSkip: {
          color: "#9aa8bc",
          fontSize: 13,
        },
        tooltip: {
          borderRadius: 10,
          fontSize: 13.5,
        },
      }}
    />
  );
}
