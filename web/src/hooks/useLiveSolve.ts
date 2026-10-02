import { useCallback, useEffect, useReducer } from 'react';
import type { LiveEvent } from '../types/api';

/** How long a rejected answer stays on screen (red) before it fades out. */
const REJECT_FLASH_MS = 900;

export interface LiveSolveState {
  /** Answers the solver has accepted, by clue ID ("12-across"). */
  committed: Record<string, string>;
  /** Answers the model has written in the current reply but the solver hasn't judged yet. */
  tentative: Record<string, string>;
  /** Answers just rejected or removed, shown briefly in red. */
  rejected: Record<string, { word: string; at: number }>;
  /** Phase the latest event came from, e.g. "Pass 2" or "Verification". */
  phase: string | null;
  /** Number of events received; zero means nothing to show. */
  events: number;
}

const initialState: LiveSolveState = { committed: {}, tentative: {}, rejected: {}, phase: null, events: 0 };

type Action = { type: 'event'; event: LiveEvent; now: number } | { type: 'expire'; now: number };

function reducer(state: LiveSolveState, action: Action): LiveSolveState {
  if (action.type === 'expire') {
    const rejected = Object.fromEntries(
      Object.entries(state.rejected).filter(([, r]) => action.now - r.at < REJECT_FLASH_MS),
    );
    return Object.keys(rejected).length === Object.keys(state.rejected).length ? state : { ...state, rejected };
  }

  const { event, now } = action;
  const events = state.events + 1;
  switch (event.type) {
    case 'snapshot':
      return { ...state, committed: { ...event.answers }, tentative: {}, events };
    case 'tentative':
      return { ...state, tentative: { ...state.tentative, [event.clue]: event.word }, phase: event.phase, events };
    case 'commit': {
      const committed = { ...state.committed };
      const rejected = { ...state.rejected };
      for (const clue of event.removed) {
        if (committed[clue]) rejected[clue] = { word: committed[clue], at: now };
        delete committed[clue];
      }
      Object.assign(committed, event.answers);
      // Anything the model wrote that didn't end up in the grid was rejected
      for (const [clue, word] of Object.entries(state.tentative)) {
        if (committed[clue] !== word) rejected[clue] = { word, at: now };
      }
      return { committed, tentative: {}, rejected, phase: event.phase, events };
    }
  }
}

/** Accumulates live solve events into what the overlay should draw. */
export function useLiveSolve() {
  const [state, dispatch] = useReducer(reducer, initialState);

  const onLive = useCallback((event: LiveEvent) => {
    dispatch({ type: 'event', event, now: Date.now() });
  }, []);

  // Fade out rejected answers
  const hasRejected = Object.keys(state.rejected).length > 0;
  useEffect(() => {
    if (!hasRejected) return;
    const timer = setInterval(() => dispatch({ type: 'expire', now: Date.now() }), 150);
    return () => clearInterval(timer);
  }, [hasRejected]);

  return { live: state, onLive };
}
