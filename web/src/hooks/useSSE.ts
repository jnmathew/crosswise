import { useState, useEffect, useRef } from 'react';
import type { LiveEvent, SolveProgress } from '../types/api';

/**
 * Subscribe to a solve's progress stream.
 *
 * Live solve events (stage "live") go to `onLive` instead of replacing `data`,
 * so they never disturb the progress banner or the stage-driven effects.
 */
export function useSSE(url: string | null, onLive?: (event: LiveEvent) => void) {
  const [data, setData] = useState<SolveProgress | null>(null);
  const [done, setDone] = useState(false);
  const sourceRef = useRef<EventSource | null>(null);
  const onLiveRef = useRef(onLive);

  useEffect(() => {
    onLiveRef.current = onLive;
  }, [onLive]);

  useEffect(() => {
    if (!url) return;

    setData(null);   // eslint-disable-line react-hooks/set-state-in-effect -- reset state on URL change before subscribing
    setDone(false);

    const source = new EventSource(url);
    sourceRef.current = source;

    source.onmessage = (event) => {
      const progress: SolveProgress = JSON.parse(event.data);
      if (progress.stage === 'live') {
        if (progress.live) onLiveRef.current?.(progress.live);
        return;
      }
      setData(progress);
      if (['complete', 'failed', 'verification_failed', 'cancelled'].includes(progress.stage)) {
        setDone(true);
        source.close();
      }
    };

    source.onerror = () => {
      source.close();
    };

    return () => {
      source.close();
      sourceRef.current = null;
    };
  }, [url]);

  return { data, done };
}
