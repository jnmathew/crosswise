import { useState, useEffect, useRef } from 'react';
import type { SolveProgress } from '../types/api';

export function useSSE(url: string | null) {
  const [data, setData] = useState<SolveProgress | null>(null);
  const [done, setDone] = useState(false);
  const sourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!url) return;

    setData(null);   // eslint-disable-line react-hooks/set-state-in-effect -- reset state on URL change before subscribing
    setDone(false);

    const source = new EventSource(url);
    sourceRef.current = source;

    source.onmessage = (event) => {
      const progress: SolveProgress = JSON.parse(event.data);
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
