import { useState, useEffect, useRef } from 'react';
import type { SolveProgress } from '../types/api';

export function useSSE(url: string | null) {
  const [data, setData] = useState<SolveProgress | null>(null);
  const [done, setDone] = useState(false);
  const sourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!url) return;

    setData(null);
    setDone(false);

    const source = new EventSource(url);
    sourceRef.current = source;

    source.onmessage = (event) => {
      const progress: SolveProgress = JSON.parse(event.data);
      setData(progress);
      if (progress.stage === 'complete' || progress.stage === 'failed' || progress.stage === 'verification_failed') {
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
