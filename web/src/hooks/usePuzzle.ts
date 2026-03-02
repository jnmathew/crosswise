import { useState, useEffect, useCallback, useRef } from 'react';
import type { PuzzleData } from '../types/puzzle';
import { transformPuzzle, type CrosswordData } from '../utils/transformPuzzle';

interface UsePuzzleResult {
  puzzle: PuzzleData | null;
  crosswordData: CrosswordData | null;
  loading: boolean;
  error: string | null;
  refetch: (updateCrosswordData?: boolean) => void;
}

export function usePuzzle(puzzleId: string): UsePuzzleResult {
  const [puzzle, setPuzzle] = useState<PuzzleData | null>(null);
  const [crosswordData, setCrosswordData] = useState<CrosswordData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [fetchKey, setFetchKey] = useState(0);
  const updateCrosswordDataRef = useRef(false);

  const refetch = useCallback((updateCrosswordData = false) => {
    updateCrosswordDataRef.current = updateCrosswordData;
    setFetchKey((k) => k + 1);
  }, []);

  useEffect(() => {
    const isRefetch = fetchKey > 0;
    if (!isRefetch) {
      setLoading(true);
    }
    setError(null);

    let cancelled = false;
    const maxRetries = 5;

    async function loadPuzzle(attempt: number) {
      try {
        const res = await fetch(`/puzzles/${puzzleId}.json`);
        if (!res.ok) throw new Error(`Failed to load puzzle: ${res.status}`);
        const contentType = res.headers.get('content-type') || '';
        if (!contentType.includes('json')) {
          // Vite may return index.html for newly created files
          throw new Error('Puzzle not ready yet');
        }
        const data: PuzzleData = await res.json();
        if (!cancelled) {
          setPuzzle(data);
          // Update crosswordData on initial load, or when explicitly requested
          // (e.g. skeleton→real clues). Skip on solver-complete refetch to
          // avoid react-crossword re-init wiping user's entered letters.
          if (!isRefetch || updateCrosswordDataRef.current) {
            setCrosswordData(transformPuzzle(data));
            updateCrosswordDataRef.current = false;
          }
          setLoading(false);
        }
      } catch (err: unknown) {
        if (cancelled) return;
        if (attempt < maxRetries) {
          setTimeout(() => loadPuzzle(attempt + 1), 1000);
        } else {
          setError(err instanceof Error ? err.message : 'Failed to load puzzle');
          setLoading(false);
        }
      }
    }

    loadPuzzle(0);
    return () => { cancelled = true; };
  }, [puzzleId, fetchKey]);

  return { puzzle, crosswordData, loading, error, refetch };
}
