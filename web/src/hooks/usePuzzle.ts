import { useState, useEffect, useCallback } from 'react';
import type { PuzzleData } from '../types/puzzle';
import { transformPuzzle, type CrosswordData } from '../utils/transformPuzzle';

interface UsePuzzleResult {
  puzzle: PuzzleData | null;
  crosswordData: CrosswordData | null;
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

export function usePuzzle(puzzleId: string): UsePuzzleResult {
  const [puzzle, setPuzzle] = useState<PuzzleData | null>(null);
  const [crosswordData, setCrosswordData] = useState<CrosswordData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [fetchKey, setFetchKey] = useState(0);

  const refetch = useCallback(() => {
    setFetchKey((k) => k + 1);
  }, []);

  useEffect(() => {
    setLoading(true);
    setError(null);

    fetch(`/puzzles/${puzzleId}.json`)
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to load puzzle: ${res.status}`);
        return res.json();
      })
      .then((data: PuzzleData) => {
        setPuzzle(data);
        setCrosswordData(transformPuzzle(data));
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, [puzzleId, fetchKey]);

  return { puzzle, crosswordData, loading, error, refetch };
}
