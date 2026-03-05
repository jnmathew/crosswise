import { useRef, useState, useCallback, useMemo, useEffect, useContext } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  CrosswordProvider,
  CrosswordGrid,
  DirectionClues,
  CrosswordContext,
  type CrosswordProviderImperative,
} from '@jaredreisinger/react-crossword';
import { usePuzzle } from '../hooks/usePuzzle';
import { useHints } from '../hooks/useHints';
import { useSSE } from '../hooks/useSSE';
import { crosswordTheme, crosswordThemeDark } from '../styles/theme';
import { useTheme } from '../hooks/useTheme';
import HintPanel from './HintPanel';
import type { PuzzleClue } from '../types/puzzle';

type Direction = 'across' | 'down';

const GRID_HEIGHT = 500;

/**
 * Inner component that reads CrosswordContext to detect active clue changes
 * from any source (cell click, clue click, keyboard navigation).
 */
function ClueTracker({
  onClueChange,
  onPositionChange,
}: {
  onClueChange: (direction: Direction, number: string) => void;
  onPositionChange: (pos: { row: number; col: number }) => void;
}) {
  const { selectedDirection, selectedNumber, selectedPosition } = useContext(CrosswordContext);

  useEffect(() => {
    if (selectedDirection && selectedNumber) {
      onClueChange(selectedDirection, selectedNumber);
    }
  }, [selectedDirection, selectedNumber, onClueChange]);

  useEffect(() => {
    if (selectedPosition) {
      onPositionChange(selectedPosition);
    }
  }, [selectedPosition, onPositionChange]);

  return null;
}

export default function CrosswordPlayer() {
  const { id } = useParams<{ id: string }>();
  const puzzleId = id ?? '';
  const { puzzle, crosswordData, loading, error, refetch } = usePuzzle(puzzleId);
  const crosswordRef = useRef<CrosswordProviderImperative>(null);
  const { revealHint, revealExplanation, revealAnswer, getClueState } = useHints();
  const { dark } = useTheme();

  const refocusGrid = useCallback(() => {
    requestAnimationFrame(() => {
      const input = document.querySelector<HTMLInputElement>('[aria-label="crossword-input"]');
      input?.focus();
    });
  }, []);

  const [activeDirection, setActiveDirection] = useState<Direction | null>(null);
  const [activeNumber, setActiveNumber] = useState<number | null>(null);
  const [activePosition, setActivePosition] = useState<{ row: number; col: number } | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [flashCells, setFlashCells] = useState<{ row: number; col: number; correct: boolean }[]>([]);
  const [showCelebration, setShowCelebration] = useState(false);

  // Solve timer (persisted to localStorage)
  const timerKey = `crosswise-timer-${puzzleId}`;
  const [elapsedSeconds, setElapsedSeconds] = useState(() => {
    const stored = localStorage.getItem(timerKey);
    return stored ? parseInt(stored, 10) || 0 : 0;
  });
  const [timerRunning, setTimerRunning] = useState(false);
  const [showShortcuts, setShowShortcuts] = useState(false);
  const [pencilMode, setPencilMode] = useState(false);
  const [pencilCells, setPencilCells] = useState<Set<string>>(new Set());
  const [showSaved, setShowSaved] = useState(false);
  const savedTimer = useRef<ReturnType<typeof setTimeout> | null>(null);


  // Editable puzzle name (null = use derived default from puzzle metadata)
  const [nameOverride, setNameOverride] = useState<string | null>(null);
  const [editingName, setEditingName] = useState(false);
  const nameInputRef = useRef<HTMLInputElement>(null);

  // Reference image modal
  const [showRefImage, setShowRefImage] = useState<'original' | null>(null);

  // Collapsible solve banner
  const [bannerExpanded, setBannerExpanded] = useState(true);

  // Timer: tick every second when running
  useEffect(() => {
    if (!timerRunning) return;
    const id = setInterval(() => {
      setElapsedSeconds((s) => {
        const next = s + 1;
        localStorage.setItem(timerKey, String(next));
        return next;
      });
    }, 1000);
    return () => clearInterval(id);
  }, [timerRunning, timerKey]);

  // Timer: pause when tab is hidden, resume when visible (only if it was running)
  const timerWasRunning = useRef(false);
  useEffect(() => {
    const handler = () => {
      if (document.hidden) {
        setTimerRunning((r) => { timerWasRunning.current = r; return false; });
      } else if (timerWasRunning.current) {
        setTimerRunning(true);
      }
    };
    document.addEventListener('visibilitychange', handler);
    return () => document.removeEventListener('visibilitychange', handler);
  }, []);

  // Undo: Ctrl+Z / Cmd+Z
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
        e.preventDefault();
        const entry = undoStack.current.pop();
        if (!entry || !crosswordRef.current) return;
        const { row, col, prev, wasPencil } = entry;
        const key = `${row},${col}`;
        if (prev) {
          crosswordRef.current.setGuess(row, col, prev);
          userGridRef.current[key] = prev;
          setPencilCells((s) => { const n = new Set(s); if (wasPencil) n.add(key); else n.delete(key); return n; });
        } else {
          crosswordRef.current.setGuess(row, col, '');
          delete userGridRef.current[key];
          setPencilCells((s) => { const n = new Set(s); n.delete(key); return n; });
        }
      }
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, []);

  // Solve diagnostics
  type DiagCandidate = { word: string; source: string; confidence: number; verified: boolean };
  type TraceEvent = { phase: string; candidates?: number; result?: string };
  type DiagClue = {
    clue_id: string; text: string; length: number; category: string | null;
    candidates: DiagCandidate[]; candidate_count: number;
    assigned_answer: string | null; status: 'solved' | 'unsolved' | 'no_candidates';
    trace?: TraceEvent[]; solved_by?: string | null;
  };
  type TimelineEntry = { t: number; phase: string; summary: string };
  const [diagnostics, setDiagnostics] = useState<DiagClue[] | null>(null);
  const [timeline, setTimeline] = useState<TimelineEntry[] | null>(null);
  const [diagOpen, setDiagOpen] = useState(false);
  const [diagExpandedClue, setDiagExpandedClue] = useState<string | null>(null);

  // Refs for auto-scrolling clue lists
  const acrossCluesRef = useRef<HTMLDivElement>(null);
  const downCluesRef = useRef<HTMLDivElement>(null);

  // Track user cell input for "Check Word" feature
  const userGridRef = useRef<Record<string, string>>({});

  // Undo stack
  const undoStack = useRef<{ row: number; col: number; prev: string | null; wasPencil: boolean }[]>([]);

  const puzzleName = nameOverride ?? (puzzle?.metadata.name || puzzleId.replace('IMG_', 'Puzzle #'));

  // Check if a pipeline/solve is in progress by polling session status
  const [isSolving, setIsSolving] = useState(false);
  const [pipelineError, setPipelineError] = useState<string | null>(null);
  useEffect(() => {
    if (!puzzleId) return;
    fetch(`/api/${puzzleId}/status`)
      .then((r) => r.ok ? r.json() : null)
      .then((data) => {
        if (data && ['ocr_running', 'solving', 'generating_hints'].includes(data.status)) {
          setIsSolving(true);
        }
      })
      .catch(() => {});
  }, [puzzleId]);

  // Connect SSE when puzzle is being solved
  const sseUrl = isSolving ? `/api/${puzzleId}/progress` : null;
  const { data: progress, done: solveDone } = useSSE(sseUrl);

  // Show a toast that auto-dismisses after a timeout
  const showToast = useCallback((message: string, duration = 5000) => {
    setToast(message);
    return setTimeout(() => setToast(null), duration);
  }, []);

  // Flash cells green/red and auto-clear after 800ms
  const flashCellsFeedback = useCallback((cells: { row: number; col: number; correct: boolean }[]) => {
    setFlashCells(cells);
    setTimeout(() => setFlashCells([]), 800);
  }, []);

  // Refetch puzzle when solve completes
  useEffect(() => {
    if (solveDone && progress?.stage === 'complete') {
      setIsSolving(false);
      refetch();
      const timer = showToast('Puzzle solved! Hints are now available.');
      return () => clearTimeout(timer);
    }
    if (solveDone && progress?.stage === 'failed') {
      setIsSolving(false);
      refetch();
      const timer = showToast('Solve finished with partial results.');
      return () => clearTimeout(timer);
    }
    if (solveDone && progress?.stage === 'cancelled') {
      setIsSolving(false);
      const timer = showToast('Solve cancelled.');
      return () => clearTimeout(timer);
    }
  }, [solveDone, progress?.stage, refetch, showToast]);

  // When clues are ready (OCR complete), refetch puzzle and update crosswordData
  const cluesRefetched = useRef(false);
  useEffect(() => {
    if (!progress) return;
    if (cluesRefetched.current) return;
    if (progress.stage === 'clues_ready' || (progress.progress ?? 0) > 0.06) {
      cluesRefetched.current = true;
      refetch(true);
    }
  }, [progress, refetch]);

  // Handle verification failure — show error, stop solving
  useEffect(() => {
    if (progress?.stage === 'verification_failed') {
      setPipelineError(progress.message || 'Clue verification failed. The image may not contain a valid crossword.');
      setIsSolving(false);
    }
  }, [progress?.stage, progress?.message]);

  // Fetch solve diagnostics, guarding against non-JSON responses
  const fetchDiagnostics = useCallback(() => {
    if (!puzzleId) return;
    fetch(`/api/${puzzleId}/diagnostics`)
      .then((r) => {
        const ct = r.headers.get('content-type') || '';
        if (r.ok && ct.includes('application/json')) return r.json();
        return null;
      })
      .then((data) => {
        if (!data) return;
        // Handle both old format (array) and new format ({clues, timeline})
        if (Array.isArray(data)) {
          setDiagnostics(data);
        } else {
          setDiagnostics(data.clues || []);
          if (data.timeline) setTimeline(data.timeline);
        }
      })
      .catch(() => {});
  }, [puzzleId]);

  // Fetch diagnostics once solve completes
  useEffect(() => {
    if (solveDone && progress?.stage === 'complete') fetchDiagnostics();
  }, [solveDone, progress?.stage, fetchDiagnostics]);

  // Also try to load diagnostics on mount (for already-solved puzzles)
  useEffect(() => { fetchDiagnostics(); }, [fetchDiagnostics]);

  const handleCancelSolve = useCallback(() => {
    fetch(`/api/${puzzleId}/cancel`, { method: 'POST' }).catch(() => {});
  }, [puzzleId]);

  // Scroll active clue into view in the clue list
  const scrollClueIntoView = useCallback((dir: Direction, num: number) => {
    const container = dir === 'across'
      ? acrossCluesRef.current
      : downCluesRef.current;

    const selector = `[aria-label="clue-${num}-${dir}"]`;
    // Try scoped search first, then fall back to document
    const clueEl = container?.querySelector(selector) ?? document.querySelector(selector);
    if (clueEl) {
      clueEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, []);

  // Called by ClueTracker (inside CrosswordProvider) whenever the active clue changes
  const handleClueChange = useCallback((direction: Direction, number: string) => {
    const num = parseInt(number, 10);
    setActiveDirection(direction);
    setActiveNumber(num);
    scrollClueIntoView(direction, num);
  }, [scrollClueIntoView]);

  const handlePositionChange = useCallback((pos: { row: number; col: number }) => {
    setActivePosition(pos);
  }, []);

  const clueMap = useMemo(() => {
    if (!puzzle) return {};
    const map: Record<string, PuzzleClue> = {};
    for (const clue of puzzle.clues.across) {
      map[`${clue.number}-across`] = clue;
    }
    for (const clue of puzzle.clues.down) {
      map[`${clue.number}-down`] = clue;
    }
    return map;
  }, [puzzle]);

  const activeClue = activeDirection && activeNumber
    ? clueMap[`${activeNumber}-${activeDirection}`] ?? null
    : null;

  const hintState = activeDirection && activeNumber
    ? getClueState(activeDirection, activeNumber)
    : { hintRevealed: false, explanationRevealed: false, answerRevealed: false };

  // onClueSelected fires when a clue is clicked in the list (not from cell clicks).
  // ClueTracker handles all sources via context, but we keep this for the callback prop.
  const handleClueSelected = useCallback(() => {
    // ClueTracker handles state updates — no-op here to avoid double-updates
  }, []);

  // Check if all white cells are filled correctly
  const checkCompletion = useCallback(() => {
    if (!puzzle) return;
    for (const dir of ['across', 'down'] as const) {
      for (const clue of puzzle.clues[dir]) {
        if (!clue.answer || clue.answer.includes('?')) continue;
        const [sr, sc] = clue.start;
        for (let i = 0; i < clue.length; i++) {
          const r = dir === 'across' ? sr : sr + i;
          const c = dir === 'across' ? sc + i : sc;
          const user = userGridRef.current[`${r},${c}`];
          if (!user || user !== clue.answer[i].toUpperCase()) return;
        }
      }
    }
    // All cells correct!
    setTimerRunning(false);
    setShowCelebration(true);
  }, [puzzle]);

  const timerAutoStarted = useRef(false);
  const handleCellChange = useCallback((row: number, col: number, char: string) => {
    const key = `${row},${col}`;
    // Push to undo stack (cap at 200)
    const prev = userGridRef.current[key] || null;
    const wasPencil = pencilCells.has(key);
    if (char !== (prev || '')) {
      undoStack.current.push({ row, col, prev, wasPencil });
      if (undoStack.current.length > 200) undoStack.current.shift();
    }
    if (char) {
      userGridRef.current[key] = char.toUpperCase();
      // Track pencil vs ink
      setPencilCells((prev) => {
        const next = new Set(prev);
        if (pencilMode) {
          next.add(key);
        } else {
          next.delete(key); // ink overwrites pencil
        }
        return next;
      });
      // Auto-start timer on first letter typed
      if (!timerAutoStarted.current) {
        timerAutoStarted.current = true;
        setTimerRunning(true);
      }
      checkCompletion();
    } else {
      delete userGridRef.current[key];
      setPencilCells((prev) => { const next = new Set(prev); next.delete(key); return next; });
    }
    // Show saved indicator (debounced)
    if (savedTimer.current) clearTimeout(savedTimer.current);
    savedTimer.current = setTimeout(() => {
      setShowSaved(true);
      savedTimer.current = setTimeout(() => setShowSaved(false), 1500);
    }, 500);
  // eslint-disable-next-line react-hooks/exhaustive-deps -- pencilCells read for undo snapshot; stale read is acceptable
  }, [checkCompletion, pencilMode]);

  // Helper: get correct letter at a cell by finding the clue that covers it
  const getCorrectLetter = useCallback((row: number, col: number): string | null => {
    if (!puzzle) return null;
    for (const dir of ['across', 'down'] as const) {
      for (const clue of puzzle.clues[dir]) {
        if (!clue.answer || clue.answer.includes('?')) continue;
        const [sr, sc] = clue.start;
        for (let i = 0; i < clue.length; i++) {
          const r = dir === 'across' ? sr : sr + i;
          const c = dir === 'across' ? sc + i : sc;
          if (r === row && c === col) return clue.answer[i].toUpperCase();
        }
      }
    }
    return null;
  }, [puzzle]);

  // --- Check actions ---
  const handleCheckLetter = useCallback(() => {
    if (!activePosition) return;
    const { row, col } = activePosition;
    const correct = getCorrectLetter(row, col);
    if (!correct) { showToast('No answer available for this cell'); return; }
    const user = userGridRef.current[`${row},${col}`];
    if (!user) { showToast('Empty cell'); return; }
    const isCorrect = user === correct;
    flashCellsFeedback([{ row, col, correct: isCorrect }]);
    showToast(isCorrect ? 'Correct!' : 'Incorrect');
  }, [activePosition, getCorrectLetter, showToast, flashCellsFeedback]);

  const handleCheckWord = useCallback(() => {
    if (!activeClue || !activeDirection) return;
    const answer = activeClue.answer;
    if (!answer || answer.includes('?')) { showToast('No answer available'); return; }
    const [sr, sc] = activeClue.start;
    let allFilled = true;
    const cells: { row: number; col: number; correct: boolean }[] = [];
    for (let i = 0; i < activeClue.length; i++) {
      const r = activeDirection === 'across' ? sr : sr + i;
      const c = activeDirection === 'across' ? sc + i : sc;
      const letter = userGridRef.current[`${r},${c}`];
      if (!letter) { allFilled = false; break; }
      cells.push({ row: r, col: c, correct: letter === answer[i].toUpperCase() });
    }
    if (!allFilled) { showToast('Fill in all letters first'); return; }
    flashCellsFeedback(cells);
    const allCorrect = cells.every((c) => c.correct);
    showToast(allCorrect ? 'Correct!' : 'Not quite — try again');
  }, [activeClue, activeDirection, showToast, flashCellsFeedback]);

  const handleCheckPuzzle = useCallback(() => {
    if (!puzzle) return;
    let correct = 0;
    let total = 0;
    for (const dir of ['across', 'down'] as const) {
      for (const clue of puzzle.clues[dir]) {
        if (!clue.answer || clue.answer.includes('?')) continue;
        total++;
        const [sr, sc] = clue.start;
        let word = '';
        let filled = true;
        for (let i = 0; i < clue.length; i++) {
          const r = dir === 'across' ? sr : sr + i;
          const c = dir === 'across' ? sc + i : sc;
          const letter = userGridRef.current[`${r},${c}`];
          if (letter) word += letter;
          else { filled = false; break; }
        }
        if (filled && word === clue.answer.toUpperCase()) correct++;
      }
    }
    showToast(`${correct} of ${total} words correct`);
  }, [puzzle, showToast]);

  // --- Reveal actions ---
  const handleRevealLetter = useCallback(() => {
    if (!activePosition || !crosswordRef.current) return;
    const { row, col } = activePosition;
    const correct = getCorrectLetter(row, col);
    if (!correct) { showToast('No answer available for this cell'); return; }
    crosswordRef.current.setGuess(row, col, correct);
    userGridRef.current[`${row},${col}`] = correct;
    if (activeDirection && activeNumber) revealAnswer(activeDirection, activeNumber);
  }, [activePosition, getCorrectLetter, activeDirection, activeNumber, revealAnswer, showToast]);

  const handleRevealWord = useCallback(() => {
    if (!activeDirection || !activeNumber || !activeClue?.answer || !crosswordRef.current) return;
    revealAnswer(activeDirection, activeNumber);
    const answer = activeClue.answer!;
    const [sr, sc] = activeClue.start;
    for (let i = 0; i < answer.length; i++) {
      const r = activeDirection === 'across' ? sr : sr + i;
      const c = activeDirection === 'across' ? sc + i : sc;
      crosswordRef.current.setGuess(r, c, answer[i].toUpperCase());
      userGridRef.current[`${r},${c}`] = answer[i].toUpperCase();
    }
  }, [activeDirection, activeNumber, activeClue, revealAnswer]);

  const handleRevealPuzzle = useCallback(() => {
    if (!puzzle || !crosswordRef.current) return;
    if (!window.confirm('This will reveal all answers and cannot be undone. Are you sure?')) return;
    for (const dir of ['across', 'down'] as const) {
      for (const clue of puzzle.clues[dir]) {
        if (!clue.answer || clue.answer.includes('?')) continue;
        revealAnswer(dir === 'across' ? 'across' : 'down', clue.number);
        const [sr, sc] = clue.start;
        for (let i = 0; i < clue.answer.length; i++) {
          const r = dir === 'across' ? sr : sr + i;
          const c = dir === 'across' ? sc + i : sc;
          crosswordRef.current!.setGuess(r, c, clue.answer[i].toUpperCase());
          userGridRef.current[`${r},${c}`] = clue.answer[i].toUpperCase();
        }
      }
    }
  }, [puzzle, revealAnswer]);

  // --- Clear actions ---
  const handleClearLetter = useCallback(() => {
    if (!activePosition || !crosswordRef.current) return;
    const { row, col } = activePosition;
    const key = `${row},${col}`;
    crosswordRef.current.setGuess(row, col, '');
    delete userGridRef.current[key];
    setPencilCells((prev) => { const n = new Set(prev); n.delete(key); return n; });
  }, [activePosition]);

  const handleClearWord = useCallback(() => {
    if (!activeClue || !activeDirection || !crosswordRef.current) return;
    const [sr, sc] = activeClue.start;
    for (let i = 0; i < activeClue.length; i++) {
      const r = activeDirection === 'across' ? sr : sr + i;
      const c = activeDirection === 'across' ? sc + i : sc;
      const key = `${r},${c}`;
      crosswordRef.current.setGuess(r, c, '');
      delete userGridRef.current[key];
      setPencilCells((prev) => { const n = new Set(prev); n.delete(key); return n; });
    }
  }, [activeClue, activeDirection]);

  const handleClearPuzzle = useCallback(() => {
    if (!crosswordRef.current) return;
    if (!window.confirm('Clear all your answers? This cannot be undone.')) return;
    crosswordRef.current.reset();
    userGridRef.current = {};
    setPencilCells(new Set());
    undoStack.current = [];
  }, []);

  const savePuzzleName = useCallback((name: string) => {
    setNameOverride(name);
    setEditingName(false);
    fetch(`/api/puzzles/${puzzleId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
  }, [puzzleId]);

  // Focus name input when entering edit mode
  useEffect(() => {
    if (editingName && nameInputRef.current) {
      nameInputRef.current.focus();
      nameInputRef.current.select();
    }
  }, [editingName]);

  if (loading) return <div style={styles.loading}>Loading puzzle...</div>;
  if (error) return <div style={styles.error}>Error: {error}</div>;
  if (!crosswordData || !puzzle) return <div style={styles.error}>No puzzle data</div>;

  // Grid SVG scales to fill width; actual height depends on row/col ratio
  const gridHeight = Math.round(GRID_HEIGHT * (puzzle.grid.rows / puzzle.grid.cols));

  return (
    <div style={styles.container} className="player-container page-fade">
      <Link to="/" style={styles.brandLink} className="brand-link">
        <img src="/crosswise.svg" alt="" style={styles.brandLogo} />
        <h1 style={styles.brand}>Crosswise</h1>
      </Link>

      {/* Solve progress banner (collapsible) */}
      {isSolving && (
        <div style={styles.solveBanner} className="solve-banner">
          <div
            style={styles.solveBannerHeader}
            onClick={() => setBannerExpanded((v) => !v)}
          >
            <div style={styles.solveBannerContent}>
              <div style={styles.spinner} />
              <span>
                {progress
                  ? progress.message
                  : 'Solving puzzle in the background...'}
              </span>
            </div>
            <span style={styles.bannerToggle}>
              {bannerExpanded ? '\u25B2' : '\u25BC'}
            </span>
          </div>
          {bannerExpanded && (
            <>
              {progress && progress.progress >= 0 && progress.progress <= 1 && (
                <div style={styles.bannerProgress}>
                  <div
                    style={{
                      ...styles.bannerProgressFill,
                      width: `${Math.round(progress.progress * 100)}%`,
                    }}
                  />
                </div>
              )}
              {progress?.warning && (
                <p style={styles.solveBannerWarning}>
                  {progress.warning}
                </p>
              )}
              <p style={styles.solveBannerSub}>
                You can start filling in answers while the solver works. Hints will appear when ready.
              </p>
              <button
                style={styles.stopBtn}
                className="btn-stop"
                onClick={(e) => { e.stopPropagation(); handleCancelSolve(); }}
              >
                Stop Solve
              </button>
            </>
          )}
        </div>
      )}

      {/* Pipeline error banner (e.g. verification failed) */}
      {pipelineError && (
        <div style={styles.errorBanner} className="error-banner">
          <span>{pipelineError}</span>
          <Link to="/upload" style={styles.errorBannerLink}>Try another image</Link>
        </div>
      )}

      {/* Solve diagnostics panel */}
      {diagnostics && (() => {
        const solvedCount = diagnostics.filter((d) => d.status === 'solved').length;
        const unsolved = diagnostics.filter((d) => d.status !== 'solved');

        const solvedByColor: Record<string, string> = {
          db_prefill: '#9ca3af',
          constraint_prop: '#16a34a',
          constraint_prop_final: '#16a34a',
          conflict_resolution: '#ea580c',
          post_resolution: '#2563eb',
          csp_cleanup: '#7c3aed',
        };
        const solvedByLabel = (s: string | null | undefined) => {
          if (!s) return null;
          const color = s.startsWith('llm_pass') ? '#2563eb' : (solvedByColor[s] || '#666');
          const label = s.replace(/_/g, ' ');
          return <span style={{ fontSize: '10px', padding: '1px 5px', borderRadius: '3px', background: color + '18', color, fontWeight: 500, marginLeft: '4px' }}>{label}</span>;
        };

        const traceLineEl = (trace?: TraceEvent[]) => {
          if (!trace || trace.length === 0) return null;
          const parts = trace.map((ev) => {
            if (ev.phase === 'db_lookup') return `DB: ${ev.candidates}`;
            if (ev.phase === 'web_search') return `Web: ${ev.result}`;
            if (ev.phase === 'opus') return `Opus: ${ev.candidates}`;
            if (ev.phase === 'sonnet_pad') return `Sonnet: ${ev.candidates}`;
            return ev.phase;
          });
          return <div style={{ fontSize: '11px', color: '#888', paddingLeft: '16px' }}>{parts.join(' \u00B7 ')}</div>;
        };

        const renderDiagClue = (d: DiagClue) => (
          <div key={d.clue_id} style={styles.diagClue}>
            <div
              style={styles.diagClueHeader}
              onClick={() => setDiagExpandedClue(diagExpandedClue === d.clue_id ? null : d.clue_id)}
            >
              <span>
                <strong>{d.clue_id}</strong>: {d.text} ({d.length})
                {d.assigned_answer && (
                  <span style={{ color: '#16a34a', marginLeft: '6px' }}>= {d.assigned_answer}</span>
                )}
                {solvedByLabel(d.solved_by)}
              </span>
              <span style={{ fontSize: '11px', color: '#999' }}>
                {d.candidate_count === 0 ? 'no candidates' : `${d.candidate_count} cands`}
                {' '}{diagExpandedClue === d.clue_id ? '\u25B2' : '\u25BC'}
              </span>
            </div>
            {traceLineEl(d.trace)}
            {diagExpandedClue === d.clue_id && d.candidates.length > 0 && (
              <div style={styles.diagCandidates}>
                {d.candidates.slice(0, 20).map((c, i) => (
                  <div key={i} style={styles.diagCandRow}>
                    <span style={{ fontFamily: 'monospace' }}>{c.word}</span>
                    <span style={{ color: '#888', fontSize: '11px' }}>
                      {c.source} &middot; {Math.round(c.confidence * 100)}%
                      {c.verified && ' \u2713'}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        );

        return (
          <div style={styles.diagPanel}>
            <div style={styles.diagHeader} onClick={() => setDiagOpen((v) => !v)}>
              <span>
                Solve Log: {solvedCount}/{diagnostics.length} clues solved
                {unsolved.length > 0 && `. ${unsolved.length} unsolved.`}
              </span>
              <span style={styles.bannerToggle}>{diagOpen ? '\u25B2' : '\u25BC'}</span>
            </div>
            {diagOpen && (
              <div style={styles.diagBody}>
                {/* Timeline */}
                {timeline && timeline.length > 0 && (
                  <details style={{ marginBottom: '8px' }}>
                    <summary style={{ cursor: 'pointer', fontSize: '13px', color: '#666' }}>
                      Timeline ({timeline.length} events)
                    </summary>
                    <div style={{ fontFamily: 'monospace', fontSize: '11px', lineHeight: 1.6, padding: '4px 0' }}>
                      {timeline.map((ev, i) => (
                        <div key={i}>
                          <span style={{ color: '#999', marginRight: '6px' }}>{ev.t}s</span>
                          <span style={{ color: '#555', fontWeight: 500 }}>{ev.phase}</span>
                          <span style={{ color: '#888', marginLeft: '6px' }}>{ev.summary}</span>
                        </div>
                      ))}
                    </div>
                  </details>
                )}

                {unsolved.length > 0 && (
                  <div style={{ marginBottom: '8px' }}>
                    <strong>Unsolved clues:</strong>
                    {unsolved.map(renderDiagClue)}
                  </div>
                )}
                <details>
                  <summary style={{ cursor: 'pointer', fontSize: '13px', color: '#666' }}>
                    All clues ({diagnostics.length})
                  </summary>
                  {diagnostics.map(renderDiagClue)}
                </details>
              </div>
            )}
          </div>
        );
      })()}

      {/* Toast notification */}
      {toast && (
        <div style={styles.toast} className="toast">
          {toast}
          <button style={styles.toastClose} onClick={() => setToast(null)}>&times;</button>
        </div>
      )}

      {/* Completion celebration modal */}
      {showCelebration && (
        <div style={styles.modalOverlay} onClick={() => setShowCelebration(false)}>
          <div style={styles.celebrationModal} className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div style={styles.celebrationEmoji}>&#127881;</div>
            <h2 style={styles.celebrationTitle}>Puzzle Complete!</h2>
            <p style={styles.celebrationTime}>
              Solved in {String(Math.floor(elapsedSeconds / 60)).padStart(2, '0')}:{String(elapsedSeconds % 60).padStart(2, '0')}
            </p>
            <button
              style={styles.celebrationBtn}
              className="btn-primary"
              onClick={() => setShowCelebration(false)}
            >
              Continue
            </button>
          </div>
        </div>
      )}

      <div style={styles.header} className="player-header">
        <Link to="/" style={styles.backLink} className="link-back">&larr; Puzzles</Link>
        {editingName ? (
          <input
            ref={nameInputRef}
            style={styles.nameInput}
            value={puzzleName}
            onChange={(e) => setNameOverride(e.target.value)}
            onBlur={() => savePuzzleName(puzzleName)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') savePuzzleName(puzzleName);
              if (e.key === 'Escape') { setNameOverride(null); setEditingName(false); }
            }}
          />
        ) : (
          <span
            style={styles.puzzleName}
            onClick={() => setEditingName(true)}
            title="Click to rename"
          >
            {puzzleName}
          </span>
        )}
        <button
          style={styles.refImageBtn}
          className="btn-ref"
          onClick={() => setShowRefImage('original')}
          title="View source photos"
        >
          Source Image
        </button>
        <div style={styles.toolbarGroup} className="player-toolbar">
        <div style={styles.timerBox} className="timer-box">
          <span style={styles.timerDisplay}>
            {String(Math.floor(elapsedSeconds / 60)).padStart(2, '0')}:
            {String(elapsedSeconds % 60).padStart(2, '0')}
          </span>
          <button
            style={styles.timerToggle}
            onClick={() => { setTimerRunning((r) => !r); refocusGrid(); }}
            title={timerRunning ? 'Pause timer' : 'Resume timer'}
            className="btn-timer"
          >
            {timerRunning ? (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" style={{ display: 'block' }}><rect x="5" y="3" width="5" height="18" rx="1" /><rect x="14" y="3" width="5" height="18" rx="1" /></svg>
            ) : (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" style={{ display: 'block' }}><path d="M6 3.5L20 12 6 20.5z" /></svg>
            )}
          </button>
          <button
            style={styles.timerToggle}
            className="btn-timer"
            onClick={() => { setElapsedSeconds(0); localStorage.setItem(timerKey, '0'); refocusGrid(); }}
            title="Reset timer"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ display: 'block' }}><path d="M1 4v6h6" /><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" /></svg>
          </button>
        </div>
        {showSaved && (
          <span style={styles.savedIndicator}>Saved</span>
        )}
        <button
          style={{
            ...styles.shortcutsBtn,
            ...(pencilMode
              ? dark
                ? { backgroundColor: '#2563eb', borderColor: '#fff', boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.3)', color: '#fff' }
                : { backgroundColor: '#2563eb', borderColor: '#fff', boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.2)', color: '#fff' }
              : {}),
          }}
          className="btn-ref"
          onClick={() => { setPencilMode((m) => !m); refocusGrid(); }}
          title={pencilMode ? 'Switch to ink mode' : 'Switch to pencil mode'}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill={pencilMode ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ display: 'block' }}><path d="M17 3a2.83 2.83 0 114 4L7.5 20.5 2 22l1.5-5.5z" /></svg>
        </button>
        <div style={{ position: 'relative' }}>
          <button
            style={styles.shortcutsBtn}
            className="btn-ref"
            onClick={() => setShowShortcuts((s) => !s)}
            title="Keyboard shortcuts"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="2" y="4" width="20" height="16" rx="2" /><line x1="6" y1="8" x2="6" y2="8" /><line x1="10" y1="8" x2="10" y2="8" /><line x1="14" y1="8" x2="14" y2="8" /><line x1="18" y1="8" x2="18" y2="8" /><line x1="6" y1="12" x2="6" y2="12" /><line x1="10" y1="12" x2="10" y2="12" /><line x1="14" y1="12" x2="14" y2="12" /><line x1="18" y1="12" x2="18" y2="12" /><line x1="8" y1="16" x2="16" y2="16" /></svg>
          </button>
          {showShortcuts && (
            <div
              style={styles.shortcutsPopover}
              className="shortcuts-popover"
              tabIndex={-1}
              ref={(el) => el?.focus()}
              onKeyDown={(e) => { if (e.key === 'Escape') setShowShortcuts(false); }}
              onBlur={(e) => { if (!e.currentTarget.contains(e.relatedTarget)) setShowShortcuts(false); }}
            >
              <div style={styles.shortcutsTitle}>Keyboard Shortcuts</div>
              <div style={styles.shortcutRow}><kbd style={styles.kbd}>&larr; &rarr; &uarr; &darr;</kbd> Navigate cells</div>
              <div style={styles.shortcutRow}><kbd style={styles.kbd}>Tab</kbd> / <kbd style={styles.kbd}>Space</kbd> Switch direction</div>
              <div style={styles.shortcutRow}><kbd style={styles.kbd}>Home</kbd> / <kbd style={styles.kbd}>End</kbd> Jump to start/end</div>
              <div style={styles.shortcutRow}><kbd style={styles.kbd}>Backspace</kbd> Delete &amp; go back</div>
              <div style={styles.shortcutRow}><kbd style={styles.kbd}>A-Z</kbd> Type letter &amp; advance</div>
              <div style={styles.shortcutRow}><kbd style={styles.kbd}>Ctrl+Z</kbd> Undo last change</div>
            </div>
          )}
        </div>
        </div>
      </div>

      {/* Reference image modal */}
      {showRefImage && (
        <div style={styles.modalOverlay} onClick={() => setShowRefImage(null)}>
          <div style={styles.modalContent} className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div style={styles.modalHeader}>
              <span style={{ fontWeight: 600 }}>Source Image</span>
              <button style={styles.modalClose} onClick={() => setShowRefImage(null)}>&times;</button>
            </div>
            <img
              src={`/api/files/${puzzleId}/original.jpg`}
              alt="Original crossword photo"
              style={styles.refImage}
            />
          </div>
        </div>
      )}

      <CrosswordProvider
        key={`${puzzleId}-${puzzle.metadata.verification}`}
        ref={crosswordRef}
        data={crosswordData}
        theme={dark ? crosswordThemeDark : crosswordTheme}
        useStorage={true}
        storageKey={`crosswise-${puzzleId}`}
        onClueSelected={handleClueSelected}
        onCellChange={handleCellChange}
      >
        <ClueTracker onClueChange={handleClueChange} onPositionChange={handlePositionChange} />
        {/* Active clue bar with hints + check/reveal */}
        <HintPanel
          clue={activeClue}
          direction={activeDirection}
          hintState={hintState}
          isSolving={isSolving}
          onRevealHint={() => { if (activeDirection && activeNumber) revealHint(activeDirection, activeNumber); refocusGrid(); }}
          onRevealExplanation={() => { if (activeDirection && activeNumber) revealExplanation(activeDirection, activeNumber); refocusGrid(); }}
          onCheckLetter={() => { handleCheckLetter(); refocusGrid(); }}
          onCheckWord={() => { handleCheckWord(); refocusGrid(); }}
          onCheckPuzzle={() => { handleCheckPuzzle(); refocusGrid(); }}
          onRevealLetter={() => { handleRevealLetter(); refocusGrid(); }}
          onRevealWord={() => { handleRevealWord(); refocusGrid(); }}
          onRevealPuzzle={() => { handleRevealPuzzle(); refocusGrid(); }}
          onClearLetter={() => { handleClearLetter(); refocusGrid(); }}
          onClearWord={() => { handleClearWord(); refocusGrid(); }}
          onClearPuzzle={() => { handleClearPuzzle(); refocusGrid(); }}
        />
        <div style={styles.body} className="player-body">
          <div style={{ width: GRID_HEIGHT, height: gridHeight, flexShrink: 0, position: 'relative' }} className="player-grid">
            <CrosswordGrid />
            {(flashCells.length > 0 || pencilCells.size > 0) && puzzle && (
              <svg
                style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
                viewBox={`0 0 ${puzzle.grid.cols} ${puzzle.grid.rows}`}
              >
                {/* Pencil cell text overlay */}
                {Array.from(pencilCells).map((key) => {
                  const [r, c] = key.split(',').map(Number);
                  const letter = userGridRef.current[key];
                  if (!letter) return null;
                  const bg = dark ? '#1e1e2e' : '#fff';
                  return (
                    <g key={`pencil-${key}`}>
                      {/* Cover original text */}
                      <rect x={c + 0.15} y={r + 0.2} width={0.7} height={0.7} fill={bg} />
                      {/* Pencil letter */}
                      <text
                        x={c + 0.5}
                        y={r + 0.58}
                        textAnchor="middle"
                        dominantBaseline="middle"
                        style={{ fontFamily: "'Caveat', cursive", fontSize: '0.7px', fill: dark ? '#93c5fd' : '#4a7ac7' }}
                      >
                        {letter}
                      </text>
                      {/* Dot indicator */}
                      <circle cx={c + 0.88} cy={r + 0.14} r={0.06} fill="#93c5fd" />
                    </g>
                  );
                })}
                {/* Flash feedback */}
                {flashCells.map((fc) => (
                  <rect
                    key={`${fc.row}-${fc.col}`}
                    x={fc.col}
                    y={fc.row}
                    width={1}
                    height={1}
                    fill={fc.correct ? '#22c55e' : '#ef4444'}
                    opacity={0.35}
                  >
                    <animate attributeName="opacity" from="0.45" to="0" dur="0.8s" fill="freeze" />
                  </rect>
                ))}
              </svg>
            )}
          </div>

          <div style={{ ...styles.cluesRow, height: gridHeight }} className="player-clues">
            <div ref={acrossCluesRef} style={styles.clueBox} className="clue-box">
              <DirectionClues direction="across" />
            </div>
            <div ref={downCluesRef} style={styles.clueBox} className="clue-box">
              <DirectionClues direction="down" />
            </div>
          </div>
        </div>
      </CrosswordProvider>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: '20px 40px',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  },
  brandLink: {
    textDecoration: 'none',
    color: 'inherit',
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    marginBottom: '16px',
  },
  brandLogo: {
    width: '32px',
    height: '32px',
  },
  brand: {
    margin: 0,
    fontSize: '42px',
    lineHeight: 0,
    fontFamily: '"Georgia", "Times New Roman", serif',
    fontWeight: 400,
    letterSpacing: '6px',
    textTransform: 'uppercase' as const,
    textAlign: 'center' as const,
  },
  solveBanner: {
    padding: '12px 20px',
    backgroundColor: '#eff6ff',
    border: '1px solid #bfdbfe',
    borderRadius: '8px',
    fontSize: '14px',
    color: '#1e40af',
    marginBottom: '12px',
    width: '100%',
    maxWidth: '1100px',
  },
  solveBannerHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    cursor: 'pointer',
  },
  solveBannerContent: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
  },
  bannerToggle: {
    fontSize: '12px',
    color: '#6b7280',
    userSelect: 'none' as const,
    padding: '0 4px',
  },
  solveBannerWarning: {
    margin: '6px 0 0 0',
    padding: '4px 8px',
    fontSize: '12px',
    color: '#92400e',
    backgroundColor: '#fef3c7',
    borderRadius: '4px',
  },
  solveBannerSub: {
    margin: '4px 0 0 0',
    fontSize: '12px',
    color: '#6b7280',
  },
  stopBtn: {
    marginTop: '6px',
    padding: '4px 14px',
    fontSize: '12px',
    color: '#b91c1c',
    backgroundColor: '#fef2f2',
    border: '1px solid #fca5a5',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  spinner: {
    width: '16px',
    height: '16px',
    border: '2px solid #bfdbfe',
    borderTopColor: '#2563eb',
    borderRadius: '50%',
    animation: 'crosswise-spin 0.8s linear infinite',
    flexShrink: 0,
  },
  bannerProgress: {
    height: '6px',
    backgroundColor: '#dbeafe',
    borderRadius: '3px',
    overflow: 'hidden',
    marginTop: '4px',
  },
  bannerProgressFill: {
    height: '100%',
    backgroundColor: '#2563eb',
    borderRadius: '3px',
    transition: 'width 0.3s ease',
  },
  toolbarGroup: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    flexShrink: 0,
  },
  savedIndicator: {
    fontSize: '11px',
    color: '#16a34a',
    fontWeight: 500,
    opacity: 0.7,
    flexShrink: 0,
  },
  toast: {
    position: 'fixed' as const,
    top: '20px',
    right: '20px',
    padding: '12px 20px',
    backgroundColor: '#065f46',
    color: '#fff',
    borderRadius: '8px',
    fontSize: '14px',
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    zIndex: 1000,
    boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
  },
  toastClose: {
    background: 'none',
    border: 'none',
    color: '#fff',
    fontSize: '18px',
    cursor: 'pointer',
    padding: 0,
    lineHeight: 1,
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: '20px',
    marginBottom: '20px',
    width: '100%',
    maxWidth: '1100px',
  },
  backLink: {
    color: '#2563eb',
    textDecoration: 'none',
    fontSize: '14px',
  },
  puzzleName: {
    fontSize: '14px',
    color: '#444',
    flex: 1,
    minWidth: 0,
    cursor: 'text',
    border: '1px solid #d1d5db',
    borderRadius: '4px',
    padding: '0 8px',
    height: '32px',
    lineHeight: '32px',
    backgroundColor: '#f9fafb',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap' as const,
  },
  nameInput: {
    fontSize: '14px',
    color: '#333',
    flex: 1,
    border: '1px solid #93c5fd',
    borderRadius: '4px',
    padding: '0 8px',
    height: '32px',
    lineHeight: '32px',
    outline: 'none',
    backgroundColor: '#eff6ff',
  },
  refImageBtn: {
    height: '32px',
    padding: '0 10px',
    fontSize: '12px',
    border: '1px solid #d1d5db',
    borderRadius: '4px',
    backgroundColor: '#f9fafb',
    cursor: 'pointer',
    color: '#555',
    flexShrink: 0,
  },
  timerBox: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    flexShrink: 0,
    height: '32px',
    padding: '0 10px',
    border: '1px solid #d1d5db',
    borderRadius: '4px',
    backgroundColor: '#f9fafb',
  },
  timerDisplay: {
    fontFamily: '"SF Mono", "Menlo", monospace',
    fontSize: '14px',
    color: '#555',
    minWidth: '48px',
  },
  timerToggle: {
    background: 'none',
    border: 'none',
    fontSize: '16px',
    cursor: 'pointer',
    padding: 0,
    width: '24px',
    height: '24px',
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#888',
    lineHeight: 1,
  },
  shortcutsBtn: {
    height: '32px',
    padding: '0 10px',
    fontSize: '12px',
    border: '1px solid #d1d5db',
    borderRadius: '4px',
    backgroundColor: '#f9fafb',
    cursor: 'pointer',
    color: '#555',
    flexShrink: 0,
    whiteSpace: 'nowrap' as const,
  },
  modalOverlay: {
    position: 'fixed' as const,
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.6)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 2000,
  },
  modalContent: {
    backgroundColor: '#fff',
    borderRadius: '10px',
    maxWidth: '90vw',
    maxHeight: '90vh',
    overflow: 'auto',
    boxShadow: '0 8px 32px rgba(0,0,0,0.3)',
  },
  modalHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '8px 16px',
    borderBottom: '1px solid #eee',
  },
  modalClose: {
    background: 'none',
    border: 'none',
    fontSize: '22px',
    cursor: 'pointer',
    color: '#666',
    padding: '0 4px',
    lineHeight: 1,
  },
  refImage: {
    display: 'block',
    maxWidth: '85vw',
    maxHeight: '80vh',
    objectFit: 'contain' as const,
  },
  body: {
    display: 'flex',
    gap: '24px',
    alignItems: 'flex-start',
    width: '100%',
    maxWidth: '1100px',
  },
  cluesRow: {
    display: 'flex',
    gap: '12px',
    flex: 1,
    minWidth: 0,
  },
  clueBox: {
    flex: 1,
    overflowY: 'auto',
    border: '1px solid #ddd',
    borderRadius: '6px',
    padding: '8px 12px',
    fontSize: '14px',
    minWidth: 0,
    boxSizing: 'border-box' as const,
  },
  loading: {
    textAlign: 'center',
    padding: '60px',
    fontSize: '18px',
    color: '#666',
  },
  error: {
    textAlign: 'center',
    padding: '60px',
    fontSize: '18px',
    color: '#c44',
  },
  errorBanner: {
    padding: '12px 20px',
    backgroundColor: '#fef2f2',
    border: '1px solid #fecaca',
    borderRadius: '8px',
    fontSize: '14px',
    color: '#991b1b',
    marginBottom: '12px',
    width: '100%',
    maxWidth: '1100px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  errorBannerLink: {
    color: '#2563eb',
    textDecoration: 'none',
    fontWeight: 600,
    fontSize: '14px',
    flexShrink: 0,
  },
  diagPanel: {
    padding: '10px 16px',
    backgroundColor: '#fefce8',
    border: '1px solid #fde68a',
    borderRadius: '8px',
    fontSize: '13px',
    color: '#92400e',
    marginBottom: '12px',
    width: '100%',
    maxWidth: '1100px',
  },
  diagHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    cursor: 'pointer',
    fontWeight: 600,
  },
  diagBody: {
    marginTop: '8px',
    maxHeight: '400px',
    overflowY: 'auto' as const,
  },
  diagClue: {
    margin: '4px 0',
    padding: '6px 8px',
    backgroundColor: '#fffbeb',
    borderRadius: '4px',
    border: '1px solid #fef3c7',
  },
  diagClueHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    cursor: 'pointer',
    fontSize: '13px',
  },
  diagCandidates: {
    marginTop: '4px',
    paddingLeft: '12px',
    borderLeft: '2px solid #fde68a',
  },
  diagCandRow: {
    display: 'flex',
    justifyContent: 'space-between',
    padding: '1px 0',
    fontSize: '12px',
  },
  shortcutsPopover: {
    position: 'absolute' as const,
    top: '100%',
    right: 0,
    marginTop: '6px',
    padding: '12px 16px',
    backgroundColor: '#fff',
    border: '1px solid #ddd',
    borderRadius: '8px',
    boxShadow: '0 4px 16px rgba(0,0,0,0.12)',
    zIndex: 1000,
    whiteSpace: 'nowrap' as const,
    fontSize: '13px',
    color: '#444',
    minWidth: '220px',
    outline: 'none',
  },
  shortcutsTitle: {
    fontWeight: 600,
    fontSize: '14px',
    marginBottom: '8px',
    color: '#222',
  },
  shortcutRow: {
    padding: '3px 0',
    display: 'flex',
    gap: '8px',
    alignItems: 'center',
  },
  kbd: {
    display: 'inline-block',
    padding: '1px 6px',
    backgroundColor: '#f3f4f6',
    border: '1px solid #d1d5db',
    borderRadius: '3px',
    fontFamily: '"SF Mono", "Menlo", monospace',
    fontSize: '11px',
    color: '#374151',
    lineHeight: '18px',
  },
  celebrationModal: {
    backgroundColor: '#fff',
    borderRadius: '16px',
    padding: '40px 48px',
    textAlign: 'center' as const,
    boxShadow: '0 8px 32px rgba(0,0,0,0.3)',
    maxWidth: '360px',
  },
  celebrationEmoji: {
    fontSize: '48px',
    marginBottom: '8px',
  },
  celebrationTitle: {
    margin: '0 0 8px 0',
    fontSize: '24px',
    fontWeight: 700,
    color: '#111',
  },
  celebrationTime: {
    margin: '0 0 24px 0',
    fontSize: '18px',
    fontFamily: '"SF Mono", "Menlo", monospace',
    color: '#555',
  },
  celebrationBtn: {
    padding: '10px 32px',
    fontSize: '15px',
    backgroundColor: '#2563eb',
    color: '#fff',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    fontWeight: 500,
  },
};
