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
import { crosswordTheme } from '../styles/theme';
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

  const [activeDirection, setActiveDirection] = useState<Direction | null>(null);
  const [activeNumber, setActiveNumber] = useState<number | null>(null);
  const [activePosition, setActivePosition] = useState<{ row: number; col: number } | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  // Solve timer (persisted to localStorage)
  const timerKey = `crosswise-timer-${puzzleId}`;
  const [elapsedSeconds, setElapsedSeconds] = useState(() => {
    const stored = localStorage.getItem(timerKey);
    return stored ? parseInt(stored, 10) || 0 : 0;
  });
  const [timerRunning, setTimerRunning] = useState(false);


  // Editable puzzle name (null = use derived default from puzzle metadata)
  const [nameOverride, setNameOverride] = useState<string | null>(null);
  const [editingName, setEditingName] = useState(false);
  const nameInputRef = useRef<HTMLInputElement>(null);

  // Reference image modal
  const [showRefImage, setShowRefImage] = useState<'original' | 'masked' | null>(null);

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

  // Timer: pause when tab is hidden, resume when visible
  useEffect(() => {
    const handler = () => {
      if (document.hidden) setTimerRunning(false);
      else setTimerRunning(true);
    };
    document.addEventListener('visibilitychange', handler);
    return () => document.removeEventListener('visibilitychange', handler);
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
  useEffect(() => {
    if (progress?.stage === 'clues_ready') {
      refetch(true);
    }
  }, [progress?.stage, refetch]);

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

  const handleCellChange = useCallback((row: number, col: number, char: string) => {
    const key = `${row},${col}`;
    if (char) {
      userGridRef.current[key] = char.toUpperCase();
    } else {
      delete userGridRef.current[key];
    }
  }, []);

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
    if (!user) showToast('Empty cell');
    else if (user === correct) showToast('Correct!');
    else showToast('Incorrect');
  }, [activePosition, getCorrectLetter, showToast]);

  const handleCheckWord = useCallback(() => {
    if (!activeClue || !activeDirection) return;
    const answer = activeClue.answer;
    if (!answer || answer.includes('?')) { showToast('No answer available'); return; }
    const [sr, sc] = activeClue.start;
    let userWord = '';
    let allFilled = true;
    for (let i = 0; i < activeClue.length; i++) {
      const r = activeDirection === 'across' ? sr : sr + i;
      const c = activeDirection === 'across' ? sc + i : sc;
      const letter = userGridRef.current[`${r},${c}`];
      if (letter) userWord += letter;
      else { allFilled = false; break; }
    }
    if (!allFilled) showToast('Fill in all letters first');
    else if (userWord === answer.toUpperCase()) showToast('Correct!');
    else showToast('Not quite — try again');
  }, [activeClue, activeDirection, showToast]);

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
    <div style={styles.container}>
      <Link to="/" style={styles.brandLink}>
        <img src="/crosswise.svg" alt="" style={styles.brandLogo} />
        <h1 style={styles.brand}>Crosswise</h1>
      </Link>

      {/* Solve progress banner (collapsible) */}
      {isSolving && (
        <div style={styles.solveBanner}>
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
              <p style={styles.solveBannerSub}>
                You can start filling in answers while the solver works. Hints will appear when ready.
              </p>
              <button
                style={styles.stopBtn}
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
        <div style={styles.errorBanner}>
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
        <div style={styles.toast}>
          {toast}
          <button style={styles.toastClose} onClick={() => setToast(null)}>&times;</button>
        </div>
      )}

      <div style={styles.header}>
        <Link to="/" style={styles.backLink}>&larr; Puzzles</Link>
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
          onClick={() => setShowRefImage('original')}
          title="View source photos"
        >
          Source Image
        </button>
        <div style={styles.timer}>
          <span style={styles.timerDisplay}>
            {String(Math.floor(elapsedSeconds / 60)).padStart(2, '0')}:
            {String(elapsedSeconds % 60).padStart(2, '0')}
          </span>
          <button
            style={styles.timerToggle}
            onClick={() => setTimerRunning((r) => !r)}
            title={timerRunning ? 'Pause timer' : 'Resume timer'}
          >
            {timerRunning ? '\u23F8' : '\u25B6'}
          </button>
          <button
            style={styles.timerToggle}
            onClick={() => { setElapsedSeconds(0); localStorage.setItem(timerKey, '0'); }}
            title="Reset timer"
          >
            &#8634;
          </button>
        </div>
      </div>

      {/* Reference image modal */}
      {showRefImage && (
        <div style={styles.modalOverlay} onClick={() => setShowRefImage(null)}>
          <div style={styles.modalContent} onClick={(e) => e.stopPropagation()}>
            <div style={styles.modalHeader}>
              <div style={styles.modalTabs}>
                <button
                  style={showRefImage === 'original' ? styles.modalTabActive : styles.modalTab}
                  onClick={() => setShowRefImage('original')}
                >
                  Original
                </button>
                <button
                  style={showRefImage === 'masked' ? styles.modalTabActive : styles.modalTab}
                  onClick={() => setShowRefImage('masked')}
                >
                  Masked
                </button>
              </div>
              <button style={styles.modalClose} onClick={() => setShowRefImage(null)}>&times;</button>
            </div>
            <img
              src={`/api/files/${puzzleId}/${showRefImage === 'masked' ? 'masked' : 'original'}.jpg`}
              alt={showRefImage === 'masked' ? 'Masked crossword photo' : 'Original crossword photo'}
              style={styles.refImage}
            />
          </div>
        </div>
      )}

      <CrosswordProvider
        ref={crosswordRef}
        data={crosswordData}
        theme={crosswordTheme}
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
          onRevealHint={() => activeDirection && activeNumber && revealHint(activeDirection, activeNumber)}
          onRevealExplanation={() => activeDirection && activeNumber && revealExplanation(activeDirection, activeNumber)}
          onCheckLetter={handleCheckLetter}
          onCheckWord={handleCheckWord}
          onCheckPuzzle={handleCheckPuzzle}
          onRevealLetter={handleRevealLetter}
          onRevealWord={handleRevealWord}
          onRevealPuzzle={handleRevealPuzzle}
        />
        <div style={styles.body}>
          <div style={{ width: GRID_HEIGHT, height: gridHeight, flexShrink: 0 }}>
            <CrosswordGrid />
          </div>

          <div style={{ ...styles.cluesRow, height: gridHeight }}>
            <div ref={acrossCluesRef} style={styles.clueBox}>
              <DirectionClues direction="across" />
            </div>
            <div ref={downCluesRef} style={styles.clueBox}>
              <DirectionClues direction="down" />
            </div>
          </div>
        </div>
      </CrosswordProvider>
    </div>
  );
}

const spinnerKeyframes = `
@keyframes crosswise-spin {
  to { transform: rotate(360deg); }
}
`;

// Inject spinner keyframes once
if (typeof document !== 'undefined' && !document.getElementById('crosswise-spin-style')) {
  const style = document.createElement('style');
  style.id = 'crosswise-spin-style';
  style.textContent = spinnerKeyframes;
  document.head.appendChild(style);
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
    fontSize: '16px',
    color: '#444',
    flex: 1,
    cursor: 'text',
    border: '1px solid #d1d5db',
    borderRadius: '4px',
    padding: '2px 8px',
    backgroundColor: '#f9fafb',
  },
  nameInput: {
    fontSize: '16px',
    color: '#333',
    flex: 1,
    border: '1px solid #93c5fd',
    borderRadius: '4px',
    padding: '2px 8px',
    outline: 'none',
    backgroundColor: '#eff6ff',
  },
  refImageBtn: {
    padding: '4px 10px',
    fontSize: '12px',
    border: '1px solid #d1d5db',
    borderRadius: '4px',
    backgroundColor: '#f9fafb',
    cursor: 'pointer',
    color: '#555',
    flexShrink: 0,
  },
  timer: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    flexShrink: 0,
  },
  timerDisplay: {
    fontFamily: '"SF Mono", "Menlo", monospace',
    fontSize: '15px',
    color: '#555',
    minWidth: '48px',
  },
  timerToggle: {
    background: 'none',
    border: 'none',
    fontSize: '14px',
    cursor: 'pointer',
    padding: '2px 4px',
    color: '#888',
    lineHeight: 1,
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
  modalTabs: {
    display: 'flex',
    gap: '4px',
  },
  modalTab: {
    padding: '6px 14px',
    border: '1px solid #d1d5db',
    borderRadius: '4px',
    backgroundColor: '#fff',
    cursor: 'pointer',
    fontSize: '13px',
    color: '#666',
  },
  modalTabActive: {
    padding: '6px 14px',
    border: '1px solid #2563eb',
    borderRadius: '4px',
    backgroundColor: '#eff6ff',
    cursor: 'pointer',
    fontSize: '13px',
    color: '#2563eb',
    fontWeight: 600,
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
};
