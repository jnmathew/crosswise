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
}: {
  onClueChange: (direction: Direction, number: string) => void;
}) {
  const { selectedDirection, selectedNumber } = useContext(CrosswordContext);

  useEffect(() => {
    if (selectedDirection && selectedNumber) {
      onClueChange(selectedDirection, selectedNumber);
    }
  }, [selectedDirection, selectedNumber, onClueChange]);

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
  const [toast, setToast] = useState<string | null>(null);
  const [checkResult, setCheckResult] = useState<'correct' | 'incorrect' | 'incomplete' | null>(null);

  // Editable puzzle name (null = use derived default from puzzle metadata)
  const [nameOverride, setNameOverride] = useState<string | null>(null);
  const [editingName, setEditingName] = useState(false);
  const nameInputRef = useRef<HTMLInputElement>(null);

  // Reference image modal
  const [showRefImage, setShowRefImage] = useState<'original' | 'masked' | null>(null);

  // Togglable correct counter
  const [showCounter, setShowCounter] = useState(() => {
    const stored = localStorage.getItem('crosswise-show-counter');
    return stored === 'true'; // default to hidden
  });

  // Collapsible solve banner
  const [bannerExpanded, setBannerExpanded] = useState(true);

  // Solve diagnostics
  type DiagCandidate = { word: string; source: string; confidence: number; verified: boolean };
  type DiagClue = {
    clue_id: string; text: string; length: number; category: string | null;
    candidates: DiagCandidate[]; candidate_count: number;
    assigned_answer: string | null; status: 'solved' | 'unsolved' | 'no_candidates';
  };
  const [diagnostics, setDiagnostics] = useState<DiagClue[] | null>(null);
  const [diagOpen, setDiagOpen] = useState(false);
  const [diagExpandedClue, setDiagExpandedClue] = useState<string | null>(null);

  // Refs for auto-scrolling clue lists
  const acrossCluesRef = useRef<HTMLDivElement>(null);
  const downCluesRef = useRef<HTMLDivElement>(null);

  // Track user cell input for "Check Word" feature and correct count
  const userGridRef = useRef<Record<string, string>>({});
  const [userCorrectCount, setUserCorrectCount] = useState(0);

  const puzzleName = nameOverride ?? (puzzle?.metadata.name || puzzleId.replace('IMG_', 'Puzzle #'));

  // Check if puzzle is still being solved (has unsolved clues)
  const isSolving = useMemo(() => {
    if (!puzzle) return false;
    const allClues = [...puzzle.clues.across, ...puzzle.clues.down];
    const hasAnyAnswer = allClues.some((c) => c.answer !== null && !c.answer.includes('?'));
    const allSolved = allClues.every((c) => c.answer !== null && !c.answer.includes('?'));
    // If no clue has an answer, a solve is likely in progress
    return !hasAnyAnswer && !allSolved;
  }, [puzzle]);

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
      refetch();
      const timer = showToast('Puzzle solved! Hints are now available.');
      return () => clearTimeout(timer);
    }
    if (solveDone && progress?.stage === 'failed') {
      refetch();
      const timer = showToast('Solve finished with partial results.');
      return () => clearTimeout(timer);
    }
  }, [solveDone, progress?.stage, refetch, showToast]);

  // Fetch solve diagnostics, guarding against non-JSON responses
  const fetchDiagnostics = useCallback(() => {
    if (!puzzleId) return;
    fetch(`/api/${puzzleId}/diagnostics`)
      .then((r) => {
        const ct = r.headers.get('content-type') || '';
        if (r.ok && ct.includes('application/json')) return r.json();
        return null;
      })
      .then((data) => { if (data) setDiagnostics(data); })
      .catch(() => {});
  }, [puzzleId]);

  // Fetch diagnostics once solve completes
  useEffect(() => {
    if (solveDone && progress?.stage === 'complete') fetchDiagnostics();
  }, [solveDone, progress?.stage, fetchDiagnostics]);

  // Also try to load diagnostics on mount (for already-solved puzzles)
  useEffect(() => { fetchDiagnostics(); }, [fetchDiagnostics]);

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
  // from ANY source: cell click, clue click, keyboard navigation
  const handleClueChange = useCallback((direction: Direction, number: string) => {
    const num = parseInt(number, 10);
    setActiveDirection(direction);
    setActiveNumber(num);
    setCheckResult(null);
    scrollClueIntoView(direction, num);
  }, [scrollClueIntoView]);

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

  const totalClues = puzzle ? puzzle.clues.across.length + puzzle.clues.down.length : 0;

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

  // Recount how many words the user has filled in correctly
  const recomputeCorrectCount = useCallback(() => {
    if (!puzzle) return;
    let correct = 0;
    const allClues = [...puzzle.clues.across, ...puzzle.clues.down];
    for (const clue of allClues) {
      if (!clue.answer || clue.answer.includes('?')) continue;
      const dir = puzzle.clues.across.includes(clue) ? 'across' : 'down';
      const [startRow, startCol] = clue.start;
      let word = '';
      let allFilled = true;
      for (let i = 0; i < clue.length; i++) {
        const r = dir === 'across' ? startRow : startRow + i;
        const c = dir === 'across' ? startCol + i : startCol;
        const letter = userGridRef.current[`${r},${c}`];
        if (letter) {
          word += letter;
        } else {
          allFilled = false;
          break;
        }
      }
      if (allFilled && word === clue.answer.toUpperCase()) {
        correct++;
      }
    }
    setUserCorrectCount(correct);
  }, [puzzle]);

  const handleCellChange = useCallback((row: number, col: number, char: string) => {
    const key = `${row},${col}`;
    if (char) {
      userGridRef.current[key] = char.toUpperCase();
    } else {
      delete userGridRef.current[key];
    }
    recomputeCorrectCount();
  }, [recomputeCorrectCount]);

  const handleAnswerCorrect = useCallback(() => {
    recomputeCorrectCount();
  }, [recomputeCorrectCount]);

  // Check if the user's current word matches the answer
  const handleCheckWord = useCallback(() => {
    if (!activeClue || !activeDirection || activeNumber == null) return;
    const answer = activeClue.answer;
    if (!answer || answer.includes('?')) return;

    const [startRow, startCol] = activeClue.start;
    let userWord = '';
    let allFilled = true;

    for (let i = 0; i < activeClue.length; i++) {
      const row = activeDirection === 'across' ? startRow : startRow + i;
      const col = activeDirection === 'across' ? startCol + i : startCol;
      const letter = userGridRef.current[`${row},${col}`];
      if (letter) {
        userWord += letter;
      } else {
        allFilled = false;
        userWord += ' ';
      }
    }

    if (!allFilled) {
      setCheckResult('incomplete');
    } else if (userWord === answer.toUpperCase()) {
      setCheckResult('correct');
    } else {
      setCheckResult('incorrect');
    }

    // Clear result after 3 seconds
    setTimeout(() => setCheckResult(null), 3000);
  }, [activeClue, activeDirection, activeNumber]);

  const handleRevealAnswer = useCallback(() => {
    if (!activeDirection || !activeNumber || !activeClue?.answer || !crosswordRef.current) return;
    revealAnswer(activeDirection, activeNumber);

    const answer = activeClue.answer!;
    const [startRow, startCol] = activeClue.start;
    for (let i = 0; i < answer.length; i++) {
      const row = activeDirection === 'across' ? startRow : startRow + i;
      const col = activeDirection === 'across' ? startCol + i : startCol;
      crosswordRef.current.setGuess(row, col, answer[i].toUpperCase());
    }
  }, [activeDirection, activeNumber, activeClue, revealAnswer]);

  const savePuzzleName = useCallback((name: string) => {
    setNameOverride(name);
    setEditingName(false);
    fetch(`/api/puzzles/${puzzleId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
  }, [puzzleId]);

  const toggleCounter = useCallback(() => {
    setShowCounter((prev) => {
      const next = !prev;
      localStorage.setItem('crosswise-show-counter', String(next));
      return next;
    });
  }, []);

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

  const clueScrollHeight = (GRID_HEIGHT - 8) / 2; // 8px gap between the two boxes

  return (
    <div style={styles.container}>
      <Link to="/" style={styles.brandLink}>
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
            </>
          )}
        </div>
      )}

      {/* Solve diagnostics panel */}
      {diagnostics && (() => {
        const solvedCount = diagnostics.filter((d) => d.status === 'solved').length;
        const unsolved = diagnostics.filter((d) => d.status !== 'solved');
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
                {unsolved.length > 0 && (
                  <div style={{ marginBottom: '8px' }}>
                    <strong>Unsolved clues:</strong>
                    {unsolved.map((d) => (
                      <div key={d.clue_id} style={styles.diagClue}>
                        <div
                          style={styles.diagClueHeader}
                          onClick={() => setDiagExpandedClue(
                            diagExpandedClue === d.clue_id ? null : d.clue_id
                          )}
                        >
                          <span><strong>{d.clue_id}</strong>: {d.text} ({d.length} letters)</span>
                          <span style={{ fontSize: '11px', color: '#999' }}>
                            {d.candidate_count === 0
                              ? 'no candidates'
                              : `${d.candidate_count} candidates`}
                            {' '}{diagExpandedClue === d.clue_id ? '\u25B2' : '\u25BC'}
                          </span>
                        </div>
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
                    ))}
                  </div>
                )}
                <details>
                  <summary style={{ cursor: 'pointer', fontSize: '13px', color: '#666' }}>
                    All clues ({diagnostics.length})
                  </summary>
                  {diagnostics.map((d) => (
                    <div key={d.clue_id} style={styles.diagClue}>
                      <div
                        style={styles.diagClueHeader}
                        onClick={() => setDiagExpandedClue(
                          diagExpandedClue === d.clue_id ? null : d.clue_id
                        )}
                      >
                        <span>
                          <strong>{d.clue_id}</strong>: {d.text} ({d.length})
                          {d.assigned_answer && (
                            <span style={{ color: '#16a34a', marginLeft: '6px' }}>
                              = {d.assigned_answer}
                            </span>
                          )}
                        </span>
                        <span style={{ fontSize: '11px', color: '#999' }}>
                          {d.candidate_count} cands {diagExpandedClue === d.clue_id ? '\u25B2' : '\u25BC'}
                        </span>
                      </div>
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
                  ))}
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
          Photo
        </button>
        <div style={styles.progress} onClick={toggleCounter} title="Click to toggle">
          {showCounter && (
            <span style={styles.correctBadge}>{userCorrectCount}/{totalClues} correct</span>
          )}
          {!showCounter && (
            <span style={styles.counterHidden}>&#x2022;&#x2022;&#x2022;</span>
          )}
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
        onAnswerCorrect={handleAnswerCorrect}
      >
        <ClueTracker onClueChange={handleClueChange} />
        <div style={styles.body}>
          <div style={{ width: GRID_HEIGHT, height: GRID_HEIGHT, flexShrink: 0 }}>
            <CrosswordGrid />
          </div>

          <div style={styles.cluesColumn}>
            <div ref={acrossCluesRef} style={{ ...styles.clueBox, height: clueScrollHeight }}>
              <DirectionClues direction="across" />
            </div>
            <div ref={downCluesRef} style={{ ...styles.clueBox, height: clueScrollHeight }}>
              <DirectionClues direction="down" />
            </div>
          </div>

          <div style={styles.hintContainer}>
            <HintPanel
              clue={activeClue}
              direction={activeDirection}
              hintState={hintState}
              checkResult={checkResult}
              isSolving={isSolving}
              onRevealHint={() => activeDirection && activeNumber && revealHint(activeDirection, activeNumber)}
              onRevealExplanation={() => activeDirection && activeNumber && revealExplanation(activeDirection, activeNumber)}
              onRevealAnswer={handleRevealAnswer}
              onCheckWord={handleCheckWord}
            />
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
  },
  brand: {
    margin: '0 0 16px 0',
    fontSize: '42px',
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
    cursor: 'pointer',
    borderBottom: '1px dashed transparent',
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
  progress: {
    fontSize: '14px',
    color: '#666',
    cursor: 'pointer',
    userSelect: 'none' as const,
    flexShrink: 0,
  },
  correctBadge: {
    backgroundColor: '#f0fdf4',
    border: '1px solid #bbf7d0',
    borderRadius: '12px',
    padding: '4px 12px',
    fontSize: '13px',
    fontWeight: 600,
    color: '#15803d',
  },
  counterHidden: {
    color: '#ccc',
    fontSize: '16px',
    letterSpacing: '2px',
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
  cluesColumn: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
    width: '280px',
    flexShrink: 0,
  },
  clueBox: {
    overflowY: 'auto',
    border: '1px solid #ddd',
    borderRadius: '6px',
    padding: '8px 12px',
    fontSize: '14px',
  },
  hintContainer: {
    width: '280px',
    flexShrink: 0,
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
