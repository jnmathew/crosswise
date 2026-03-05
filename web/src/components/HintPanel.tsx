import type { PuzzleClue } from '../types/puzzle';

type Direction = 'across' | 'down';

interface ClueHintState {
  hintRevealed: boolean;
  explanationRevealed: boolean;
  answerRevealed: boolean;
}

interface HintPanelProps {
  clue: PuzzleClue | null;
  direction: Direction | null;
  hintState: ClueHintState;
  isSolving?: boolean;
  onRevealHint: () => void;
  onRevealExplanation: () => void;
  onCheckLetter: () => void;
  onCheckWord: () => void;
  onCheckPuzzle: () => void;
  onRevealLetter: () => void;
  onRevealWord: () => void;
  onRevealPuzzle: () => void;
  onClearLetter: () => void;
  onClearWord: () => void;
  onClearPuzzle: () => void;
}

export default function HintPanel({
  clue,
  direction,
  hintState,
  isSolving,
  onRevealHint,
  onRevealExplanation,
  onCheckLetter,
  onCheckWord,
  onCheckPuzzle,
  onRevealLetter,
  onRevealWord,
  onRevealPuzzle,
  onClearLetter,
  onClearWord,
  onClearPuzzle,
}: HintPanelProps) {
  const hasSolution = clue && direction && clue.answer !== null && !clue.answer.includes('?');
  const hasHint = clue?.hint !== null;
  const hasExplanation = clue?.explanation !== null;

  return (
    <div style={styles.bar} className="hint-bar">
      {/* Left: clue text + revealed hints */}
      <div style={styles.clueArea}>
        {!clue || !direction ? (
          <span style={styles.placeholder}>Select a clue</span>
        ) : (
          <>
            <span style={styles.clueText}>
              <strong>{clue.number}{direction === 'across' ? 'A' : 'D'}</strong>
              {' '}{clue.text}
            </span>
            {hintState.hintRevealed && hasHint && (
              <span style={styles.hintText}>{clue.hint}</span>
            )}
            {hintState.explanationRevealed && hasExplanation && (
              <span style={styles.hintText}>{clue.explanation}</span>
            )}
            {hintState.answerRevealed && (
              <span style={styles.answerText}>{clue.answer}</span>
            )}
          </>
        )}
      </div>

      {/* Right: action buttons */}
      <div style={styles.actions} className="hint-actions">
        {!hasSolution && clue && direction && isSolving && (
          <span style={styles.solvingNote}>Solving...</span>
        )}

        {hasSolution && (
          <>
            {/* Hint / Explain */}
            <div style={styles.hintButtons}>
              {hasHint && !hintState.hintRevealed && (
                <button style={styles.btn} className="btn-hint" onClick={onRevealHint}>Hint</button>
              )}
              {hasExplanation && !hintState.explanationRevealed && (
                <button
                  style={{
                    ...styles.btn,
                    ...(hintState.hintRevealed || hintState.answerRevealed ? {} : styles.btnDisabled),
                  }}
                  className="btn-hint"
                  onClick={onRevealExplanation}
                  disabled={!hintState.hintRevealed && !hintState.answerRevealed}
                >
                  Explain
                </button>
              )}
            </div>

            {/* Check / Reveal stacked */}
            <div style={styles.checkRevealStack}>
              <div style={styles.actionGroup}>
                <span style={styles.actionLabel}>Check</span>
                <div style={styles.segmented} className="segmented">
                  <button style={styles.segBtn} className="btn-seg" onClick={onCheckLetter}>Letter</button>
                  <button style={styles.segBtn} className="btn-seg" onClick={onCheckWord}>Word</button>
                  <button style={styles.segBtn} className="btn-seg" onClick={onCheckPuzzle}>Puzzle</button>
                </div>
              </div>
              <div style={styles.actionGroup}>
                <span style={styles.actionLabel}>Reveal</span>
                <div style={styles.segmented} className="segmented">
                  <button style={styles.segBtn} className="btn-seg" onClick={onRevealLetter}>Letter</button>
                  <button style={styles.segBtn} className="btn-seg" onClick={onRevealWord}>Word</button>
                  <button style={styles.segBtn} className="btn-seg" onClick={onRevealPuzzle}>Puzzle</button>
                </div>
              </div>
              <div style={styles.actionGroup}>
                <span style={styles.actionLabel}>Clear</span>
                <div style={styles.segmented} className="segmented">
                  <button style={styles.segBtn} className="btn-seg" onClick={onClearLetter}>Letter</button>
                  <button style={styles.segBtn} className="btn-seg" onClick={onClearWord}>Word</button>
                  <button style={styles.segBtn} className="btn-seg" onClick={onClearPuzzle}>Puzzle</button>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  bar: {
    display: 'flex',
    alignItems: 'center',
    gap: '16px',
    padding: '12px 16px',
    backgroundColor: '#f0f4f8',
    borderRadius: '8px',
    marginBottom: '12px',
    minHeight: '48px',
    width: '100%',
    maxWidth: '1100px',
  },
  clueArea: {
    flex: 1,
    minWidth: 0,
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  placeholder: {
    color: '#999',
    fontStyle: 'italic',
    fontSize: '16px',
  },
  clueText: {
    fontSize: '17px',
    color: '#333',
  },
  hintText: {
    fontSize: '13px',
    color: '#4b5563',
    fontStyle: 'italic',
  },
  answerText: {
    fontSize: '14px',
    fontWeight: 'bold',
    color: '#15803d',
    letterSpacing: '2px',
  },
  solvingNote: {
    color: '#6b7280',
    fontStyle: 'italic',
    fontSize: '13px',
    whiteSpace: 'nowrap' as const,
  },
  actions: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    flexShrink: 0,
  },
  hintButtons: {
    display: 'flex',
    gap: '6px',
  },
  btn: {
    padding: '5px 12px',
    border: '1px solid #9ca3af',
    borderRadius: '4px',
    backgroundColor: '#fff',
    cursor: 'pointer',
    fontSize: '12px',
    whiteSpace: 'nowrap' as const,
  },
  btnDisabled: {
    opacity: 0.4,
    cursor: 'not-allowed',
  },
  checkRevealStack: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  actionGroup: {
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
  },
  actionLabel: {
    fontSize: '11px',
    color: '#666',
    fontWeight: 600,
    width: '38px',
    textAlign: 'right' as const,
    flexShrink: 0,
  },
  segmented: {
    display: 'flex',
    border: '1px solid #d1d5db',
    borderRadius: '4px',
    overflow: 'hidden',
  },
  segBtn: {
    padding: '5px 10px',
    border: 'none',
    borderRight: '1px solid #d1d5db',
    backgroundColor: '#fff',
    cursor: 'pointer',
    fontSize: '11px',
    color: '#444',
    whiteSpace: 'nowrap' as const,
  },
};
