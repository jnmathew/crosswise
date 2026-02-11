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
  checkResult?: 'correct' | 'incorrect' | 'incomplete' | null;
  isSolving?: boolean;
  onRevealHint: () => void;
  onRevealExplanation: () => void;
  onRevealAnswer: () => void;
  onCheckWord?: () => void;
}

export default function HintPanel({
  clue,
  direction,
  hintState,
  checkResult,
  isSolving,
  onRevealHint,
  onRevealExplanation,
  onRevealAnswer,
  onCheckWord,
}: HintPanelProps) {
  if (!clue || !direction) {
    return <div style={styles.panel}><p style={styles.placeholder}>Select a clue to see hints</p></div>;
  }

  const hasSolution = clue.answer !== null && !clue.answer.includes('?');
  const hasHint = clue.hint !== null;
  const hasExplanation = clue.explanation !== null;

  return (
    <div style={styles.panel}>
      <div style={styles.clueHeader}>
        <span style={styles.clueNumber}>{clue.number}-{direction}</span>
        <span style={styles.clueLength}>({clue.length} letters)</span>
      </div>
      <p style={styles.clueText}>{clue.text}</p>

      {/* Check Word button — always available when answer is known */}
      {hasSolution && onCheckWord && (
        <div style={styles.section}>
          <button
            style={{ ...styles.button, ...styles.checkButton }}
            onClick={onCheckWord}
          >
            Check Word
          </button>
          {checkResult === 'correct' && (
            <p style={styles.checkCorrect}>Correct!</p>
          )}
          {checkResult === 'incorrect' && (
            <p style={styles.checkIncorrect}>Not quite — try again</p>
          )}
          {checkResult === 'incomplete' && (
            <p style={styles.checkIncomplete}>Fill in all letters first</p>
          )}
        </div>
      )}

      {/* Solving in progress indicator */}
      {!hasSolution && isSolving && (
        <p style={styles.solvingNote}>Solver is working — hints will appear when ready</p>
      )}

      {/* No solution and not solving */}
      {!hasSolution && !isSolving && (
        <p style={styles.unsolved}>Unsolved — no hints available</p>
      )}

      {hasSolution && hasHint && (
        <div style={styles.section}>
          {hintState.hintRevealed ? (
            <p style={styles.revealedText}>{clue.hint}</p>
          ) : (
            <button style={styles.button} onClick={onRevealHint}>
              Get Hint
            </button>
          )}
        </div>
      )}

      {hasSolution && hasExplanation && (
        <div style={styles.section}>
          {hintState.explanationRevealed ? (
            <p style={styles.revealedText}>{clue.explanation}</p>
          ) : (
            <button
              style={{
                ...styles.button,
                ...(hintState.hintRevealed || hintState.answerRevealed
                  ? {}
                  : styles.buttonDisabled),
              }}
              onClick={onRevealExplanation}
              disabled={!hintState.hintRevealed && !hintState.answerRevealed}
            >
              Explain
            </button>
          )}
        </div>
      )}

      {hasSolution && (
        <div style={styles.section}>
          {hintState.answerRevealed ? (
            <p style={styles.answerText}>Answer: {clue.answer}</p>
          ) : (
            <button style={{ ...styles.button, ...styles.revealButton }} onClick={onRevealAnswer}>
              Reveal Answer
            </button>
          )}
        </div>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  panel: {
    padding: '16px',
    border: '1px solid #ddd',
    borderRadius: '8px',
    backgroundColor: '#fafafa',
    minWidth: '280px',
  },
  placeholder: {
    color: '#999',
    fontStyle: 'italic',
  },
  clueHeader: {
    display: 'flex',
    alignItems: 'baseline',
    gap: '8px',
    marginBottom: '4px',
  },
  clueNumber: {
    fontWeight: 'bold',
    color: '#333',
    textTransform: 'uppercase' as const,
  },
  clueLength: {
    fontSize: '12px',
    color: '#888',
  },
  clueText: {
    color: '#444',
    fontSize: '14px',
    lineHeight: '1.4',
    margin: '0 0 12px 0',
  },
  unsolved: {
    color: '#b55',
    fontStyle: 'italic',
    fontSize: '14px',
  },
  solvingNote: {
    color: '#6b7280',
    fontStyle: 'italic',
    fontSize: '13px',
    margin: 0,
  },
  section: {
    marginTop: '8px',
  },
  button: {
    padding: '8px 16px',
    border: '1px solid #aaa',
    borderRadius: '4px',
    backgroundColor: '#fff',
    cursor: 'pointer',
    fontSize: '14px',
    width: '100%',
  },
  checkButton: {
    backgroundColor: '#f0f9ff',
    borderColor: '#7dd3fc',
    color: '#0369a1',
    fontWeight: 500,
  },
  checkCorrect: {
    color: '#15803d',
    fontWeight: 600,
    fontSize: '14px',
    textAlign: 'center' as const,
    margin: '6px 0 0 0',
  },
  checkIncorrect: {
    color: '#b91c1c',
    fontWeight: 500,
    fontSize: '14px',
    textAlign: 'center' as const,
    margin: '6px 0 0 0',
  },
  checkIncomplete: {
    color: '#92400e',
    fontSize: '13px',
    textAlign: 'center' as const,
    margin: '6px 0 0 0',
  },
  buttonDisabled: {
    opacity: 0.4,
    cursor: 'not-allowed',
  },
  revealButton: {
    backgroundColor: '#fee',
    borderColor: '#c88',
  },
  revealedText: {
    backgroundColor: '#eef6ff',
    padding: '8px 12px',
    borderRadius: '4px',
    fontSize: '14px',
    lineHeight: '1.4',
    margin: 0,
  },
  answerText: {
    backgroundColor: '#eeffee',
    padding: '8px 12px',
    borderRadius: '4px',
    fontWeight: 'bold',
    fontSize: '16px',
    letterSpacing: '2px',
    margin: 0,
  },
};
