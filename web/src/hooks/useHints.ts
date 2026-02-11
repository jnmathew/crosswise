import { useReducer, useCallback } from 'react';

type Direction = 'across' | 'down';

interface ClueHintState {
  hintRevealed: boolean;
  explanationRevealed: boolean;
  answerRevealed: boolean;
}

interface HintState {
  [key: string]: ClueHintState;
}

type HintAction =
  | { type: 'REVEAL_HINT'; direction: Direction; number: number }
  | { type: 'REVEAL_EXPLANATION'; direction: Direction; number: number }
  | { type: 'REVEAL_ANSWER'; direction: Direction; number: number };

function makeKey(direction: Direction, number: number): string {
  return `${number}-${direction}`;
}

function hintReducer(state: HintState, action: HintAction): HintState {
  const key = makeKey(action.direction, action.number);
  const current = state[key] ?? { hintRevealed: false, explanationRevealed: false, answerRevealed: false };

  switch (action.type) {
    case 'REVEAL_HINT':
      return { ...state, [key]: { ...current, hintRevealed: true } };
    case 'REVEAL_EXPLANATION':
      return { ...state, [key]: { ...current, explanationRevealed: true } };
    case 'REVEAL_ANSWER':
      return { ...state, [key]: { ...current, answerRevealed: true, explanationRevealed: true } };
    default:
      return state;
  }
}

export function useHints() {
  const [state, dispatch] = useReducer(hintReducer, {});

  const revealHint = useCallback((direction: Direction, number: number) => {
    dispatch({ type: 'REVEAL_HINT', direction, number });
  }, []);

  const revealExplanation = useCallback((direction: Direction, number: number) => {
    dispatch({ type: 'REVEAL_EXPLANATION', direction, number });
  }, []);

  const revealAnswer = useCallback((direction: Direction, number: number) => {
    dispatch({ type: 'REVEAL_ANSWER', direction, number });
  }, []);

  const getClueState = useCallback(
    (direction: Direction, number: number): ClueHintState => {
      const key = makeKey(direction, number);
      return state[key] ?? { hintRevealed: false, explanationRevealed: false, answerRevealed: false };
    },
    [state]
  );

  return { revealHint, revealExplanation, revealAnswer, getClueState };
}
