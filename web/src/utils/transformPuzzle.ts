import type { PuzzleData, PuzzleClue } from '../types/puzzle';

interface CrosswordClueInput {
  clue: string;
  answer: string;
  row: number;
  col: number;
}

export interface CrosswordData {
  across: Record<string, CrosswordClueInput>;
  down: Record<string, CrosswordClueInput>;
}

function transformClues(clues: PuzzleClue[]): Record<string, CrosswordClueInput> {
  const result: Record<string, CrosswordClueInput> = {};
  for (const clue of clues) {
    const answer = clue.answer ?? '?'.repeat(clue.length);
    result[String(clue.number)] = {
      clue: clue.text,
      answer: answer.toUpperCase(),
      row: clue.start[0],
      col: clue.start[1],
    };
  }
  return result;
}

export function transformPuzzle(puzzle: PuzzleData): CrosswordData {
  return {
    across: transformClues(puzzle.clues.across),
    down: transformClues(puzzle.clues.down),
  };
}
