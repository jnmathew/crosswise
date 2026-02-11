export interface PuzzleCell {
  row: number;
  col: number;
  is_block: boolean;
  clue_number: number | null;
}

export interface PuzzleClue {
  number: number;
  text: string;
  start: [number, number]; // [row, col]
  length: number;
  answer: string | null;
  hint: string | null;
  explanation: string | null;
}

export interface PuzzleGrid {
  rows: number;
  cols: number;
  cells: PuzzleCell[][];
}

export interface PuzzleMetadata {
  source_image: string;
  grid_size: [number, number];
  total_clues: number;
  verification: string;
  name?: string;
}

export interface PuzzleData {
  metadata: PuzzleMetadata;
  grid: PuzzleGrid;
  clues: {
    across: PuzzleClue[];
    down: PuzzleClue[];
  };
}
