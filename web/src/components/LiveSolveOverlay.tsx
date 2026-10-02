import { useMemo } from 'react';
import type { PuzzleData } from '../types/puzzle';
import type { LiveSolveState } from '../hooks/useLiveSolve';

type CellState = 'committed' | 'tentative' | 'rejected';

interface Props {
  puzzle: PuzzleData;
  live: LiveSolveState;
}

/** Cells of a clue ID like "12-across", or [] if the puzzle has no such clue. */
function clueCells(puzzle: PuzzleData, clueId: string): [number, number][] {
  const [num, direction] = clueId.split('-');
  if (direction !== 'across' && direction !== 'down') return [];
  const clue = puzzle.clues[direction].find((c) => String(c.number) === num);
  if (!clue) return [];
  const [row, col] = clue.start;
  return Array.from({ length: clue.length }, (_, i) =>
    direction === 'across' ? [row, col + i] : [row + i, col],
  );
}

/**
 * Read-only view of the solver's grid, drawn over the player's grid while
 * "watch the solve" is on. The player's own entries underneath are untouched.
 */
export default function LiveSolveOverlay({ puzzle, live }: Props) {
  const { rows, cols, cells } = puzzle.grid;

  const letters = useMemo(() => {
    const map = new Map<string, { letter: string; state: CellState }>();
    const place = (answers: Record<string, string>, state: CellState) => {
      for (const [clueId, word] of Object.entries(answers)) {
        clueCells(puzzle, clueId).forEach(([r, c], i) => {
          if (word[i]) map.set(`${r},${c}`, { letter: word[i], state });
        });
      }
    };
    // Later layers win: a fading rejection never hides a real letter
    place(Object.fromEntries(Object.entries(live.rejected).map(([k, v]) => [k, v.word])), 'rejected');
    place(live.committed, 'committed');
    place(live.tentative, 'tentative');
    return map;
  }, [puzzle, live.committed, live.tentative, live.rejected]);

  return (
    <svg
      className="live-overlay"
      viewBox={`0 0 ${cols} ${rows}`}
      role="img"
      aria-label="Live view of the solver filling the grid"
    >
      {cells.flat().map((cell) => {
        const key = `${cell.row},${cell.col}`;
        const entry = letters.get(key);
        return (
          <g key={key}>
            <rect
              x={cell.col}
              y={cell.row}
              width={1}
              height={1}
              className={cell.is_block ? 'live-block' : 'live-cell'}
            />
            {cell.clue_number != null && !cell.is_block && (
              <text x={cell.col + 0.06} y={cell.row + 0.27} className="live-number">
                {cell.clue_number}
              </text>
            )}
            {entry && (
              // Keyed on letter + state so each change remounts and replays the pop animation
              <text
                key={`${entry.letter}-${entry.state}`}
                x={cell.col + 0.5}
                y={cell.row + 0.78}
                className={`live-letter live-${entry.state}`}
              >
                {entry.letter}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}
