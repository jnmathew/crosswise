import { useRef, useState, useEffect, useCallback } from 'react';

interface GridCell {
  row: number;
  col: number;
  is_block: boolean;
  clue_number: number | null;
}

interface GridData {
  rows: number;
  cols: number;
  cells: GridCell[][];
}

interface GridEditorProps {
  sessionId: string;
  warpedGridUrl: string;
  initialSlotCount: number;
  onConfirm: (blackCells: boolean[][]) => void;
  onBack: () => void;
  onManualCrop?: () => void;
  submitting: boolean;
}

const CANVAS_SIZE = 560;
const MIN_DIM = 3;
const MAX_DIM = 30;

export default function GridEditor({
  sessionId,
  warpedGridUrl,
  initialSlotCount,
  onConfirm,
  onBack,
  onManualCrop,
  submitting,
}: GridEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [gridData, setGridData] = useState<GridData | null>(null);
  const [blackCells, setBlackCells] = useState<boolean[][] | null>(null);
  const [slotCount, setSlotCount] = useState(initialSlotCount);
  const [loading, setLoading] = useState(true);
  const [resizing, setResizing] = useState(false);

  // Fetch grid data on mount
  useEffect(() => {
    fetch(`/api/files/${sessionId}/grid_data.json?t=${Date.now()}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: GridData) => {
        setGridData(data);
        const cells = data.cells.map((row) => row.map((c) => c.is_block));
        setBlackCells(cells);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [sessionId, warpedGridUrl]);

  // Load warped grid image
  useEffect(() => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      imgRef.current = img;
      redraw();
    };
    img.src = warpedGridUrl;
  }, [warpedGridUrl]); // eslint-disable-line react-hooks/exhaustive-deps

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !blackCells || !gridData) return;

    const ctx = canvas.getContext('2d')!;
    const { rows, cols } = gridData;

    canvas.width = CANVAS_SIZE;
    canvas.height = CANVAS_SIZE;

    // Draw warped grid as background
    ctx.drawImage(img, 0, 0, CANVAS_SIZE, CANVAS_SIZE);

    const cellW = CANVAS_SIZE / cols;
    const cellH = CANVAS_SIZE / rows;

    // Overlay black cells with semi-transparent red, white cells with nothing
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const x = c * cellW;
        const y = r * cellH;

        if (blackCells[r][c]) {
          ctx.fillStyle = 'rgba(220, 40, 40, 0.35)';
          ctx.fillRect(x, y, cellW, cellH);
          // Draw B label
          ctx.fillStyle = '#c00';
          ctx.font = `bold ${Math.min(cellW, cellH) * 0.35}px sans-serif`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText('B', x + cellW / 2, y + cellH / 2);
        }

        // Draw cell border
        ctx.strokeStyle = 'rgba(0, 100, 255, 0.3)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, cellW, cellH);
      }
    }
  }, [blackCells, gridData]);

  // Redraw when state changes
  useEffect(() => {
    redraw();
  }, [redraw]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!gridData || !blackCells) return;

    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    const col = Math.floor(x / (CANVAS_SIZE / gridData.cols));
    const row = Math.floor(y / (CANVAS_SIZE / gridData.rows));

    if (row >= 0 && row < gridData.rows && col >= 0 && col < gridData.cols) {
      setBlackCells((prev) => {
        if (!prev) return prev;
        const next = prev.map((r) => [...r]);
        next[row][col] = !next[row][col];
        return next;
      });
    }
  };

  const handleResize = async (newRows: number, newCols: number) => {
    if (newRows < MIN_DIM || newRows > MAX_DIM || newCols < MIN_DIM || newCols > MAX_DIM) return;
    setResizing(true);
    try {
      const res = await fetch(`/api/${sessionId}/resize-grid`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rows: newRows, cols: newCols }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        console.error('Resize failed:', err.detail || res.statusText);
        return;
      }
      const data = await res.json();
      setGridData({ rows: newRows, cols: newCols, cells: [] });
      setBlackCells(data.black_cells);
      setSlotCount(data.clue_slot_count);
      // Update gridData with full cell info from black_cells
      const cells: GridCell[][] = data.black_cells.map((row: boolean[], r: number) =>
        row.map((is_block: boolean, c: number) => ({ row: r, col: c, is_block, clue_number: null }))
      );
      setGridData({ rows: newRows, cols: newCols, cells });
    } finally {
      setResizing(false);
    }
  };

  const handleConfirm = () => {
    if (blackCells) {
      onConfirm(blackCells);
    }
  };

  const blackCount = blackCells
    ? blackCells.reduce((sum, row) => sum + row.filter(Boolean).length, 0)
    : 0;

  if (loading) return <div style={styles.loading}>Loading grid data...</div>;
  if (!gridData || !blackCells) return <div style={styles.error}>Failed to load grid data</div>;

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>Verify Black Cells</h2>
      <p style={styles.subtitle}>
        Click cells to toggle black/white. Use +/- to adjust grid dimensions.
      </p>

      <div style={styles.dimControls}>
        <div style={styles.dimGroup}>
          <button
            style={styles.dimButton}
            onClick={() => handleResize(gridData.rows - 1, gridData.cols)}
            disabled={resizing || gridData.rows <= MIN_DIM}
          >
            &minus;
          </button>
          <span style={styles.dimLabel}>{gridData.rows} rows</span>
          <button
            style={styles.dimButton}
            onClick={() => handleResize(gridData.rows + 1, gridData.cols)}
            disabled={resizing || gridData.rows >= MAX_DIM}
          >
            +
          </button>
        </div>
        <div style={styles.dimGroup}>
          <button
            style={styles.dimButton}
            onClick={() => handleResize(gridData.rows, gridData.cols - 1)}
            disabled={resizing || gridData.cols <= MIN_DIM}
          >
            &minus;
          </button>
          <span style={styles.dimLabel}>{gridData.cols} cols</span>
          <button
            style={styles.dimButton}
            onClick={() => handleResize(gridData.rows, gridData.cols + 1)}
            disabled={resizing || gridData.cols >= MAX_DIM}
          >
            +
          </button>
        </div>
        {resizing && <span style={styles.resizingLabel}>Resizing...</span>}
      </div>

      <div style={styles.stats}>
        <span>{gridData.rows} &times; {gridData.cols} grid</span>
        <span>{blackCount} black cells</span>
        <span>{slotCount} clue slots</span>
      </div>

      <div style={styles.canvasWrap}>
        <canvas
          ref={canvasRef}
          style={styles.canvas}
          onClick={handleCanvasClick}
        />
      </div>

      <div style={styles.actions}>
        <button style={styles.secondaryButton} onClick={onBack}>
          &larr; Re-upload
        </button>
        {onManualCrop && (
          <button style={styles.cropButton} onClick={onManualCrop}>
            Select grid manually
          </button>
        )}
        <button
          style={styles.primaryButton}
          onClick={handleConfirm}
          disabled={submitting || resizing}
        >
          {submitting ? 'Saving...' : 'Confirm Grid'}
        </button>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '12px',
  },
  title: {
    margin: 0,
    fontSize: '20px',
    fontWeight: 600,
  },
  subtitle: {
    margin: 0,
    fontSize: '14px',
    color: '#666',
    maxWidth: '500px',
    textAlign: 'center' as const,
  },
  dimControls: {
    display: 'flex',
    gap: '24px',
    alignItems: 'center',
  },
  dimGroup: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  dimButton: {
    width: '32px',
    height: '32px',
    border: '1px solid #ccc',
    borderRadius: '6px',
    backgroundColor: '#fff',
    fontSize: '18px',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  dimLabel: {
    fontSize: '15px',
    fontWeight: 600,
    minWidth: '60px',
    textAlign: 'center' as const,
  },
  resizingLabel: {
    fontSize: '13px',
    color: '#888',
    fontStyle: 'italic',
  },
  stats: {
    display: 'flex',
    gap: '24px',
    fontSize: '14px',
    color: '#555',
  },
  canvasWrap: {
    border: '1px solid #ddd',
    borderRadius: '4px',
    overflow: 'hidden',
  },
  canvas: {
    display: 'block',
    width: `${CANVAS_SIZE}px`,
    height: `${CANVAS_SIZE}px`,
    cursor: 'pointer',
  },
  actions: {
    display: 'flex',
    gap: '12px',
    marginTop: '8px',
  },
  primaryButton: {
    padding: '10px 24px',
    backgroundColor: '#2563eb',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    fontSize: '15px',
    cursor: 'pointer',
  },
  secondaryButton: {
    padding: '10px 24px',
    backgroundColor: '#fff',
    color: '#555',
    border: '1px solid #ccc',
    borderRadius: '6px',
    fontSize: '15px',
    cursor: 'pointer',
  },
  cropButton: {
    padding: '10px 24px',
    backgroundColor: '#fff',
    color: '#b45309',
    border: '1px solid #fbbf24',
    borderRadius: '6px',
    fontSize: '15px',
    cursor: 'pointer',
  },
  loading: {
    textAlign: 'center',
    padding: '40px',
    color: '#666',
  },
  error: {
    textAlign: 'center',
    padding: '40px',
    color: '#c44',
  },
};
