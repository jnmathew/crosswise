import { useRef, useState, useEffect, useCallback } from 'react';
import type { MaskRect, MaskData } from '../types/api';

type Mode = 'mask' | 'separator';

interface ImageMaskerProps {
  imageUrl: string;
  onSubmit: (data: MaskData) => void;
  submitting: boolean;
  /** Optional URL to a masked example image to show in the help overlay */
  exampleMaskedUrl?: string;
}

const CANVAS_MAX_WIDTH = 800;

export default function ImageMasker({ imageUrl, onSubmit, submitting, exampleMaskedUrl }: ImageMaskerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  const [mode, setMode] = useState<Mode>('mask');
  const [rectangles, setRectangles] = useState<MaskRect[]>([]);
  const [separators, setSeparators] = useState<MaskRect[]>([]);
  const [showHelp, setShowHelp] = useState(false);

  // Drawing state
  const [drawing, setDrawing] = useState(false);
  const [startPoint, setStartPoint] = useState<{ x: number; y: number } | null>(null);
  const [currentPoint, setCurrentPoint] = useState<{ x: number; y: number } | null>(null);

  // Scale factor: image pixels / canvas pixels
  const scaleRef = useRef(1);

  const toImageCoords = useCallback((canvasX: number, canvasY: number) => {
    const s = scaleRef.current;
    return { x: Math.round(canvasX * s), y: Math.round(canvasY * s) };
  }, []);

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;

    const ctx = canvas.getContext('2d')!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const s = scaleRef.current;

    // Draw mask rectangles
    ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.strokeStyle = '#aaa';
    ctx.lineWidth = 1;
    for (const r of rectangles) {
      const x = r.x1 / s;
      const y = r.y1 / s;
      const w = (r.x2 - r.x1) / s;
      const h = (r.y2 - r.y1) / s;
      ctx.fillRect(x, y, w, h);
      ctx.strokeRect(x, y, w, h);
    }

    // Draw separators
    ctx.strokeStyle = '#00ffff';
    ctx.lineWidth = 3;
    for (const sep of separators) {
      ctx.beginPath();
      ctx.moveTo(sep.x1 / s, sep.y1 / s);
      ctx.lineTo(sep.x2 / s, sep.y2 / s);
      ctx.stroke();
    }

    // Draw current in-progress shape
    if (drawing && startPoint && currentPoint) {
      if (mode === 'mask') {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
        ctx.strokeStyle = '#888';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        const x = startPoint.x;
        const y = startPoint.y;
        const w = currentPoint.x - startPoint.x;
        const h = currentPoint.y - startPoint.y;
        ctx.fillRect(x, y, w, h);
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);
      } else {
        ctx.strokeStyle = '#00ffff';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(startPoint.x, startPoint.y);
        ctx.lineTo(currentPoint.x, currentPoint.y);
        ctx.stroke();
        ctx.setLineDash([]);

        // Draw start dot
        ctx.fillStyle = '#00ffff';
        ctx.beginPath();
        ctx.arc(startPoint.x, startPoint.y, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }, [rectangles, separators, drawing, startPoint, currentPoint, mode]);

  // Load image
  useEffect(() => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      imgRef.current = img;
      const canvas = canvasRef.current;
      if (!canvas) return;

      const scale = img.width > CANVAS_MAX_WIDTH ? img.width / CANVAS_MAX_WIDTH : 1;
      scaleRef.current = scale;
      canvas.width = img.width / scale;
      canvas.height = img.height / scale;
      redraw();
    };
    img.src = imageUrl;
  }, [imageUrl]); // eslint-disable-line react-hooks/exhaustive-deps

  // Redraw when shapes change
  useEffect(() => {
    redraw();
  }, [redraw]);

  const getCanvasPos = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const pos = getCanvasPos(e);
    setStartPoint(pos);
    setCurrentPoint(pos);
    setDrawing(true);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!drawing) return;
    setCurrentPoint(getCanvasPos(e));
  };

  const handleMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!drawing || !startPoint) return;
    const end = getCanvasPos(e);

    const p1 = toImageCoords(startPoint.x, startPoint.y);
    const p2 = toImageCoords(end.x, end.y);

    if (mode === 'mask') {
      const rect: MaskRect = {
        x1: Math.min(p1.x, p2.x),
        y1: Math.min(p1.y, p2.y),
        x2: Math.max(p1.x, p2.x),
        y2: Math.max(p1.y, p2.y),
      };
      // Ignore tiny accidental clicks
      if (Math.abs(rect.x2 - rect.x1) > 5 && Math.abs(rect.y2 - rect.y1) > 5) {
        setRectangles((prev) => [...prev, rect]);
      }
    } else {
      const sep: MaskRect = { x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y };
      const dist = Math.hypot(p2.x - p1.x, p2.y - p1.y);
      if (dist > 10) {
        setSeparators((prev) => [...prev, sep]);
      }
    }

    setDrawing(false);
    setStartPoint(null);
    setCurrentPoint(null);
  };

  const handleUndo = () => {
    if (mode === 'mask') {
      setRectangles((prev) => prev.slice(0, -1));
    } else {
      setSeparators((prev) => prev.slice(0, -1));
    }
  };

  const handleClear = () => {
    setRectangles([]);
    setSeparators([]);
  };

  const handleSubmit = () => {
    onSubmit({ rectangles, separators });
  };

  return (
    <div style={styles.container}>
      {/* Help overlay */}
      {showHelp && (
        <div style={styles.helpOverlay} onClick={() => setShowHelp(false)}>
          <div style={styles.helpPanel} onClick={(e) => e.stopPropagation()}>
            <div style={styles.helpHeader}>
              <span style={styles.helpTitle}>How to Mask &amp; Separate</span>
              <button style={styles.helpClose} onClick={() => setShowHelp(false)}>&times;</button>
            </div>
            <div style={styles.helpBody}>
              <p style={styles.helpText}>
                The OCR needs to see <strong>only the clue text</strong> — nothing else.
                Use the two tools below to clean up the photo before submitting.
              </p>

              <div style={styles.helpSection}>
                <h4 style={styles.helpSectionTitle}>1. Mask Mode (white rectangles)</h4>
                <p style={styles.helpText}>
                  Draw rectangles over anything that <em>isn't</em> clue text:
                  the crossword grid, advertisements, page numbers, acrostic puzzles,
                  photos, or other non-clue content. These areas get whited out so
                  the OCR ignores them.
                </p>
              </div>

              <div style={styles.helpSection}>
                <h4 style={styles.helpSectionTitle}>2. Separator Mode (cyan lines)</h4>
                <p style={styles.helpText}>
                  Draw lines between clue columns. Drag from the top of the gap
                  between two columns to the bottom, following the angle of the text.
                  This tells the OCR where one column ends and the next begins.
                  <strong> Tilted lines matching the text angle work best.</strong>
                </p>
              </div>

              {exampleMaskedUrl && (
                <div style={styles.helpSection}>
                  <h4 style={styles.helpSectionTitle}>Example result</h4>
                  <p style={styles.helpText}>
                    Here's what a properly masked image looks like — white rectangles
                    over non-clue areas, cyan lines between columns:
                  </p>
                  <img
                    src={exampleMaskedUrl}
                    alt="Example of masked crossword image"
                    style={styles.helpImage}
                  />
                </div>
              )}

              <div style={styles.helpSection}>
                <h4 style={styles.helpSectionTitle}>Tips</h4>
                <ul style={styles.helpList}>
                  <li>You can switch between Mask and Separator mode at any time</li>
                  <li>Use Undo to remove the last drawn shape</li>
                  <li>If OCR verification fails, come back and adjust your masks</li>
                  <li>Make sure separator lines span the full height of the clue columns</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      <div style={styles.toolbar}>
        <div style={styles.modeGroup}>
          <button
            style={mode === 'mask' ? styles.modeActive : styles.modeButton}
            onClick={() => setMode('mask')}
          >
            Mask ({rectangles.length})
          </button>
          <button
            style={mode === 'separator' ? styles.modeActive : styles.modeButton}
            onClick={() => setMode('separator')}
          >
            Separator ({separators.length})
          </button>
          <button
            style={styles.helpButton}
            onClick={() => setShowHelp(true)}
            title="How to use the masker"
          >
            ?
          </button>
        </div>
        <div style={styles.actionGroup}>
          <button style={styles.actionButton} onClick={handleUndo}>
            Undo
          </button>
          <button style={styles.actionButton} onClick={handleClear}>
            Clear All
          </button>
          <button
            style={styles.submitButton}
            onClick={handleSubmit}
            disabled={submitting}
          >
            {submitting ? 'Processing...' : 'Submit Masks'}
          </button>
        </div>
      </div>

      <div style={styles.hint}>
        {mode === 'mask'
          ? 'Draw rectangles over areas to mask out (grids, ads, etc.)'
          : 'Draw lines between clue columns (drag from top to bottom)'}
      </div>

      <div style={styles.canvasWrap}>
        <canvas
          ref={canvasRef}
          style={styles.canvas}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={() => {
            if (drawing) {
              setDrawing(false);
              setStartPoint(null);
              setCurrentPoint(null);
            }
          }}
        />
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
    width: '100%',
  },
  toolbar: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    width: '100%',
    maxWidth: `${CANVAS_MAX_WIDTH}px`,
  },
  modeGroup: {
    display: 'flex',
    gap: '4px',
  },
  modeButton: {
    padding: '6px 14px',
    border: '1px solid #ccc',
    borderRadius: '4px',
    backgroundColor: '#fff',
    cursor: 'pointer',
    fontSize: '13px',
  },
  modeActive: {
    padding: '6px 14px',
    border: '1px solid #2563eb',
    borderRadius: '4px',
    backgroundColor: '#eff6ff',
    color: '#2563eb',
    cursor: 'pointer',
    fontSize: '13px',
    fontWeight: 600,
  },
  actionGroup: {
    display: 'flex',
    gap: '8px',
  },
  actionButton: {
    padding: '6px 14px',
    border: '1px solid #ccc',
    borderRadius: '4px',
    backgroundColor: '#fff',
    cursor: 'pointer',
    fontSize: '13px',
  },
  submitButton: {
    padding: '6px 18px',
    border: 'none',
    borderRadius: '4px',
    backgroundColor: '#2563eb',
    color: '#fff',
    cursor: 'pointer',
    fontSize: '13px',
    fontWeight: 600,
  },
  helpButton: {
    width: '28px',
    height: '28px',
    borderRadius: '50%',
    border: '1px solid #93c5fd',
    backgroundColor: '#eff6ff',
    color: '#2563eb',
    fontSize: '14px',
    fontWeight: 700,
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 0,
    flexShrink: 0,
  },
  helpOverlay: {
    position: 'fixed' as const,
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.5)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 2000,
  },
  helpPanel: {
    backgroundColor: '#fff',
    borderRadius: '12px',
    maxWidth: '640px',
    maxHeight: '90vh',
    overflow: 'auto',
    boxShadow: '0 8px 32px rgba(0,0,0,0.25)',
    width: '90vw',
  },
  helpHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '16px 20px',
    borderBottom: '1px solid #eee',
  },
  helpTitle: {
    fontWeight: 600,
    fontSize: '17px',
  },
  helpClose: {
    background: 'none',
    border: 'none',
    fontSize: '24px',
    cursor: 'pointer',
    color: '#666',
    padding: '0 4px',
    lineHeight: 1,
  },
  helpBody: {
    padding: '16px 20px 24px',
  },
  helpSection: {
    marginTop: '16px',
  },
  helpSectionTitle: {
    margin: '0 0 6px 0',
    fontSize: '14px',
    fontWeight: 600,
    color: '#333',
  },
  helpText: {
    fontSize: '14px',
    color: '#555',
    lineHeight: '1.5',
    margin: '0 0 8px 0',
  },
  helpList: {
    fontSize: '14px',
    color: '#555',
    lineHeight: '1.6',
    margin: '4px 0 0 0',
    paddingLeft: '20px',
  },
  helpImage: {
    display: 'block',
    maxWidth: '100%',
    borderRadius: '6px',
    border: '1px solid #ddd',
    marginTop: '8px',
  },
  hint: {
    fontSize: '13px',
    color: '#888',
    fontStyle: 'italic',
  },
  canvasWrap: {
    border: '1px solid #ddd',
    borderRadius: '4px',
    overflow: 'hidden',
    maxWidth: '100%',
  },
  canvas: {
    display: 'block',
    maxWidth: '100%',
    cursor: 'crosshair',
  },
};
