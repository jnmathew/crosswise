import { useRef, useState, useEffect, useCallback } from 'react';

interface GridCropperProps {
  imageUrl: string;
  onSubmit: (corners: number[][]) => void;
  onBack: () => void;
  submitting: boolean;
}

const CANVAS_MAX_WIDTH = 800;

export default function GridCropper({ imageUrl, onSubmit, onBack, submitting }: GridCropperProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const scaleRef = useRef(1);

  const [drawing, setDrawing] = useState(false);
  const [startPoint, setStartPoint] = useState<{ x: number; y: number } | null>(null);
  const [currentPoint, setCurrentPoint] = useState<{ x: number; y: number } | null>(null);
  const [rect, setRect] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null);

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

    // Draw finalized rectangle
    if (rect) {
      const rx = rect.x1 / s;
      const ry = rect.y1 / s;
      const rw = (rect.x2 - rect.x1) / s;
      const rh = (rect.y2 - rect.y1) / s;

      // Dim area outside selection
      ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
      ctx.fillRect(0, 0, canvas.width, ry);
      ctx.fillRect(0, ry, rx, rh);
      ctx.fillRect(rx + rw, ry, canvas.width - rx - rw, rh);
      ctx.fillRect(0, ry + rh, canvas.width, canvas.height - ry - rh);

      ctx.strokeStyle = '#2563eb';
      ctx.lineWidth = 2;
      ctx.strokeRect(rx, ry, rw, rh);
    }

    // Draw in-progress rectangle
    if (drawing && startPoint && currentPoint) {
      const x = Math.min(startPoint.x, currentPoint.x);
      const y = Math.min(startPoint.y, currentPoint.y);
      const w = Math.abs(currentPoint.x - startPoint.x);
      const h = Math.abs(currentPoint.y - startPoint.y);

      ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
      ctx.fillRect(0, 0, canvas.width, y);
      ctx.fillRect(0, y, x, h);
      ctx.fillRect(x + w, y, canvas.width - x - w, h);
      ctx.fillRect(0, y + h, canvas.width, canvas.height - y - h);

      ctx.strokeStyle = '#2563eb';
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);
    }
  }, [rect, drawing, startPoint, currentPoint]);

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

  useEffect(() => {
    redraw();
  }, [redraw]);

  const getCanvasPos = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current!;
    const r = canvas.getBoundingClientRect();
    const scaleX = canvas.width / r.width;
    const scaleY = canvas.height / r.height;
    return {
      x: (e.clientX - r.left) * scaleX,
      y: (e.clientY - r.top) * scaleY,
    };
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const pos = getCanvasPos(e);
    setStartPoint(pos);
    setCurrentPoint(pos);
    setDrawing(true);
    setRect(null);
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

    const x1 = Math.min(p1.x, p2.x);
    const y1 = Math.min(p1.y, p2.y);
    const x2 = Math.max(p1.x, p2.x);
    const y2 = Math.max(p1.y, p2.y);

    if (Math.abs(x2 - x1) > 20 && Math.abs(y2 - y1) > 20) {
      setRect({ x1, y1, x2, y2 });
    }

    setDrawing(false);
    setStartPoint(null);
    setCurrentPoint(null);
  };

  const handleSubmit = () => {
    if (!rect) return;
    // Convert rectangle to 4 corner points (TL, TR, BR, BL)
    const corners: number[][] = [
      [rect.x1, rect.y1],
      [rect.x2, rect.y1],
      [rect.x2, rect.y2],
      [rect.x1, rect.y2],
    ];
    onSubmit(corners);
  };

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>Select Grid Manually</h2>
      <p style={styles.subtitle}>
        Draw a rectangle around the crossword grid in the original photo.
      </p>

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

      <div style={styles.actions}>
        <button style={styles.secondaryButton} className="btn-secondary" onClick={onBack}>
          &larr; Back to Grid Editor
        </button>
        {rect && (
          <button style={styles.clearButton} className="btn-clear" onClick={() => setRect(null)}>
            Clear Selection
          </button>
        )}
        <button
          style={{
            ...styles.primaryButton,
            ...((!rect || submitting) ? styles.primaryDisabled : {}),
          }}
          className="btn-primary"
          onClick={handleSubmit}
          disabled={!rect || submitting}
        >
          {submitting ? 'Detecting grid...' : 'Use This Selection'}
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
    width: '100%',
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
  primaryDisabled: {
    opacity: 0.5,
    cursor: 'not-allowed',
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
  clearButton: {
    padding: '10px 24px',
    backgroundColor: '#fff',
    color: '#b91c1c',
    border: '1px solid #fca5a5',
    borderRadius: '6px',
    fontSize: '15px',
    cursor: 'pointer',
  },
};
