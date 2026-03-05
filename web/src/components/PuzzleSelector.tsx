import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import type { PuzzleListEntry } from '../types/api';

function GridThumbnail({ mask, size }: { mask: boolean[][]; size: number }) {
  const rows = mask.length;
  const cols = mask[0]?.length || 0;
  if (!rows || !cols) return null;
  const cellSize = size / Math.max(rows, cols);
  const w = cols * cellSize;
  const h = rows * cellSize;
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ flexShrink: 0, borderRadius: '4px', border: '1px solid #ccc' }}>
      <rect width={w} height={h} fill="#fff" />
      {mask.map((row, r) =>
        row.map((isBlock, c) =>
          isBlock ? (
            <rect key={`${r}-${c}`} x={c * cellSize} y={r * cellSize} width={cellSize} height={cellSize} fill="#333" />
          ) : null
        )
      )}
      {/* Grid lines */}
      {Array.from({ length: rows + 1 }, (_, r) => (
        <line key={`h${r}`} x1={0} y1={r * cellSize} x2={w} y2={r * cellSize} stroke="#ccc" strokeWidth={0.5} />
      ))}
      {Array.from({ length: cols + 1 }, (_, c) => (
        <line key={`v${c}`} x1={c * cellSize} y1={0} x2={c * cellSize} y2={h} stroke="#ccc" strokeWidth={0.5} />
      ))}
    </svg>
  );
}

export default function PuzzleSelector() {
  const [puzzles, setPuzzles] = useState<PuzzleListEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [solvingId, setSolvingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    fetch('/api/puzzles')
      .then((res) => res.json())
      .then((data) => {
        setPuzzles(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const handleDelete = async (e: React.MouseEvent, puzzleId: string) => {
    e.preventDefault();
    e.stopPropagation();
    if (!window.confirm('Delete this puzzle? This cannot be undone.')) return;
    try {
      const res = await fetch(`/api/puzzles/${puzzleId}`, { method: 'DELETE' });
      if (res.ok) {
        setPuzzles((prev) => prev.filter((p) => p.id !== puzzleId));
      } else {
        setError('Failed to delete puzzle');
      }
    } catch {
      setError('Failed to delete puzzle');
    }
  };

  const handleResolve = async (e: React.MouseEvent, puzzleId: string) => {
    e.preventDefault();
    e.stopPropagation();
    setSolvingId(puzzleId);
    try {
      const res = await fetch(`/api/${puzzleId}/solve`, { method: 'POST' });
      if (res.ok) {
        navigate(`/puzzle/${puzzleId}`);
      } else {
        const data = await res.json().catch(() => ({}));
        setError(data.detail || 'Failed to start solve');
        setSolvingId(null);
      }
    } catch {
      setError('Failed to start solve');
      setSolvingId(null);
    }
  };

  return (
    <div style={styles.container} className="selector-container page-fade">
      <div style={styles.brandHeader}>
        <img src="/crosswise.svg" alt="" style={styles.brandLogo} />
        <h1 style={styles.heading}>Crosswise</h1>
      </div>
      <p style={styles.subtitle}>Select a puzzle to play</p>

      {error && (
        <div style={styles.errorBanner} className="error-banner">
          {error}
          <button style={styles.errorClose} onClick={() => setError(null)}>&times;</button>
        </div>
      )}

      {loading && (
        <div style={styles.grid} className="puzzle-grid">
          {[0, 1, 2].map((i) => (
            <div key={i} style={styles.skeletonCard} className="skeleton-pulse">
              <div style={styles.skeletonTitle} />
              <div style={styles.skeletonMeta} />
              <div style={styles.skeletonId} />
            </div>
          ))}
        </div>
      )}

      <div style={styles.grid} className="puzzle-grid">
        <Link to="/upload" style={styles.uploadCard} className="puzzle-card upload-tile">
          <div style={styles.uploadCardInner}>
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ opacity: 0.5 }}>
              <line x1="12" y1="5" x2="12" y2="19" />
              <line x1="5" y1="12" x2="19" y2="12" />
            </svg>
            <span style={styles.uploadCardText}>Upload Puzzle</span>
          </div>
        </Link>
        {puzzles.map((p) => (
          <Link to={`/puzzle/${p.id}`} key={p.id} style={styles.card} className="puzzle-card">
            <div style={styles.cardBody}>
              {p.gridMask.length > 0 && (
                <GridThumbnail mask={p.gridMask} size={72} />
              )}
              <div style={styles.cardInfo}>
                <h2 style={styles.cardTitle}>{p.title}</h2>
                <div style={styles.meta}>
                  <span>{p.gridSize[0]} &times; {p.gridSize[1]}</span>
                  <span>{p.totalClues} clues</span>
                  <span
                    style={{
                      color: p.solved === p.totalClues ? '#2a7' : '#b85',
                    }}
                  >
                    {p.solved}/{p.totalClues} solved
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '4px' }}>
                  <div style={styles.id}>{p.id}</div>
                  <div style={{ display: 'flex', gap: '6px' }}>
                    {p.solved < p.totalClues && (
                      <button
                        style={styles.resolveBtn}
                        className="btn-resolve"
                        onClick={(e) => handleResolve(e, p.id)}
                        disabled={solvingId === p.id}
                      >
                        {solvingId === p.id ? 'Starting...' : 'Re-solve'}
                      </button>
                    )}
                    <button
                      style={styles.deleteBtn}
                      className="btn-clear"
                      onClick={(e) => handleDelete(e, p.id)}
                      title="Delete puzzle"
                    >
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ display: 'block' }}><polyline points="3 6 5 6 21 6" /><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6" /><path d="M10 11v6" /><path d="M14 11v6" /><path d="M9 6V4a1 1 0 011-1h4a1 1 0 011 1v2" /></svg>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </Link>
        ))}
      </div>

      {!loading && puzzles.length === 0 && (
        <div style={styles.emptyState}>
          <svg width="80" height="80" viewBox="0 0 100 100" fill="none" style={{ marginBottom: '16px', opacity: 0.5 }}>
            <rect x="10" y="10" width="80" height="80" rx="4" stroke="#999" strokeWidth="2" />
            <rect x="10" y="10" width="20" height="20" fill="#ccc" />
            <rect x="50" y="10" width="20" height="20" fill="#ccc" />
            <rect x="30" y="50" width="20" height="20" fill="#ccc" />
            <rect x="70" y="50" width="20" height="20" fill="#ccc" />
            <rect x="10" y="70" width="20" height="20" fill="#ccc" />
            <rect x="70" y="70" width="20" height="20" fill="#ccc" />
          </svg>
          <p style={styles.emptyTitle}>No puzzles yet</p>
          <p style={styles.emptySubtitle}>Upload a crossword photo to get started</p>
          <Link to="/upload" style={styles.emptyCta} className="btn-primary">
            Upload Your First Puzzle
          </Link>
        </div>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    maxWidth: '800px',
    margin: '0 auto',
    padding: '40px 20px',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    textAlign: 'center' as const,
  },
  brandHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '10px',
    marginBottom: '4px',
  },
  brandLogo: {
    width: '32px',
    height: '32px',
  },
  heading: {
    margin: 0,
    fontSize: '42px',
    lineHeight: 0,
    fontFamily: '"Georgia", "Times New Roman", serif',
    fontWeight: 400,
    letterSpacing: '6px',
    textTransform: 'uppercase' as const,
    textAlign: 'center' as const,
  },
  subtitle: {
    color: '#666',
    fontSize: '16px',
    marginBottom: '16px',
    textAlign: 'center' as const,
  },
  uploadCard: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '20px',
    border: '2px dashed #ccc',
    borderRadius: '8px',
    textDecoration: 'none',
    color: '#999',
    minHeight: '120px',
  },
  uploadCardInner: {
    display: 'flex',
    flexDirection: 'column' as const,
    alignItems: 'center',
    gap: '8px',
  },
  uploadCardText: {
    fontSize: '14px',
    fontWeight: 500,
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
    gap: '16px',
  },
  card: {
    display: 'block',
    padding: '20px',
    border: '1px solid #ddd',
    borderRadius: '8px',
    textDecoration: 'none',
    color: 'inherit',
    transition: 'box-shadow 0.15s',
    backgroundColor: '#fff',
  },
  cardBody: {
    display: 'flex',
    gap: '16px',
    alignItems: 'flex-start',
    textAlign: 'left' as const,
  },
  cardInfo: {
    flex: 1,
    minWidth: 0,
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '4px',
  },
  cardTitle: {
    margin: '0 0 8px 0',
    fontSize: '18px',
  },
  meta: {
    display: 'flex',
    gap: '16px',
    fontSize: '14px',
    color: '#666',
  },
  id: {
    marginTop: '8px',
    fontSize: '12px',
    color: '#aaa',
  },
  emptyState: {
    display: 'flex',
    flexDirection: 'column' as const,
    alignItems: 'center',
    padding: '48px 20px',
    marginTop: '16px',
  },
  emptyTitle: {
    fontSize: '20px',
    fontWeight: 600,
    color: '#444',
    margin: '0 0 6px 0',
  },
  emptySubtitle: {
    fontSize: '15px',
    color: '#999',
    margin: '0 0 20px 0',
  },
  emptyCta: {
    display: 'inline-block',
    padding: '12px 28px',
    backgroundColor: '#2563eb',
    color: '#fff',
    borderRadius: '8px',
    textDecoration: 'none',
    fontSize: '15px',
    fontWeight: 500,
  },
  errorBanner: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '12px',
    padding: '10px 16px',
    backgroundColor: '#fef2f2',
    color: '#b91c1c',
    border: '1px solid #fecaca',
    borderRadius: '6px',
    fontSize: '14px',
    marginBottom: '16px',
    width: '100%',
    textAlign: 'left' as const,
  },
  errorClose: {
    background: 'none',
    border: 'none',
    color: '#b91c1c',
    fontSize: '18px',
    cursor: 'pointer',
    padding: '0 4px',
    lineHeight: 1,
    flexShrink: 0,
  },
  resolveBtn: {
    padding: '4px 12px',
    fontSize: '12px',
    backgroundColor: '#f59e0b',
    color: '#fff',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  deleteBtn: {
    padding: '4px 6px',
    backgroundColor: '#fff',
    color: '#999',
    border: '1px solid #ddd',
    borderRadius: '4px',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  skeletonCard: {
    padding: '24px',
    border: '1px solid #eee',
    borderRadius: '8px',
    backgroundColor: '#fafafa',
  },
  skeletonTitle: {
    width: '60%',
    height: '20px',
    backgroundColor: '#e5e7eb',
    borderRadius: '4px',
    marginBottom: '12px',
  },
  skeletonMeta: {
    width: '80%',
    height: '14px',
    backgroundColor: '#e5e7eb',
    borderRadius: '4px',
    marginBottom: '8px',
  },
  skeletonId: {
    width: '40%',
    height: '12px',
    backgroundColor: '#e5e7eb',
    borderRadius: '4px',
  },
};
