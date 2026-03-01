import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import type { PuzzleListEntry } from '../types/api';

export default function PuzzleSelector() {
  const [puzzles, setPuzzles] = useState<PuzzleListEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [solvingId, setSolvingId] = useState<string | null>(null);
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
        alert(data.detail || 'Failed to start solve');
        setSolvingId(null);
      }
    } catch {
      alert('Failed to start solve');
      setSolvingId(null);
    }
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.heading}>Crosswise</h1>
      <p style={styles.subtitle}>Select a puzzle to play</p>

      <Link to="/upload" style={styles.uploadLink}>
        + Upload New Puzzle
      </Link>

      {loading && <p style={styles.loadingText}>Loading puzzles...</p>}

      <div style={styles.grid}>
        {puzzles.map((p) => (
          <Link to={`/puzzle/${p.id}`} key={p.id} style={styles.card}>
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
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={styles.id}>{p.id}</div>
              {p.solved < p.totalClues && (
                <button
                  style={styles.resolveBtn}
                  onClick={(e) => handleResolve(e, p.id)}
                  disabled={solvingId === p.id}
                >
                  {solvingId === p.id ? 'Starting...' : 'Re-solve'}
                </button>
              )}
            </div>
          </Link>
        ))}
      </div>

      {!loading && puzzles.length === 0 && (
        <p style={styles.empty}>No puzzles yet. Upload one to get started!</p>
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
  heading: {
    fontSize: '42px',
    fontFamily: '"Georgia", "Times New Roman", serif',
    fontWeight: 400,
    letterSpacing: '6px',
    textTransform: 'uppercase' as const,
    textAlign: 'center' as const,
    marginBottom: '4px',
  },
  subtitle: {
    color: '#666',
    fontSize: '16px',
    marginBottom: '16px',
    textAlign: 'center' as const,
  },
  uploadLink: {
    display: 'inline-block',
    padding: '10px 24px',
    backgroundColor: '#2563eb',
    color: '#fff',
    borderRadius: '6px',
    textDecoration: 'none',
    fontSize: '15px',
    marginBottom: '32px',
  },
  loadingText: {
    color: '#999',
    fontSize: '15px',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
    gap: '16px',
  },
  card: {
    display: 'block',
    padding: '24px',
    border: '1px solid #ddd',
    borderRadius: '8px',
    textDecoration: 'none',
    color: 'inherit',
    transition: 'box-shadow 0.15s',
    backgroundColor: '#fff',
  },
  cardTitle: {
    margin: '0 0 12px 0',
    fontSize: '20px',
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
  empty: {
    color: '#999',
    fontSize: '15px',
    marginTop: '24px',
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
};
