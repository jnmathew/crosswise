import { useState, useCallback } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useUploadPipeline } from '../hooks/useUploadPipeline';
import ImageMasker from './ImageMasker';
import type { MaskData } from '../types/api';

export default function MaskRetryPage() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();
  const { startPipeline } = useUploadPipeline();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = useCallback(async (data: MaskData) => {
    if (!sessionId) return;
    setError(null);
    setSubmitting(true);
    try {
      const result = await startPipeline(sessionId, data);
      navigate(`/puzzle/${result.puzzle_id}`);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Pipeline failed');
    } finally {
      setSubmitting(false);
    }
  }, [sessionId, startPipeline, navigate]);

  if (!sessionId) return null;

  return (
    <div style={styles.container} className="upload-container page-fade">
      <Link to="/" style={styles.brandLink} className="brand-link">
        <img src="/crosswise.svg" alt="" style={styles.brandLogo} />
        <h1 style={styles.brand}>Crosswise</h1>
      </Link>

      <p style={styles.hint}>
        OCR failed on this image. Mask out non-crossword areas (comics, ads, previous puzzle answers) and retry.
      </p>

      {error && <div style={styles.error}>{error}</div>}

      <ImageMasker
        imageUrl={`/api/files/${sessionId}/original.jpg`}
        onSubmit={handleSubmit}
        submitting={submitting}
      />
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: '20px 40px',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    maxWidth: '900px',
    margin: '0 auto',
  },
  brandLink: {
    textDecoration: 'none',
    color: 'inherit',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '10px',
    marginBottom: '16px',
  },
  brandLogo: {
    width: '32px',
    height: '32px',
  },
  brand: {
    margin: 0,
    fontSize: '42px',
    lineHeight: 0,
    fontFamily: '"Georgia", "Times New Roman", serif',
    fontWeight: 400,
    letterSpacing: '6px',
    textTransform: 'uppercase' as const,
    textAlign: 'center' as const,
  },
  hint: {
    fontSize: '14px',
    color: '#666',
    marginBottom: '16px',
    textAlign: 'center' as const,
    maxWidth: '600px',
  },
  error: {
    backgroundColor: '#fef2f2',
    color: '#b91c1c',
    padding: '10px 16px',
    borderRadius: '6px',
    fontSize: '14px',
    marginBottom: '16px',
    maxWidth: '600px',
    width: '100%',
    textAlign: 'center' as const,
  },
};
