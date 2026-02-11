import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useUploadPipeline } from '../hooks/useUploadPipeline';
import { useSSE } from '../hooks/useSSE';
import type { UploadResponse, MaskData, MaskResponse } from '../types/api';
import GridPreview from './GridPreview';
import GridEditor from './GridEditor';
import ImageMasker from './ImageMasker';

type Step = 'upload' | 'preview' | 'grid-edit' | 'mask' | 'solving';

const STEP_LABELS: Record<Step, string> = {
  upload: 'Upload',
  preview: 'Review',
  'grid-edit': 'Grid',
  mask: 'Mask',
  solving: 'Solving',
};

export default function UploadPage() {
  const navigate = useNavigate();
  const { uploadPhoto, submitGridEdit, submitMask } = useUploadPipeline();

  const [step, setStep] = useState<Step>('upload');
  const [uploading, setUploading] = useState(false);
  const [submittingGrid, setSubmittingGrid] = useState(false);
  const [submittingMask, setSubmittingMask] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadData, setUploadData] = useState<UploadResponse | null>(null);
  const [maskResult, setMaskResult] = useState<MaskResponse | null>(null);
  const [dragOver, setDragOver] = useState(false);

  // SSE for solve progress
  const sseUrl = step === 'solving' && uploadData
    ? `/api/${uploadData.session_id}/progress`
    : null;
  const { data: progress, done: solveDone } = useSSE(sseUrl);

  // Navigate to puzzle when solve completes
  if (solveDone && progress?.stage === 'complete' && maskResult?.puzzle_id) {
    setTimeout(() => navigate(`/puzzle/${maskResult.puzzle_id}`), 800);
  }

  const handleFile = useCallback(async (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file (JPEG or PNG)');
      return;
    }
    setError(null);
    setUploading(true);
    try {
      const result = await uploadPhoto(file);
      setUploadData(result);
      setStep('preview');
    } catch (err: any) {
      setError(err.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
  }, [uploadPhoto]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleGridConfirm = useCallback(async (blackCells: boolean[][]) => {
    if (!uploadData) return;
    setError(null);
    setSubmittingGrid(true);
    try {
      const result = await submitGridEdit(uploadData.session_id, blackCells);
      setUploadData((prev) => prev ? {
        ...prev,
        clue_slot_count: result.clue_slot_count,
      } : prev);
      setStep('mask');
    } catch (err: any) {
      setError(err.message || 'Grid edit failed');
    } finally {
      setSubmittingGrid(false);
    }
  }, [uploadData, submitGridEdit]);

  const handleMaskSubmit = useCallback(async (data: MaskData) => {
    if (!uploadData) return;
    setError(null);
    setSubmittingMask(true);
    try {
      const result = await submitMask(uploadData.session_id, data);
      setMaskResult(result);
      if (result.verification_passed) {
        setStep('solving');
      } else {
        setError(
          `Verification failed: ${result.matched_count}/${result.grid_slot_count} slots matched. ` +
          `OCR found ${result.ocr_clue_count} clues. ` +
          (result.errors.length > 0 ? result.errors.join('; ') : 'Try adjusting masks and separators.')
        );
      }
    } catch (err: any) {
      setError(err.message || 'Mask submission failed');
    } finally {
      setSubmittingMask(false);
    }
  }, [uploadData, submitMask]);

  const steps: Step[] = ['upload', 'preview', 'grid-edit', 'mask', 'solving'];

  return (
    <div style={styles.container}>
      <h1 style={styles.brand}>Crosswise</h1>

      {/* Step indicator */}
      <div style={styles.steps}>
        {steps.map((s, i) => (
          <span
            key={s}
            style={step === s ? styles.stepActive : styles.step}
          >
            {i + 1}. {STEP_LABELS[s]}
          </span>
        ))}
      </div>

      {error && (
        <div style={styles.error}>{error}</div>
      )}

      {/* Step: Upload */}
      {step === 'upload' && (
        <div
          style={{
            ...styles.dropZone,
            ...(dragOver ? styles.dropZoneActive : {}),
          }}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
        >
          {uploading ? (
            <p style={styles.dropText}>Uploading &amp; detecting grid...</p>
          ) : (
            <>
              <p style={styles.dropText}>
                Drag &amp; drop a crossword image here
              </p>
              <p style={styles.dropSub}>or</p>
              <label style={styles.fileLabel}>
                Choose File
                <input
                  type="file"
                  accept="image/*"
                  style={{ display: 'none' }}
                  onChange={handleFileInput}
                />
              </label>
            </>
          )}
        </div>
      )}

      {/* Step: Preview */}
      {step === 'preview' && uploadData && (
        <GridPreview
          data={uploadData}
          onConfirm={() => setStep('grid-edit')}
          onReupload={() => {
            setStep('upload');
            setUploadData(null);
            setError(null);
          }}
        />
      )}

      {/* Step: Grid Editor */}
      {step === 'grid-edit' && uploadData && (
        <GridEditor
          sessionId={uploadData.session_id}
          warpedGridUrl={uploadData.warped_grid_url}
          initialSlotCount={uploadData.clue_slot_count}
          onConfirm={handleGridConfirm}
          onBack={() => setStep('preview')}
          submitting={submittingGrid}
        />
      )}

      {/* Step: Mask */}
      {step === 'mask' && uploadData && (
        <ImageMasker
          imageUrl={uploadData.original_image_url}
          onSubmit={handleMaskSubmit}
          submitting={submittingMask}
          exampleMaskedUrl="/masking-example.jpg"
        />
      )}

      {/* Step: Solving */}
      {step === 'solving' && (
        <div style={styles.solvingBox}>
          <h2 style={styles.solvingTitle}>
            {solveDone && progress?.stage === 'complete'
              ? 'Puzzle Ready!'
              : 'Solving Puzzle...'}
          </h2>
          {progress && (
            <>
              <p style={styles.solvingMessage}>{progress.message}</p>
              {progress.progress >= 0 && progress.progress <= 1 && (
                <div style={styles.progressBar}>
                  <div
                    style={{
                      ...styles.progressFill,
                      width: `${Math.round(progress.progress * 100)}%`,
                    }}
                  />
                </div>
              )}
            </>
          )}
          {!progress && (
            <p style={styles.solvingMessage}>Connecting...</p>
          )}
          {solveDone && progress?.stage === 'failed' && (
            <p style={styles.errorText}>Solve failed. You can still view the puzzle with partial results.</p>
          )}
        </div>
      )}
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
  brand: {
    margin: '0 0 16px 0',
    fontSize: '42px',
    fontFamily: '"Georgia", "Times New Roman", serif',
    fontWeight: 400,
    letterSpacing: '6px',
    textTransform: 'uppercase' as const,
    textAlign: 'center' as const,
  },
  steps: {
    display: 'flex',
    gap: '24px',
    marginBottom: '24px',
    fontSize: '13px',
    color: '#999',
  },
  step: {
    color: '#bbb',
  },
  stepActive: {
    color: '#2563eb',
    fontWeight: 600,
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
  dropZone: {
    border: '2px dashed #ccc',
    borderRadius: '12px',
    padding: '60px 40px',
    textAlign: 'center' as const,
    width: '100%',
    maxWidth: '500px',
    cursor: 'pointer',
    transition: 'border-color 0.15s, background-color 0.15s',
  },
  dropZoneActive: {
    borderColor: '#2563eb',
    backgroundColor: '#eff6ff',
  },
  dropText: {
    fontSize: '17px',
    color: '#555',
    margin: '0 0 8px 0',
  },
  dropSub: {
    fontSize: '14px',
    color: '#aaa',
    margin: '0 0 12px 0',
  },
  fileLabel: {
    display: 'inline-block',
    padding: '10px 24px',
    backgroundColor: '#2563eb',
    color: '#fff',
    borderRadius: '6px',
    fontSize: '15px',
    cursor: 'pointer',
  },
  solvingBox: {
    textAlign: 'center' as const,
    padding: '40px',
    border: '1px solid #ddd',
    borderRadius: '12px',
    width: '100%',
    maxWidth: '500px',
  },
  solvingTitle: {
    margin: '0 0 12px 0',
    fontSize: '20px',
  },
  solvingMessage: {
    color: '#666',
    fontSize: '15px',
    margin: '0 0 16px 0',
  },
  progressBar: {
    height: '8px',
    backgroundColor: '#e5e7eb',
    borderRadius: '4px',
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#2563eb',
    borderRadius: '4px',
    transition: 'width 0.3s ease',
  },
  errorText: {
    color: '#b91c1c',
    fontSize: '14px',
  },
};
