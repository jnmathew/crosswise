import { useState, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useUploadPipeline } from '../hooks/useUploadPipeline';
import type { UploadResponse, MaskData } from '../types/api';
import GridEditor from './GridEditor';
import GridCropper from './GridCropper';
import ImageMasker from './ImageMasker';

type Step = 'upload' | 'grid-edit' | 'manual-crop' | 'mask';

const STEP_LABELS: Record<Step, string> = {
  upload: 'Upload',
  'grid-edit': 'Grid',
  'manual-crop': 'Crop',
  mask: 'Mask',
};

export default function UploadPage() {
  const navigate = useNavigate();
  const { uploadPhoto, submitGridEdit, submitMask, startPipeline, submitManualCrop } = useUploadPipeline();

  const [step, setStep] = useState<Step>('upload');
  const [uploading, setUploading] = useState(false);
  const [submittingGrid, setSubmittingGrid] = useState(false);
  const [submittingMask, setSubmittingMask] = useState(false);
  const [submittingCrop, setSubmittingCrop] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadData, setUploadData] = useState<UploadResponse | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [ocrProvider, setOcrProvider] = useState<string>('gemini');

  useEffect(() => {
    fetch('/api/config')
      .then((r) => r.json())
      .then((data) => setOcrProvider(data.ocr_provider ?? 'gemini'))
      .catch(() => {});
  }, []);

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
      setStep('grid-edit');
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Upload failed');
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

  const handleMaskSubmit = useCallback(async (data: MaskData) => {
    if (!uploadData) return;
    setError(null);
    setSubmittingMask(true);
    try {
      const pipelineResult = await startPipeline(uploadData.session_id, data);
      navigate(`/puzzle/${pipelineResult.puzzle_id}`);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Pipeline start failed');
    } finally {
      setSubmittingMask(false);
    }
  }, [uploadData, startPipeline, navigate]);

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

      if (ocrProvider === 'gemini') {
        // Start full pipeline in background, navigate immediately
        const pipelineResult = await startPipeline(uploadData.session_id, { rectangles: [], separators: [] });
        navigate(`/puzzle/${pipelineResult.puzzle_id}`);
      } else {
        setStep('mask');
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Grid edit failed');
    } finally {
      setSubmittingGrid(false);
    }
  }, [uploadData, submitGridEdit, ocrProvider, startPipeline, navigate]);

  const handleManualCrop = useCallback(async (corners: number[][]) => {
    if (!uploadData) return;
    setError(null);
    setSubmittingCrop(true);
    try {
      const result = await submitManualCrop(uploadData.session_id, corners);
      // Cache-bust the warped grid URL so GridEditor refetches both image and grid_data.json
      setUploadData({
        ...result,
        warped_grid_url: `${result.warped_grid_url}?t=${Date.now()}`,
      });
      setStep('grid-edit');
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Manual crop failed');
    } finally {
      setSubmittingCrop(false);
    }
  }, [uploadData, submitManualCrop]);

  const baseSteps: Step[] = ocrProvider === 'gemini'
    ? ['upload', 'grid-edit']
    : ['upload', 'grid-edit', 'mask'];
  const steps: Step[] = step === 'manual-crop'
    ? ['upload', 'manual-crop', 'grid-edit']
    : baseSteps;

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

      {/* Step: Grid Editor */}
      {step === 'grid-edit' && uploadData && (
        <GridEditor
          sessionId={uploadData.session_id}
          warpedGridUrl={uploadData.warped_grid_url}
          initialSlotCount={uploadData.clue_slot_count}
          onConfirm={handleGridConfirm}
          onBack={() => {
            setStep('upload');
            setUploadData(null);
            setError(null);
          }}
          onManualCrop={() => {
            setError(null);
            setStep('manual-crop');
          }}
          submitting={submittingGrid}
        />
      )}

      {/* Step: Manual Crop */}
      {step === 'manual-crop' && uploadData && (
        <GridCropper
          imageUrl={uploadData.original_image_url}
          onSubmit={handleManualCrop}
          onBack={() => {
            setError(null);
            setStep('grid-edit');
          }}
          submitting={submittingCrop}
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
};
