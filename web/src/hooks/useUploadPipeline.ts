import type { UploadResponse, MaskData, MaskResponse, GridEditResponse, GridResizeResponse } from '../types/api';

export function useUploadPipeline() {
  async function submitGridEdit(sessionId: string, blackCells: boolean[][]): Promise<GridEditResponse> {
    const res = await fetch(`/api/${sessionId}/grid-edit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ black_cells: blackCells }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Grid edit failed');
    }
    return res.json();
  }

  async function uploadPhoto(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch('/api/upload', { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Upload failed');
    }
    return res.json();
  }

  async function submitMask(sessionId: string, masks: MaskData): Promise<MaskResponse> {
    const res = await fetch(`/api/${sessionId}/mask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(masks),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Mask submission failed');
    }
    return res.json();
  }

  async function startSolve(sessionId: string): Promise<void> {
    await fetch(`/api/${sessionId}/solve`, { method: 'POST' });
  }

  async function resizeGrid(sessionId: string, rows: number, cols: number): Promise<GridResizeResponse> {
    const res = await fetch(`/api/${sessionId}/resize-grid`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rows, cols }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Grid resize failed');
    }
    return res.json();
  }

  async function submitManualCrop(sessionId: string, corners: number[][]): Promise<UploadResponse> {
    const res = await fetch(`/api/${sessionId}/manual-crop`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ corners }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Manual crop failed');
    }
    return res.json();
  }

  return { uploadPhoto, submitGridEdit, submitMask, startSolve, resizeGrid, submitManualCrop };
}
