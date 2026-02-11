import type { UploadResponse, MaskData, MaskResponse, GridEditResponse } from '../types/api';

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

  return { uploadPhoto, submitGridEdit, submitMask, startSolve };
}
