export interface UploadResponse {
  session_id: string;
  status: string;
  grid_size: number[];
  clue_slot_count: number;
  warped_grid_url: string;
  original_image_url: string;
}

export interface MaskRect {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface MaskData {
  rectangles: MaskRect[];
  separators: MaskRect[];
}

export interface MaskResponse {
  status: string;
  verification_passed: boolean;
  ocr_clue_count: number;
  grid_slot_count: number;
  matched_count: number;
  errors: string[];
  puzzle_id: string | null;
}

export interface SolveProgress {
  stage: string;
  message: string;
  progress: number;
}

export interface GridEditResponse {
  clue_slot_count: number;
  clue_number_count: number;
  grid_size: number[];
}

export interface PuzzleListEntry {
  id: string;
  title: string;
  gridSize: number[];
  totalClues: number;
  solved: number;
}
