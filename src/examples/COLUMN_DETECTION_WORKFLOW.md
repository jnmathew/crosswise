# Column Detection Workflow for Crossword Clues

This guide explains how to use the automatic column detection pipeline for extracting crossword clues from newspaper images.

## Overview

The pipeline consists of two main steps:
1. **Interactive Masking** - Manually draw white rectangles over irrelevant areas (grids, ads, acrostics)
2. **Auto-Processing** - Automatically detect columns and add separator lines

## Step 1: Interactive Masking

Run the interactive masking tool to hide unwanted content:

```bash
source .venv/bin/activate
python src/examples/interactive_masker.py "src/examples/IMG_5527 copy.JPG"
```

### Controls:
- **Click and drag** to draw white rectangles over:
  - Crossword grids (both puzzles)
  - Acrostic puzzles
  - Ads and comics
  - Any non-clue content
- **'s'** - Save the masked image and coordinates
- **'u'** - Undo last rectangle
- **'c'** - Clear all rectangles
- **'h'** - Show help
- **'q'** - Quit

### Output:
- `IMG_5527 copy_masked.JPG` - Image with white rectangles
- `IMG_5527 copy_mask_coords.json` - Saved coordinates (for later editing)

## Step 2: Auto-Processing

Run the automatic column detection and separator drawing:

```bash
python src/examples/process_masked_image.py "src/examples/IMG_5527 copy_masked.JPG" --run-ocr
```

### What it does:
1. Loads the masked image
2. Detects column boundaries using:
   - Text region clustering (handles partial-height columns)
   - Vertical projection analysis (handles full-height columns)
   - Hybrid merging of both methods
3. Draws **aqua/cyan** vertical separator lines between columns
4. Saves the final preprocessed image
5. Optionally runs Mistral OCR to extract clues

### Output:
- `IMG_5527 copy_masked_divided.JPG` - Final image with separators
- `IMG_5527 copy_column_boundaries_vis.JPG` - Visualization with green boundaries
- `IMG_5527 copy_final_ocr_results.md` - OCR results (if --run-ocr used)

### Options:
```bash
--run-ocr               # Run Mistral OCR after preprocessing
--min-column-width 150  # Override auto-detected minimum column width
--max-columns 12        # Maximum number of columns to detect
```

## Key Features

### Handles Partial-Height Columns
The improved algorithm detects columns that don't span the entire page height by:
- Clustering text regions by x-coordinate
- Creating histograms of text positions
- Finding valleys (gaps) in the distribution

### Aqua Separators
Separator lines are drawn in **aqua/cyan** (BGR: 255, 255, 0) for visibility:
- Easy to see over white masks
- Contrasts well with gray newspaper background
- Clearly distinguishes column boundaries

### Saves Coordinates
The masking coordinates are saved to JSON, allowing you to:
- Re-open and edit masks later
- Apply the same masks to multiple images
- Document your preprocessing steps

## Example Complete Workflow

```bash
# Activate virtual environment
source .venv/bin/activate

# Step 1: Mask unwanted areas
python src/examples/interactive_masker.py "src/examples/IMG_5527 copy.JPG"
# ... draw rectangles over grids and acrostic ...
# ... press 's' to save ...

# Step 2: Auto-process with column detection and OCR
python src/examples/process_masked_image.py "src/examples/IMG_5527 copy_masked.JPG" --run-ocr

# Results are saved in the same directory
```

## Troubleshooting

### Too many/few columns detected
Adjust the minimum column width:
```bash
python src/examples/process_masked_image.py image.JPG --min-column-width 200
```

### OCR reads across columns
- Ensure you've masked the grids (they interfere with column detection)
- Try increasing separator width by editing the script
- Check that columns are properly detected in the visualization file

### Need to edit masks
Just re-run the masker on the original image - it will load the saved coordinates:
```bash
python src/examples/interactive_masker.py "src/examples/IMG_5527 copy.JPG"
```
