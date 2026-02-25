# Learnings: How the Preprocessing Pipeline Evolved

This document traces how the image preprocessing pipeline went from a naive 160-line script to a production-ready module with five fallback strategies. Each version taught me something about why "just find the biggest contour" doesn't work on real newspaper photos.

## V1: The Naive Approach (`preprocessv2.py`)

The first version was about 160 lines and made every beginner assumption:

- **One thresholding strategy**: Otsu on a Gaussian blur. No fallback.
- **"Biggest contour = grid"**: Find contours, sort by area, take the first one.
- **Hardcoded output dimensions**: `dst_w = 1300, dst_h = 1400` (13 cols x 14 rows) — baked in the grid size I happened to be testing with.
- **No error handling**: If contour approximation didn't return exactly 4 points, it just used the entire image bounds as the quad.

The script wrote debug images directly to the working directory, which is how the images below were generated.

### What V1 produced

**Step 1 — Page-level contour detection:**

The green box is what it picked as the "grid." The approach worked here because the grid happened to be the largest dark region, but this was fragile.

![Page-level grid detection](images/v1-page-contour-detection.png)

**Step 2 — Rough crop of the detected region:**

Just a bounding-box crop of the largest contour. No perspective correction yet — the grid lines are visibly skewed.

![Rough grid crop](images/v1-rough-crop.png)

**Step 3 — Tightened crop:**

Ran contour detection *again* on the rough crop to get a tighter bounding box. This "tighten" step was needed because the first detection often grabbed too much margin.

![Tightened grid](images/v1-tightened-crop.png)

**Step 4 — Perspective warp:**

Applied a perspective transform to straighten the grid. The hardcoded 1300x1400 output meant this only looked right for one specific puzzle.

![Warped grid](images/v1-perspective-warp.png)

### What I learned from V1

- Otsu thresholding fails on newspaper images with uneven lighting or halftone patterns.
- Sorting contours by area is unreliable — ads, photos, and other page elements are often larger than the crossword grid.
- Hardcoding output dimensions doesn't generalize. A 13x14 grid is not a 15x15 grid.
- Writing `cv2.imwrite("debug_page_grids.png", ...)` to the working directory instead of an output folder is messy and leads to debug artifacts getting committed to git (like these images originally were).
- The two-pass "detect then tighten" approach was a code smell — it meant the first detection wasn't precise enough. A better algorithm wouldn't need a second pass.

## V2: The Kitchen Sink (`oldpreprocess.py`)

V2 exploded to ~900 lines. This was the "throw everything at the wall" phase:

**What changed:**
- **Multi-strategy quad detection**: Canny edges, adaptive thresholding, morphological close, and `minAreaRect` as a last resort — four strategies tried in sequence.
- **Manual fallbacks**: Interactive ROI selector (drag a box) and a 4-corner click selector, because sometimes the automatic detection just didn't work and I needed a human in the loop.
- **Projection-based grid line detection**: Instead of just finding the outer boundary, this analyzed row/column intensity projections to find the actual grid lines. Peaks in the projection = grid lines.
- **Grid intersection detection**: Found where horizontal and vertical lines crossed, then clustered the intersections into a regular grid.
- **Multi-pass OCR ensemble**: For extracting clue numbers from cells, it tried three different thresholding methods (Otsu, CLAHE + Otsu, adaptive Gaussian) combined with three different Tesseract PSM modes. A 3x3 matrix of attempts per cell.
- **Smart cell filtering**: Classified black cells by mean intensity, then only ran OCR on cells that could start a clue (left or top neighbor is black/border).

### What I learned from V2

- **More strategies isn't always better.** The code became hard to debug because I couldn't tell which strategy was actually winning for a given image. I had no logging to trace the detection path.
- **Manual fallbacks are a crutch.** If I need a human to click corners, the algorithm isn't good enough. Manual selection should be for edge cases, not a regular part of the pipeline.
- **OCR on individual grid cells is fragile.** The multi-pass ensemble for digit extraction was 300+ lines of code to handle edge cases (halftone dots, broken strokes, low contrast). I eventually realized that grid clue numbers should be *computed algorithmically* from the grid structure (which cell starts an across/down word), not OCR'd from tiny cell images.
- **The script did too many things.** Grid detection, cell extraction, digit OCR, and output formatting were all in one file. When the OCR broke, I had to wade through grid detection code to find the issue.

## V3: Separation of Concerns (`image_preprocessing.py` + `grid_detection.py`)

The current version split the monolith into focused modules:

| Module | Responsibility |
|--------|---------------|
| `image_preprocessing.py` | Load image, find quad, perspective warp |
| `grid_detection.py` | Analyze warped grid: find cells, classify black/white, assign clue numbers |
| `ocr_utils.py` | OCR configuration and text extraction utilities |
| `clue_extraction.py` | Parse OCR output, match clues to grid slots |

**What changed in the quad detection:**
- **Hough line refinement** (Strategy A): First get an approximate quad from contours, then use Hough line detection *within that region* to find actual grid lines and compute more precise corners from line intersections. This was the key insight — contours give you a rough boundary, but Hough gives you the actual straight lines.
- **Five-strategy cascade with logging**: Each strategy logs whether it succeeded or failed, so I can trace exactly what happened for any image.
- **Rotation correction**: After warping, detect residual rotation via Hough lines and correct it. (Currently disabled — it was over-rotating on some inputs, a problem I haven't solved yet.)
- **Page boundary detection**: Three strategies for finding the newspaper page itself (edge-based, threshold-based, text density), useful when the photo includes the table or background.
- **No manual fallbacks in the core module.** The API server handles user interaction separately; the preprocessing module is pure computation.

**The biggest wins:**
- Clue numbers are now **computed from grid structure**, not OCR'd. This eliminated hundreds of lines of fragile digit-extraction code and is 100% accurate by definition.
- Each module has a single responsibility and can be tested independently.
- Type hints and docstrings everywhere — the code is readable by someone who didn't write it.
- No hardcoded paths or image dimensions.

## Key Takeaways

1. **Start simple, but know when to refactor.** V1 was the right thing to build first — it proved the concept. But I should have refactored earlier instead of piling features onto V2.

2. **Algorithmic solutions beat statistical ones for structured problems.** Computing clue numbers from grid structure is always correct. OCR'ing them from tiny cell images was never going to be reliable.

3. **Log your decision path.** V2's biggest debugging problem was not knowing which strategy fired. V3's cascade logging made it immediately clear.

4. **Separate concerns early.** The V2 monolith made every change risky because grid detection, OCR, and output formatting were interleaved. Splitting into modules made each piece independently testable.

5. **Don't commit debug artifacts.** V1's habit of writing `cv2.imwrite("debug.png", ...)` to the working directory led to 36 MB of PNGs tracked in git. Output should always go to a designated, gitignored directory.

---

## The LLM Solver: From 65/69 to 69/69

### The problem that wouldn't go away

Puzzle 2 (79d2dbf1dc23) was stuck at 65/69 for a long time, with the same 4 clues unsolved every run:
- 17-across "Candid" (4 letters)
- 20-across "Cooked fruit dessert" (7 letters)
- 25-down "Panhandler" (6 letters)
- 26-down "Boxing event" (4 letters)

### Root cause: forward-only solver can't backtrack

Deep-diving the grid revealed the 4 unsolved clues had crossing-letter patterns that matched **no valid English word** — OAEN, COMPRTE, BEGSAR, BOUE. The patterns were created by exactly 2 wrong committed answers:

| Wrong Answer | Correct | Why | Clues It Poisoned |
|---|---|---|---|
| 9-down "Hard-rind fruit" = **PEAR** | **PEPO** | PEAR is the obvious word, but PEPO (botanical term for hard-rind fruits like watermelon) is the crosswordese answer. The A and R from PEAR made impossible patterns at crossing positions. | 17-across (should be OPEN), 20-across (should be COMPOTE) |
| 40-across "Understand" = **SEE** | **GET** | Both valid 3-letter synonyms, but SEE was committed first. The S and E from SEE blocked the crossing clues. | 25-down (should be BEGGAR), 26-down (should be BOUT) |

The LLM solver was "forward-only" — it committed answers and never reconsidered them. When wrong answers created impossible crossing patterns, it just gave up on the crossing clues.

### The fix: conflict resolution with backtracking

Added a conflict resolution phase that runs after the main solve passes get stuck:

1. **Detect dead-ends**: For each unsolved clue, check if its crossing-letter pattern matches any word in the word index. If not, the pattern is "dead" — some crossing answer is wrong.

2. **Trace blame**: For each dead-end, identify which committed crossing answers provided the impossible letters. These are the "blamed" answers.

3. **Cluster**: Group dead-ends that share blamed answers (PEAR blocks both OPEN and COMPOTE, so they form one cluster).

4. **LLM re-solve**: Remove the blamed answers from the grid, show the LLM the conflict ("your answer PEAR for 'Hard-rind fruit' makes these crossing clues impossible"), and ask it to re-solve the whole cluster jointly. Web search is available for verification.

5. **Post-resolution pass**: After conflict resolution adds new letters to the grid, run one more solve pass to pick up newly-unblocked clues.

### Web search tool

Also added Anthropic's server-side `web_search_20250305` tool to the LLM solver. This helps the LLM verify ambiguous proper nouns before committing — e.g., "Actress — Gabor" = EVA (not ZSA, which was another persistent wrong answer that poisoned crossings with a Z).

### Results

Puzzle 2 now solves to **69/69** (perfect). Typical solver trace:
```
Pass 1: 0→22 (3 web searches)
Pass 2: 22→39 (1 web search)
Pass 3: 39→54 (2 web searches)
Pass 4: 54→64 (3 web searches)
Pass 5: 64→64 (stuck — dead-end patterns detected)
Conflict resolution cluster 1: 2 dead-ends, 9 blamed → resolved (+2)
Conflict resolution cluster 2: 1 dead-end, 3 blamed → resolved (+1)
Post-resolution pass: +2 newly-unblocked clues
FINAL: 69/69
```

Puzzle 1 (0e76e98a26db) remains **71/71** — no regression.

### Key insight

The forward-only LLM solver was fine for ~94% of clues. The last ~6% required a fundamentally different capability: recognizing that committed answers were wrong by detecting their downstream consequences (impossible crossing patterns), then backtracking surgically. This is the same reasoning a human solver uses — "if this crossing pattern is impossible, one of my earlier answers must be wrong."
