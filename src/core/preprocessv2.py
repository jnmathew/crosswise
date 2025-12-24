import cv2
import numpy as np
import sys
import os

def load_image(image_path):
    """Load an image from a file path."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at path: {image_path}")
    return img

def find_contours_for_grids(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(binary)

    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    h, w = img.shape[:2]
    grid_candidates = []

    print("total contours:", len(contours))
    for i, c in enumerate(contours[:2]):  # just look at the biggest 2
        x, y, cw, ch = cv2.boundingRect(c)
        ar = cw / float(ch)
        print(f"{i}: x={x}, y={y}, w={cw}, h={ch}, ar={ar:.2f}")

        # for now, DON'T filter by aspect ratio
        grid_candidates.append((x, y, cw, ch))
        if len(grid_candidates) == 2:
            break

    return grid_candidates

def tighten_grid(crop):
    """Run contour detection again on the cropped grid to get a tighter box."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(binary)

    cnts, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError("No contours found in cropped grid")

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(cnts[0])
    return x, y, w, h

def find_grid_quad(tight_img):
    gray = cv2.cvtColor(tight_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(binary)

    cnts, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    grid_cnt = cnts[0]

    peri = cv2.arcLength(grid_cnt, True)
    approx = cv2.approxPolyDP(grid_cnt, 0.02 * peri, True)
    
    if len(approx) != 4:
        h, w = tight_img.shape[:2]
        quad = np.array([
            [[0, 0]],
            [[w-1, 0]],
            [[w-1, h-1]],
            [[0, h-1]]
        ], dtype="float32")
        return quad
    
    return approx  # would be 4 points

def warp_grid(tight_img, quad):
    # quad is like [[[x1, y1]], [[x2, y2]], ...]
    pts = quad.reshape(4, 2).astype("float32")

    # order the points: top-left, top-right, bottom-right, bottom-left
    # simplest ordering: sort by y, then x
    # for production, use a real order_points, but here's a quick one:
    # (we'll assume it's not wildly skewed)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    dst_w = 1300  # 13 cols
    dst_h = 1400  # 14 rows

    dst = np.array([
        [0, 0],
        [dst_w - 1, 0],
        [dst_w - 1, dst_h - 1],
        [0, dst_h - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl]), dst)
    warped = cv2.warpPerspective(tight_img, M, (dst_w, dst_h))
    return warped

def slice_warped_to_cells(warped, rows=14, cols=13, outdir="cells"):
    os.makedirs(outdir, exist_ok=True)
    h, w = warped.shape[:2]
    cell_h = h // rows
    cell_w = w // cols

    idx = 0
    for r in range(rows):
        for c in range(cols):
            y1 = r * cell_h
            y2 = (r + 1) * cell_h
            x1 = c * cell_w
            x2 = (c + 1) * cell_w

            cell = warped[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(outdir, f"cell_{r:02d}_{c:02d}.png"), cell)
            idx += 1


def main():
    path = sys.argv[1]
    img = load_image(path)
    print("Image shape:", img.shape)

    # 1. page-level detection
    grids = find_contours_for_grids(img)

    if not grids:
        raise SystemExit("No grids found")

    # draw rough detections
    debug_page = img.copy()
    for i, (x, y, w, h) in enumerate(grids):
        cv2.rectangle(debug_page, (x, y), (x + w, y + h),
                      (0, 255, 0) if i == 0 else (255, 0, 0), 3)
    cv2.imwrite("debug_page_grids.png", debug_page)

    # 2. take the big one (first) and tighten it
    x, y, w, h = grids[0]
    rough_grid = img[y:y+h, x:x+w]
    cv2.imwrite("rough_grid.png", rough_grid)

    tx, ty, tw, th = tighten_grid(rough_grid)
    tight_grid = rough_grid[ty:ty+th, tx:tx+tw]
    cv2.imwrite("tight_grid.png", tight_grid)
    quad = find_grid_quad(tight_grid)
    warped = warp_grid(tight_grid, quad)
    cv2.imwrite("warped_grid.png", warped)
    slice_warped_to_cells(warped)
    #cv2.imshow("Warped", warped)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

if __name__ == "__main__":
    main()