#!/usr/bin/env python3
"""
Interactive corner selector for crossword grid.

Usage:
    python select_corners.py <image_path>

Click the 4 corners in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
Press 'r' to reset, 'q' to quit, 'Enter' when done.
"""
import sys
import cv2
import numpy as np
from pathlib import Path

# Global variables
corners = []
img_display = None
img_original = None
window_name = "Select Corners (TL, TR, BR, BL) - Press Enter when done, R to reset, Q to quit"

def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks to select corners."""
    global corners, img_display

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(corners) < 4:
            corners.append((x, y))

            # Draw the corner
            img_display = img_original.copy()

            # Draw all corners
            labels = ['TL', 'TR', 'BR', 'BL']
            colors = [(0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 255, 0)]

            for i, (cx, cy) in enumerate(corners):
                cv2.circle(img_display, (cx, cy), 8, colors[i], -1)
                cv2.putText(img_display, labels[i], (cx + 15, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)

            # Draw lines between corners if we have at least 2
            if len(corners) > 1:
                for i in range(len(corners) - 1):
                    cv2.line(img_display, corners[i], corners[i + 1], (255, 255, 255), 2)

                # Close the quad if we have all 4
                if len(corners) == 4:
                    cv2.line(img_display, corners[3], corners[0], (255, 255, 255), 2)

            cv2.imshow(window_name, img_display)

def main():
    global img_display, img_original, corners

    if len(sys.argv) != 2:
        print("Usage: python select_corners.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image: {image_path}")
        sys.exit(1)

    # Resize to fit screen if needed (max 1400px)
    h, w = img.shape[:2]
    max_dim = 1400
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        print(f"Image resized to {img.shape[1]}×{img.shape[0]}")

    img_original = img.copy()
    img_display = img.copy()

    # Create window
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n" + "="*70)
    print("CORNER SELECTOR")
    print("="*70)
    print("Click the 4 corners in order:")
    print("  1. Top-Left (TL) - Green")
    print("  2. Top-Right (TR) - Cyan")
    print("  3. Bottom-Right (BR) - Red")
    print("  4. Bottom-Left (BL) - Yellow")
    print("\n⚠️  IMPORTANT: Click in the WHITE SPACE just outside the grid,")
    print("   NOT on the black grid lines themselves. This ensures all")
    print("   grid lines are detected after perspective correction.")
    print("\nControls:")
    print("  - Click to add corner")
    print("  - Press 'r' to reset")
    print("  - Press 'Enter' when done")
    print("  - Press 'q' to quit")
    print("="*70 + "\n")

    cv2.imshow(window_name, img_display)

    while True:
        key = cv2.waitKey(1) & 0xFF

        # Reset
        if key == ord('r'):
            corners = []
            img_display = img_original.copy()
            cv2.imshow(window_name, img_display)
            print("Reset corners")

        # Quit
        elif key == ord('q'):
            print("Cancelled")
            cv2.destroyAllWindows()
            sys.exit(0)

        # Done (Enter key)
        elif key == 13:  # Enter
            if len(corners) == 4:
                break
            else:
                print(f"Need 4 corners, only {len(corners)} selected")

    cv2.destroyAllWindows()

    # Print results
    print("\n" + "="*70)
    print("SELECTED CORNERS")
    print("="*70)
    labels = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']
    for i, (label, (x, y)) in enumerate(zip(labels, corners)):
        print(f"{label:15s}: ({x:4d}, {y:4d})")

    # Format for command line
    coords_str = ','.join([f"{x},{y}" for x, y in corners])
    print("\n" + "="*70)
    print("COMMAND LINE")
    print("="*70)
    print(f"python -m src.main {image_path} --corners \"{coords_str}\"")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
