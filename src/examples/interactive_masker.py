"""
Interactive tool for masking irrelevant regions in crossword images.

Usage:
- Click and drag to draw white rectangles over areas to mask
- Press 's' to save the masked image
- Press 'u' to undo last rectangle
- Press 'c' to clear all rectangles
- Press 'q' to quit without saving
- Press 'h' for help

The tool saves:
1. Masked image with white rectangles
2. JSON file with rectangle coordinates for reuse
"""
import sys
from pathlib import Path
import json

import cv2
import numpy as np
from loguru import logger


class InteractiveMasker:
    def __init__(self, image_path: Path):
        self.image_path = image_path
        self.original = cv2.imread(str(image_path))
        if self.original is None:
            raise ValueError(f"Cannot load image: {image_path}")

        # Resize if too large for display
        self.display_scale = 1.0
        h, w = self.original.shape[:2]
        max_display_size = 1200
        if max(h, w) > max_display_size:
            self.display_scale = max_display_size / max(h, w)
            display_w = int(w * self.display_scale)
            display_h = int(h * self.display_scale)
            self.display_image = cv2.resize(self.original, (display_w, display_h))
        else:
            self.display_image = self.original.copy()

        self.working_image = self.original.copy()
        self.rectangles = []  # List of (x1, y1, x2, y2) in original coordinates
        self.separators = []  # List of ((x1, y1), (x2, y2)) for line endpoints in original coordinates

        # Drawing state
        self.mode = "mask"  # "mask" or "separator"
        self.drawing = False
        self.start_point = None
        self.current_point = None
        self.separator_start = None  # First point of separator line (display coords)

        # Window setup
        self.window_name = "Interactive Masker - Draw masks and separators"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        logger.info(f"Loaded image: {image_path} ({w}x{h})")
        logger.info(f"Display scale: {self.display_scale:.2f}")
        self.print_help()

    def print_help(self):
        """Print usage instructions."""
        print("\n" + "="*60)
        print("INTERACTIVE MASKING TOOL")
        print("="*60)
        print("Controls:")
        print("  - 'v' : Toggle between MASK mode and SEPARATOR mode")
        print("")
        print("MASK MODE (draw white rectangles):")
        print("  - Click and drag to draw white rectangles over unwanted areas")
        print("")
        print("SEPARATOR MODE (draw separator lines):")
        print("  - Click first point, then click second point to draw line")
        print("  - Lines can be tilted to match column angles")
        print("")
        print("General:")
        print("  - 's' : Save image with masks and separators")
        print("  - 'u' : Undo last action (mask or separator)")
        print("  - 'c' : Clear all masks and separators")
        print("  - 'h' : Show this help")
        print("  - 'q' : Quit without saving")
        print("="*60 + "\n")

    def scale_point(self, x, y, to_original=True):
        """Convert between display and original coordinates."""
        if to_original:
            return int(x / self.display_scale), int(y / self.display_scale)
        else:
            return int(x * self.display_scale), int(y * self.display_scale)

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing rectangles and separators."""
        if self.mode == "mask":
            # MASK MODE: Click and drag to draw rectangles
            if event == cv2.EVENT_LBUTTONDOWN:
                # Start drawing
                self.drawing = True
                self.start_point = (x, y)
                self.current_point = (x, y)

            elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing:
                    # Update current point
                    self.current_point = (x, y)

            elif event == cv2.EVENT_LBUTTONUP:
                # Finish drawing
                self.drawing = False
                self.current_point = (x, y)

                # Convert to original coordinates
                x1, y1 = self.scale_point(*self.start_point, to_original=True)
                x2, y2 = self.scale_point(*self.current_point, to_original=True)

                # Ensure x1 < x2 and y1 < y2
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                # Add rectangle
                self.rectangles.append((x1, y1, x2, y2))
                logger.info(f"Added rectangle: ({x1}, {y1}) to ({x2}, {y2})")

                # Apply masks and separators
                self.apply_all()

                # Reset drawing state
                self.start_point = None
                self.current_point = None

        elif self.mode == "separator":
            # SEPARATOR MODE: Click two points to draw a (possibly tilted) line
            if event == cv2.EVENT_LBUTTONDOWN:
                if self.separator_start is None:
                    # First click - start point
                    self.separator_start = (x, y)
                    logger.info(f"Separator start point: ({x}, {y}) - click second point")
                else:
                    # Second click - end point, create the separator
                    x1_orig, y1_orig = self.scale_point(*self.separator_start, to_original=True)
                    x2_orig, y2_orig = self.scale_point(x, y, to_original=True)

                    # Add separator as two endpoints
                    self.separators.append(((x1_orig, y1_orig), (x2_orig, y2_orig)))
                    logger.info(f"Added separator: ({x1_orig}, {y1_orig}) to ({x2_orig}, {y2_orig})")

                    # Reset separator start
                    self.separator_start = None

                    # Apply masks and separators
                    self.apply_all()

            elif event == cv2.EVENT_MOUSEMOVE:
                # Update current point for preview while drawing separator
                if self.separator_start is not None:
                    self.current_point = (x, y)

    def apply_all(self):
        """Apply all rectangle masks and separators to the image."""
        self.working_image = self.original.copy()

        # Apply white rectangle masks
        for x1, y1, x2, y2 in self.rectangles:
            cv2.rectangle(self.working_image, (x1, y1), (x2, y2), (255, 255, 255), -1)

        # Apply aqua separators (8px wide for clear OCR guidance)
        for sep in self.separators:
            (x1, y1), (x2, y2) = sep
            cv2.line(self.working_image, (x1, y1), (x2, y2), (255, 255, 0), 8)  # Aqua/cyan, 8px wide

    def get_display_image(self):
        """Get current display image with masks, separators, and preview."""
        # Start with working image (with applied masks and separators)
        if self.display_scale != 1.0:
            h, w = self.original.shape[:2]
            display_w = int(w * self.display_scale)
            display_h = int(h * self.display_scale)
            display = cv2.resize(self.working_image, (display_w, display_h))
        else:
            display = self.working_image.copy()

        # Draw preview rectangle if currently drawing in mask mode
        if self.mode == "mask" and self.drawing and self.start_point and self.current_point:
            x1, y1 = self.start_point
            x2, y2 = self.current_point
            # Draw semi-transparent preview
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Draw preview line if currently drawing separator
        if self.mode == "separator" and self.separator_start and self.current_point:
            x1, y1 = self.separator_start
            x2, y2 = self.current_point
            # Draw preview line
            cv2.line(display, (x1, y1), (x2, y2), (255, 255, 0), 2)
            # Draw start point marker
            cv2.circle(display, (x1, y1), 5, (0, 255, 0), -1)

        # Add instruction text
        font = cv2.FONT_HERSHEY_SIMPLEX
        mode_color = (0, 255, 0) if self.mode == "mask" else (255, 255, 0)  # Green for mask, cyan for separator
        mode_text = "MASK MODE" if self.mode == "mask" else "SEPARATOR MODE"

        text1 = f"Mode: {mode_text} (press 'v' to toggle)"
        text2 = f"Masks: {len(self.rectangles)} | Separators: {len(self.separators)} | Press 'h' for help"

        cv2.putText(display, text1, (10, 30), font, 0.7, mode_color, 2)
        cv2.putText(display, text2, (10, 60), font, 0.6, (0, 255, 0), 2)

        return display

    def undo_last(self):
        """Remove the last action (rectangle or separator)."""
        # If currently drawing a separator, cancel it first
        if self.mode == "separator" and self.separator_start is not None:
            self.separator_start = None
            self.current_point = None
            logger.info("Cancelled separator in progress")
            return

        # Remove the most recent action based on current mode
        if self.mode == "mask" and self.rectangles:
            removed = self.rectangles.pop()
            logger.info(f"Undid rectangle: {removed}")
            self.apply_all()
        elif self.mode == "separator" and self.separators:
            removed = self.separators.pop()
            (x1, y1), (x2, y2) = removed
            logger.info(f"Undid separator: ({x1}, {y1}) to ({x2}, {y2})")
            self.apply_all()
        else:
            logger.warning(f"No {self.mode}s to undo")

    def clear_all(self):
        """Clear all rectangles and separators."""
        rect_count = len(self.rectangles)
        sep_count = len(self.separators)

        if rect_count > 0 or sep_count > 0:
            logger.info(f"Cleared {rect_count} rectangles and {sep_count} separators")
            self.rectangles = []
            self.separators = []
            self.separator_start = None
            self.current_point = None
            self.working_image = self.original.copy()
        else:
            logger.warning("Nothing to clear")

    def save(self):
        """Save masked image with separators and coordinates."""
        if not self.rectangles and not self.separators:
            logger.warning("No masks or separators to save")
            return False

        # Generate output paths
        stem = self.image_path.stem
        parent = self.image_path.parent

        output_image_path = parent / f"{stem}_masked_divided.JPG"
        output_coords_path = parent / f"{stem}_mask_coords.json"

        # Save masked image (with both masks and separators)
        cv2.imwrite(str(output_image_path), self.working_image)
        logger.success(f"Saved image to: {output_image_path}")

        # Save coordinates
        coords_data = {
            "image_path": str(self.image_path),
            "image_size": {
                "width": self.original.shape[1],
                "height": self.original.shape[0]
            },
            "rectangles": [
                {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                for x1, y1, x2, y2 in self.rectangles
            ],
            "separators": [
                {"x1": p1[0], "y1": p1[1], "x2": p2[0], "y2": p2[1]}
                for p1, p2 in self.separators
            ]
        }

        with open(output_coords_path, 'w') as f:
            json.dump(coords_data, f, indent=2)
        logger.success(f"Saved coordinates to: {output_coords_path}")

        return True

    def load_coords(self, coords_path: Path):
        """Load rectangle and separator coordinates from JSON file."""
        if not coords_path.exists():
            logger.warning(f"Coordinates file not found: {coords_path}")
            return False

        with open(coords_path, 'r') as f:
            coords_data = json.load(f)

        # Validate image size matches
        expected_size = (coords_data["image_size"]["width"], coords_data["image_size"]["height"])
        actual_size = (self.original.shape[1], self.original.shape[0])

        if expected_size != actual_size:
            logger.warning(f"Image size mismatch: expected {expected_size}, got {actual_size}")
            return False

        # Load rectangles
        self.rectangles = [
            (rect["x1"], rect["y1"], rect["x2"], rect["y2"])
            for rect in coords_data.get("rectangles", [])
        ]

        # Load separators (if present) - handle both old and new formats
        raw_separators = coords_data.get("separators", [])
        self.separators = []
        h = self.original.shape[0]

        for sep in raw_separators:
            if isinstance(sep, int):
                # Old format: just x-coordinate, convert to full-height vertical line
                self.separators.append(((sep, 0), (sep, h)))
            elif isinstance(sep, dict):
                # New format: dict with x1, y1, x2, y2
                self.separators.append(((sep["x1"], sep["y1"]), (sep["x2"], sep["y2"])))
            elif isinstance(sep, (list, tuple)) and len(sep) == 2:
                # New format as tuple/list: ((x1, y1), (x2, y2))
                self.separators.append((tuple(sep[0]), tuple(sep[1])))

        logger.info(f"Loaded {len(self.rectangles)} rectangles and {len(self.separators)} separators from {coords_path}")
        self.apply_all()
        return True

    def run(self):
        """Run the interactive masking tool."""
        # Try to load existing coordinates
        coords_path = self.image_path.parent / f"{self.image_path.stem}_mask_coords.json"
        if coords_path.exists():
            self.load_coords(coords_path)

        while True:
            # Display image
            display = self.get_display_image()
            cv2.imshow(self.window_name, display)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                logger.info("Quitting without saving")
                break

            elif key == ord('s'):
                if self.save():
                    logger.info("Saved successfully. Press 'q' to quit or continue editing.")

            elif key == ord('u'):
                self.undo_last()

            elif key == ord('c'):
                self.clear_all()

            elif key == ord('h'):
                self.print_help()

            elif key == ord('v'):
                # Toggle mode
                self.mode = "separator" if self.mode == "mask" else "mask"
                # Reset any in-progress drawing
                self.separator_start = None
                self.current_point = None
                self.drawing = False
                self.start_point = None
                logger.info(f"Switched to {self.mode.upper()} mode")

        cv2.destroyAllWindows()


def main():
    if len(sys.argv) < 2:
        print("Usage: python interactive_masker.py <image_path>")
        print("Example: python interactive_masker.py 'IMG_5527 copy.JPG'")
        sys.exit(1)

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        # Try relative to script directory
        script_dir = Path(__file__).parent
        image_path = script_dir / sys.argv[1]

    if not image_path.exists():
        logger.error(f"Image not found: {sys.argv[1]}")
        sys.exit(1)

    masker = InteractiveMasker(image_path)
    masker.run()


if __name__ == "__main__":
    main()
