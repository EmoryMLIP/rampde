#!/usr/bin/env python3
"""
Extract subplots from CNF visualization images.

This tool splits CNF visualization images into their individual components:
- Target (optional)
- Samples
- Log Probability

Author: Claude
"""

import argparse
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np


def load_image(filepath):
    """Load image from file."""
    try:
        img = Image.open(filepath)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)


def detect_title_boundary(img_array):
    """Detect the boundary between title and content areas."""
    height = img_array.shape[0]
    
    # Strategy: Find where substantial plot content begins
    # We need to be aggressive since titles can extend quite far down
    
    # Look for substantial plot content - either purple background or lots of dark pixels
    best_boundary = int(height * 0.2)  # Default fallback
    
    for row in range(50, min(height // 2, 250)):  # Start checking from row 50
        row_pixels = img_array[row]
        
        # Method 1: Look for purple plot background
        purple_mask = (row_pixels[:, 0] < 120) & (row_pixels[:, 1] < 120) & (row_pixels[:, 2] > 80)
        purple_count = np.sum(purple_mask)
        purple_pct = purple_count / row_pixels.shape[0] * 100
        
        # Method 2: Look for substantial dark content (plot visualization)  
        dark_mask = np.any(row_pixels < 150, axis=1)
        dark_count = np.sum(dark_mask)
        dark_pct = dark_count / row_pixels.shape[0] * 100
        
        # Method 3: Look for very dark pixels (strong plot features)
        very_dark_mask = np.any(row_pixels < 100, axis=1)
        very_dark_count = np.sum(very_dark_mask)
        very_dark_pct = very_dark_count / row_pixels.shape[0] * 100
        
        # We found substantial plot content if any of these conditions are met:
        # - At least 5% purple pixels (typical plot background)
        # - At least 20% dark pixels (plot features)  
        # - At least 5% very dark pixels (strong plot features)
        if purple_pct >= 5 or dark_pct >= 20 or very_dark_pct >= 5:
            # Verify this is consistent by checking a few rows ahead
            consistent_rows = 0
            for check_row in range(row + 1, min(height, row + 10)):
                check_pixels = img_array[check_row]
                check_dark = np.sum(np.any(check_pixels < 150, axis=1))
                check_dark_pct = check_dark / check_pixels.shape[0] * 100
                if check_dark_pct >= 15:  # Consistent substantial content
                    consistent_rows += 1
            
            # If we have consistent plot content, this is our boundary
            if consistent_rows >= 3:
                return max(0, row - 2)  # Back off slightly to preserve a tiny margin
    
    # Fallback: If no clear plot boundary found, look for any substantial content
    for row in range(30, min(height // 3, 150)):
        row_pixels = img_array[row]
        content_mask = np.any(row_pixels < 200, axis=1)
        content_pct = np.sum(content_mask) / row_pixels.shape[0] * 100
        
        if content_pct >= 10:  # 10% content threshold
            return row
    
    # Final fallback
    return best_boundary


def remove_title_area(img):
    """Remove the title area from the top of the image."""
    img_array = np.array(img)
    title_boundary = detect_title_boundary(img_array)
    
    # Crop out the title area
    cropped_array = img_array[title_boundary:, :, :]
    return Image.fromarray(cropped_array)


def split_into_panels(img, skip_target=True):
    """Split image into three equal-width panels."""
    width, height = img.size
    panel_width = width // 3
    
    panels = []
    labels = []
    
    # Define panel boundaries
    panel_info = [
        ("target", 0, panel_width),
        ("samples", panel_width, 2 * panel_width),
        ("logp", 2 * panel_width, width)
    ]
    
    for label, x_start, x_end in panel_info:
        if label == "target" and skip_target:
            continue
        
        panel = img.crop((x_start, 0, x_end, height))
        panels.append(panel)
        labels.append(label)
    
    return panels, labels


def trim_whitespace(img, padding=1, debug=False):
    """Remove whitespace from image borders, but leave 1 pixel of original margin around content."""
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    
    if debug:
        print(f"  Original size: {width}x{height}")
    
    # Strategy: Find tight content boundaries, then back off by 1 pixel to preserve margin
    
    # Find non-white pixels - use a more moderate threshold
    # Consider anything below 250 as potentially content (allows for slight compression artifacts)
    is_white = np.all(img_array >= 250, axis=2)
    has_content = ~is_white
    
    # Also look for colored content (purple plot areas)
    # Purple areas: low red, low green, higher blue
    purple_content = (img_array[:, :, 0] < 200) & \
                     (img_array[:, :, 1] < 150) & \
                     (img_array[:, :, 2] > 100)
    
    # Combine both content detection methods
    content_mask = has_content | purple_content
    
    # Find content boundaries
    content_rows = np.any(content_mask, axis=1)
    content_cols = np.any(content_mask, axis=0)
    
    if not np.any(content_rows) or not np.any(content_cols):
        if debug:
            print("  No content found!")
        return img
    
    # Get tight content bounds
    row_indices = np.where(content_rows)[0]
    col_indices = np.where(content_cols)[0]
    
    tight_top = row_indices[0]
    tight_bottom = row_indices[-1] + 1
    tight_left = col_indices[0] 
    tight_right = col_indices[-1] + 1
    
    # Now back off by 1 pixel to preserve original margin (if possible)
    top = max(0, tight_top - padding)
    bottom = min(height, tight_bottom + padding)
    left = max(0, tight_left - padding)
    right = min(width, tight_right + padding)
    
    if debug:
        print(f"  Tight bounds: top={tight_top}, bottom={tight_bottom}, left={tight_left}, right={tight_right}")
        print(f"  Final bounds: top={top}, bottom={bottom}, left={left}, right={right}")
    
    # Crop with the margin-preserving bounds
    cropped = img_array[top:bottom, left:right]
    
    return Image.fromarray(cropped)


def save_panel(panel, output_path, quality=95):
    """Save panel image with high quality."""
    # Ensure RGB mode for JPEG
    if panel.mode != 'RGB':
        panel = panel.convert('RGB')
    
    panel.save(output_path, 'JPEG', quality=quality, optimize=True)
    print(f"Saved: {output_path}")


def extract_components(input_path, skip_target=True, output_dir=None, prefix=None, target_samples=False, target_logp=False):
    """Extract components from CNF visualization image."""
    input_path = Path(input_path)

    # Validate input
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist.")
        sys.exit(1)

    # Setup output directory
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Setup output prefix
    if prefix is None:
        prefix = input_path.stem

    print(f"Processing: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Output prefix: {prefix}")
    print(f"Skip target: {skip_target}")
    print(f"Target samples: {target_samples}")
    print(f"Target logp: {target_logp}")

    # Load and process image
    img = load_image(input_path)
    print(f"Loaded image: {img.size[0]}x{img.size[1]}")

    # Handle specific panel extraction modes
    if target_samples or target_logp:
        width, height = img.size
        panel_width = width // 3

        if target_samples:
            # Extract the left panel (target samples)
            panel = img.crop((0, 0, panel_width, height))
            suffix = "target"
            print("Extracting target samples (left panel)")
        elif target_logp:
            # Extract the right panel (log probability)
            panel = img.crop((2 * panel_width, 0, width, height))
            suffix = "logp"
            print("Extracting log probability (right panel)")

        # Remove title area from panel
        panel_no_title = remove_title_area(panel)

        # Trim whitespace
        trimmed = trim_whitespace(panel_no_title)

        # Generate output filename
        output_file = output_dir / f"{prefix}-{suffix}.jpg"

        # Save
        save_panel(trimmed, output_file)
        print(f"Single panel extraction complete!")
        return

    # Normal extraction mode
    # Split into panels first (don't remove title from full image)
    panels, labels = split_into_panels(img, skip_target)
    print(f"Split into {len(panels)} panels")

    # Process and save each panel
    for panel, label in zip(panels, labels):
        # Remove title area from each individual panel
        panel_no_title = remove_title_area(panel)

        # Trim whitespace
        trimmed = trim_whitespace(panel_no_title)

        # Generate output filename
        output_file = output_dir / f"{prefix}-{label}.jpg"

        # Save
        save_panel(trimmed, output_file)

    print("Extraction complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Extract components from CNF visualization images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i cnf-viz-00000.jpg
  %(prog)s -i cnf-viz-00000.jpg --no-skip-target
  %(prog)s -i input.jpg -o output_folder --prefix mytest
  %(prog)s -i cnf-viz-01000.jpg --target-samples --prefix 2spirals_float32_torchdiffeq
  %(prog)s -i cnf-viz-01000.jpg --target-logp --prefix 2spirals_float32_torchdiffeq
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input CNF visualization image'
    )
    
    parser.add_argument(
        '--skip-target',
        action='store_true',
        default=True,
        help='Skip extracting the target panel (default: True)'
    )
    
    parser.add_argument(
        '--no-skip-target',
        action='store_false',
        dest='skip_target',
        help='Extract the target panel'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        help='Output directory (default: same as input)'
    )
    
    parser.add_argument(
        '--prefix',
        help='Custom prefix for output files (default: input basename)'
    )

    parser.add_argument(
        '--target-samples',
        action='store_true',
        help='Extract only the target samples (left panel) - actual sample points'
    )

    parser.add_argument(
        '--target-logp',
        action='store_true',
        help='Extract only the log probability (right panel) - density visualization'
    )

    args = parser.parse_args()

    # Validate arguments
    exclusive_flags = [args.target_samples, args.target_logp]
    if sum(exclusive_flags) > 1:
        parser.error("Only one of --target-samples or --target-logp can be specified")

    # Run extraction
    extract_components(
        args.input,
        skip_target=args.skip_target,
        output_dir=args.output_dir,
        prefix=args.prefix,
        target_samples=args.target_samples,
        target_logp=args.target_logp
    )


if __name__ == '__main__':
    main()