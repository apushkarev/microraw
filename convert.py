#!/usr/bin/env python3
"""
Minimal CLI RAW converter with custom matrix support
"""
import os
import sys
import warnings

# Suppress all warnings from colour-science before importing
os.environ['COLOUR_SCIENCE__SUPPRESS_WARNINGS'] = '1'
warnings.filterwarnings('ignore')

import argparse
import json
from pathlib import Path

import numpy as np
import rawpy
import tifffile
import colour
from colour import CCS_ILLUMINANTS, RGB_COLOURSPACES, xy_to_XYZ
from colour.models import log_encoding_ACEScct
from colour.adaptation import matrix_chromatic_adaptation_VonKries


def load_forward_matrix(matrix_path):
    """Load forward matrix from JSON profile"""
    with open(matrix_path, 'r') as f:
        profile = json.load(f)
    return np.array(profile['forwardMatrix'])

def convert_xyz_to_aces_ap1_acescct(xyz):
    """Convert XYZ (D50) to ACES AP1 with ACEScct log encoding.

    Accepts either an image of shape (H, W, 3) or a flat array of shape (N, 3).
    Returns an array with the same shape as the input.
    """
    # Remember original shape; work on flat (N,3)
    orig_shape = xyz.shape
    if xyz.ndim == 3:
        h, w, c = orig_shape
        assert c == 3, "Expected last dimension to be 3 (XYZ)"
    elif xyz.ndim == 2 and orig_shape[1] == 3:
        h = w = None  # flat input
    else:
        raise ValueError(f"convert_xyz_to_aces_ap1_acescct: unsupported shape {orig_shape}")

    # Bradford chromatic adaptation D50 -> D60
    d50_xy = CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50']
    d60_xy = RGB_COLOURSPACES['ACES2065-1'].whitepoint
    d50 = xy_to_XYZ(d50_xy)
    d60 = xy_to_XYZ(d60_xy)

    bradford = matrix_chromatic_adaptation_VonKries(d50, d60, transform='Bradford')

    xyz_flat = xyz.reshape(-1, 3)
    xyz_d60_flat = xyz_flat @ bradford.T

    # XYZ D60 -> ACEScg (AP1) linear
    xyz_to_ap1 = RGB_COLOURSPACES['ACEScg'].matrix_XYZ_to_RGB
    ap1_flat = xyz_d60_flat @ xyz_to_ap1.T

    # ACEScct encoding
    ap1_cct_flat = log_encoding_ACEScct(ap1_flat)

    # Reshape back to original
    if h is not None:
        return ap1_cct_flat.reshape(h, w, 3)
    else:
        return ap1_cct_flat


def process_raw(raw_path, forward_matrix=None, colorspace='xyz', output_dir=None, matrix_name='default'):
    """
    Process a single RAW file
    
    Args:
        raw_path: Path to RAW file
        forward_matrix: Custom 3x3 forward matrix (Camera RGB -> XYZ D50), or None for default
        colorspace: 'xyz' or 'aces'
        output_dir: Output directory (if None, use same as input)
        matrix_name: Name of matrix profile (without extension)
    """
    raw_path = Path(raw_path)
    
    if output_dir is None:
        output_dir = raw_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing: {raw_path.name}")
    
    # Open RAW file
    with rawpy.imread(str(raw_path)) as raw:
        # Check if it's a linear DNG (already demosaiced)
        # Linear DNGs have raw_pattern == None or raw_colors_visible > raw_colors
        is_linear_dng = False
        try:
            # Multiple ways to detect linear DNG
            if raw.raw_pattern is None or len(raw.raw_pattern.shape) == 0:
                is_linear_dng = True
            elif raw.num_colors > 3:  # More than 3 color planes usually means Bayer
                is_linear_dng = False
            elif hasattr(raw, 'sizes') and raw.sizes.raw_width == raw.sizes.width * 3:
                is_linear_dng = True
        except:
            pass
        
        if is_linear_dng:
            # Linear DNG - already demosaiced
            print(f"  Detected linear DNG (pre-demosaiced)")
            try:
                # Try to get RGB image directly
                rgb_image = raw.raw_image_visible.astype(np.float64)
                
                # Reshape if needed (some linear DNGs are flat)
                if len(rgb_image.shape) == 2:
                    h, w = rgb_image.shape
                    if w % 3 == 0:
                        rgb_image = rgb_image.reshape((h, w // 3, 3))
                        rgb_image = rgb_image.astype(np.float64)
            except:
                # Fallback to postprocessing
                is_linear_dng = False
        
        # Process based on matrix mode
        if forward_matrix is not None:
            # Custom matrix mode: demosaic to camera RGB, apply WB, then custom matrix
            print(f"  Demosaic + LibRaw WB...")

            # Set user_sat higher to preserve highlights after WB multiplication
            # WB can push values beyond white_level, use 4x headroom
            
            # user_sat = int(raw.white_level * 4)
            rgb16 = raw.postprocess(
                use_camera_wb=True,
                use_auto_wb=False,
                output_color=rawpy.ColorSpace.raw,
                output_bps=16,
                gamma=(1, 1),
                no_auto_bright=True,
                # adjust_maximum_thr=0.0,
                # user_sat=user_sat,
                user_flip=0
            )
            rgb_image = rgb16.astype(np.float64) / 65535.0
            
            # Use custom forward matrix
            cam_to_xyz = forward_matrix
            print(f"  Using custom forward matrix")
            
            # Apply forward matrix: Camera RGB -> XYZ D50
            shape = rgb_image.shape
            rgb_flat = rgb_image.reshape(-1, 3)
            xyz_flat = rgb_flat @ cam_to_xyz.T
            xyz_image = xyz_flat.reshape(shape)
        else:
            # Default matrix mode: let LibRaw do the full conversion to XYZ
            print(f"  LibRaw demosaic + WB + XYZ conversion...")
            # Set user_sat higher to preserve highlights after WB multiplication
            # user_sat = int(raw.white_level * 4)
            xyz16 = raw.postprocess(
                use_camera_wb=True,
                use_auto_wb=False,
                output_color=rawpy.ColorSpace.XYZ,
                output_bps=16,
                gamma=(1, 1),
                no_auto_bright=True,
                # adjust_maximum_thr=0.0,
                # user_sat=user_sat,
                user_flip=0
            )
            xyz_image = xyz16.astype(np.float64) / 65535.0
            print(f"  Using LibRaw built-in camera matrix")
        
        # Clip negative values
        xyz_image = np.clip(xyz_image, 0, None)
        
        # Convert to target colorspace
        if colorspace == 'xyz':
            # Keep as linear XYZ D50
            output_image = xyz_image
            print(f"  Output: XYZ D50 Linear")
        
        elif colorspace == 'aces':
            # Convert XYZ D50 -> ACES AP1 with ACEScct encoding
            # Proper chromatic adaptation D50 -> D60 is critical!
            
            print(f"  Converting XYZ D50 -> ACES AP1 / ACEScct")
            
            # Use colour-science XYZ_to_RGB with proper API
            # This handles: XYZ D50 -> CAT (D50->D60) -> ACES AP1 -> ACEScct encoding
            acescct_colorspace = colour.RGB_COLOURSPACES['ACEScct']
            
            # Use our function that accepts image or flat
            output_image = convert_xyz_to_aces_ap1_acescct(xyz_image)
            print(f"  Applied Bradford CAT (D50->D60) + ACEScct encoding")
        
        else:
            raise ValueError(f"Unknown colorspace: {colorspace}")
        
        # Convert to 16-bit
        # Don't clip upper values - both ACES (log-encoded) and XYZ (linear HDR) can exceed 1.0
        # Only clip negative values to 0
        output_scaled = output_image * 65535
        output_16bit = np.clip(output_scaled, 0, None).astype(np.uint16)
        
        # Build output filename: original_name + matrix_name + colorspace
        output_suffix = f"_{matrix_name}_{colorspace}"
        output_path = output_dir / f"{raw_path.stem}{output_suffix}.tif"
        tifffile.imwrite(
            output_path,
            output_16bit,
            photometric='rgb',
            compression='lzw'
        )
        
        print(f"  Saved: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Minimal CLI RAW converter with custom matrix support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use camera's built-in matrix, output XYZ
  python convert.py --matrix default --cs xyz -i photo.NEF
  
  # Use custom profile, output ACES
  python convert.py --matrix ref/nz6.json --cs aces -i photo.NEF
  
  # Batch process folder
  python convert.py --matrix ref/camera.json --cs aces -i /path/to/folder/
        """
    )
    
    parser.add_argument(
        '--matrix',
        required=True,
        help='Forward matrix: "default" or path to JSON profile'
    )
    
    parser.add_argument(
        '--cs',
        choices=['xyz', 'aces'],
        required=True,
        help='Output colorspace: xyz (linear) or aces (AP1/ACEScct)'
    )
    
    parser.add_argument(
        '-i',
        '--input',
        required=True,
        help='Input RAW file or folder'
    )
    
    parser.add_argument(
        '-o',
        '--output',
        help='Output directory (default: same as input)'
    )

    
    args = parser.parse_args()
    
    # Load forward matrix
    if args.matrix.lower() == 'default':
        forward_matrix = None
        matrix_name = 'default'
    else:
        matrix_path = Path(args.matrix)
        if not matrix_path.exists():
            print(f"Error: Matrix file not found: {matrix_path}", file=sys.stderr)
            sys.exit(1)
        forward_matrix = load_forward_matrix(matrix_path)
        matrix_name = matrix_path.stem  # Get filename without extension
        print(f"Loaded forward matrix from: {matrix_path}")
    
    # Process input
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    # Collect RAW files
    raw_extensions = {'.nef', '.cr2', '.cr3', '.arw', '.dng', '.raw', '.orf', '.rw2', '.raf'}
    
    if input_path.is_file():
        raw_files = [input_path]
    else:
        raw_files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in raw_extensions
        ]
        raw_files.sort()
    
    if not raw_files:
        print("No RAW files found!", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nFound {len(raw_files)} RAW file(s)")
    print(f"Colorspace: {args.cs.upper()}")
    print("-" * 60)
    
    # Process each file
    for raw_file in raw_files:
        try:
            process_raw(
                raw_file,
                forward_matrix=forward_matrix,
                colorspace=args.cs,
                output_dir=args.output,
                matrix_name=matrix_name
            )
        except Exception as e:
            print(f"Error processing {raw_file.name}: {e}", file=sys.stderr)
            continue
    
    print("-" * 60)
    print(f"✓ Completed: {len(raw_files)} file(s)")


if __name__ == '__main__':
    main()
