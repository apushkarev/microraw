#!/usr/bin/env python3
import sys
import json
import argparse
import rawpy
import numpy as np
import tifffile
import os
from pathlib import Path
import colour

def load_camera_matrix(json_path):
    """Load camera color matrix and optional linear DNG correction from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Prefer ForwardMatrix (Camera RGB -> XYZ directly)
    # Otherwise use inverted ColorMatrix (XYZ -> Camera RGB, so invert it)
    if 'forwardMatrix' in data:
        print("  Using ForwardMatrix (Camera RGB -> XYZ)")
        matrix = np.array(data['forwardMatrix']).reshape(3, 3)
    else:
        print("  Using inverted ColorMatrix (XYZ -> Camera RGB)")
        color_matrix = np.array(data['colorMatrix']).reshape(3, 3)
        matrix = np.linalg.inv(color_matrix)
    
    # Load optional linear DNG correction coefficients
    linear_dng_correction = None
    if 'linearDngCorrection' in data:
        linear_dng_correction = np.array(data['linearDngCorrection'])
        print(f"  Found Linear DNG correction: R={linear_dng_correction[0]:.6f}, G={linear_dng_correction[1]:.6f}, B={linear_dng_correction[2]:.6f}")
    
    return matrix, linear_dng_correction


def convert_xyz_to_aces_ap1_acescct(xyz):
    """Convert XYZ (D50) to ACES AP1 with ACEScct log encoding (cinema standard).
    
    Args:
        xyz: Image in XYZ color space (D50 illuminant), normalized 0-1
        
    Returns:
        Image in ACES AP1 primaries with ACEScct log encoding, normalized 0-1
    """
    h, w, c = xyz.shape
    
    # Get Bradford chromatic adaptation matrix: D50 -> D60
    d50_xy = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50']
    d60_xy = colour.RGB_COLOURSPACES['ACES2065-1'].whitepoint
    d50 = colour.xy_to_XYZ(d50_xy)
    d60 = colour.xy_to_XYZ(d60_xy)
    
    # Get Bradford adaptation matrix
    bradford_matrix = colour.adaptation.matrix_chromatic_adaptation_VonKries(
        d50, d60, transform='Bradford'
    )
    
    # Apply chromatic adaptation to entire image at once
    xyz_flat = xyz.reshape(-1, 3)
    xyz_d60_flat = xyz_flat @ bradford_matrix.T
    xyz_d60 = xyz_d60_flat.reshape(h, w, 3)
    
    # Get XYZ to ACES AP1 conversion matrix (ACEScc/ACEScct use AP1, not AP0)
    xyz_to_ap1 = colour.RGB_COLOURSPACES['ACEScg'].matrix_XYZ_to_RGB
    
    # Convert XYZ (D60) to ACES AP1 linear
    aces_ap1_flat = xyz_d60_flat @ xyz_to_ap1.T
    aces_ap1_linear = aces_ap1_flat.reshape(h, w, 3)
    
    # Apply ACEScct log encoding (cinema standard with toe)
    aces_ap1_cct = colour.models.log_encoding_ACEScct(aces_ap1_linear)
    
    return aces_ap1_cct


def process_raw_image(dng_path, matrix, target_colorspace, linear_dng_correction=None, apply_wb=False, use_libraw_xyz=False):
    """Process DNG file and apply color transformation."""
    print("  Reading DNG...")
    with rawpy.imread(dng_path) as raw:
        # Get camera white balance multipliers
        wb_multipliers = np.array(raw.camera_whitebalance[:3])
        print(f"  Camera WB multipliers: {wb_multipliers}")
        
        # If using LibRaw XYZ, take a different path
        if use_libraw_xyz:
            print("  Using LibRaw built-in XYZ conversion...")
            xyz_libraw = raw.postprocess(
                use_camera_wb=True,
                use_auto_wb=False,
                no_auto_bright=True,
                output_bps=16,
                output_color=rawpy.ColorSpace.XYZ,
                gamma=(1, 1),
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD
            )
            # Normalize to 0-1
            xyz = xyz_libraw.astype(np.float64) / 65535.0
            
            # Apply target colorspace conversion if needed
            if target_colorspace == 'aces':
                print("  Converting XYZ to ACES AP1 + ACEScct...")
                # Note: LibRaw XYZ illuminant is unclear, but proceeding with D50 assumption
                print("  Chromatic adaptation: D50 -> D60 (Bradford)")
                output = convert_xyz_to_aces_ap1_acescct(xyz)
                print("  Applied ACEScct log encoding")
            else:
                output = xyz
            
            print("  Clipping and converting to 16-bit...")
            output_clipped = np.clip(output, 0, 1)
            output_16bit = (output_clipped * 65535).astype(np.uint16)
            return output_16bit
        
        # Check if pre-demosaiced (Linear DNG)
        is_linear_dng = raw.raw_pattern is None
        
        if is_linear_dng:
            print("  Detected pre-demosaiced (Linear) DNG")
            print("  Reading RGB data directly (no additional WB processing)...")
            # For Linear DNG, WB is already baked in - just read the data as-is
            rgb = raw.postprocess(
                use_camera_wb=False,
                use_auto_wb=False,
                no_auto_bright=True,
                output_bps=16,
                output_color=rawpy.ColorSpace.raw,
                gamma=(1, 1)
            )
        else:
            print("  Demosaicing Bayer pattern...")
            # Get raw RGB data with demosaicing
            rgb = raw.postprocess(
                use_camera_wb=False,
                use_auto_wb=False,
                no_auto_bright=True,
                output_bps=16,
                output_color=rawpy.ColorSpace.raw,
                gamma=(1, 1),
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD
            )
    
    print("  Converting to camera RGB (normalized)...")
    # Normalize to 0-1 range
    rgb_normalized = rgb.astype(np.float64) / 65535.0
    
    # For Linear DNGs, apply correction factors to match reference camera RGB
    if is_linear_dng and linear_dng_correction is not None:
        print("  Applying Linear DNG correction factors...")
        print(f"  Correction: R={linear_dng_correction[0]:.6f}, G={linear_dng_correction[1]:.6f}, B={linear_dng_correction[2]:.6f}")
        rgb_normalized = rgb_normalized * linear_dng_correction
        print("  White balance already applied in Linear DNG (skipping additional WB)")
    elif is_linear_dng:
        print("  Linear DNG detected but no correction coefficients found in camera profile")
        print("  Proceeding without Linear DNG correction...")
    elif apply_wb:
        print("  Applying white balance...")
        # Apply white balance (MULTIPLY by multipliers to amplify channels)
        wb_multipliers = wb_multipliers / wb_multipliers[1]  # Normalize to green=1.0
        print(f"  Normalized WB multipliers: {wb_multipliers}")
        print("  Multiplying by WB multipliers (to neutralize scene illuminant)")
        rgb_normalized = rgb_normalized * wb_multipliers
        
        # Apply linearDngCorrection if available (for regular raws too)
        if linear_dng_correction is not None:
            print("  Applying linearDngCorrection...")
            print(f"  Correction: R={linear_dng_correction[0]:.6f}, G={linear_dng_correction[1]:.6f}, B={linear_dng_correction[2]:.6f}")
            rgb_normalized = rgb_normalized * linear_dng_correction
    else:
        print("  Skipping white balance (using raw camera RGB)...")
    
    print("  Converting camera RGB to XYZ...")
    # Apply color matrix to convert camera RGB to XYZ
    h, w, c = rgb_normalized.shape
    
    if matrix is None:
        # Use LibRaw's built-in XYZ conversion
        print("  Using LibRaw XYZ conversion (no custom matrix)")
        # For now, just pass through as-is (user should use libraw_xyz_convert.py for this)
        xyz = rgb_normalized
    else:
        rgb_flat = rgb_normalized.reshape(-1, 3)
        xyz_flat = rgb_flat @ matrix.T
        xyz = xyz_flat.reshape(h, w, 3)
    
    # Convert to target color space if needed
    if target_colorspace == 'aces':
        print("  Converting XYZ (D50) to ACES AP1 + ACEScct...")
        print("  Chromatic adaptation: D50 -> D60 (Bradford)")
        output = convert_xyz_to_aces_ap1_acescct(xyz)
        print("  Applied ACEScct log encoding")
    else:
        # XYZ output
        output = xyz
    
    print("  Clipping and converting to 16-bit...")
    # Clip and convert back to 16-bit
    output_clipped = np.clip(output, 0, 1)
    output_16bit = (output_clipped * 65535).astype(np.uint16)
    
    return output_16bit

def main():
    parser = argparse.ArgumentParser(
        description='Convert RAW files to XYZ or ACES color spaces',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Convert using custom matrix
  %(prog)s --matrix ref/nz6.json --cs aces -i file.NEF
  
  # Convert using LibRaw default matrices
  %(prog)s --matrix default --cs xyz -i file.NEF
  
  # Process folder
  %(prog)s --matrix ref/nz6.json --cs aces -i /path/to/folder/
''')
    
    parser.add_argument('--matrix', required=True,
                        help='Path to camera matrix JSON or "default" for LibRaw matrices')
    parser.add_argument('--cs', dest='colorspace', required=True, choices=['xyz', 'aces'],
                        help='Output color space: xyz (linear XYZ D50) or aces (ACES AP1/ACEScct)')
    parser.add_argument('-i', dest='input_path', required=True,
                        help='Input RAW file or folder containing RAW files')
    
    args = parser.parse_args()
    
    # Load camera matrix
    if args.matrix.lower() == 'default':
        print("Using LibRaw default matrices")
        print("  Note: No ForwardMatrix loaded, will use LibRaw's built-in XYZ conversion")
        inverted_matrix = None
        linear_dng_correction = None
    else:
        print(f"Loading camera matrix from {args.matrix}")
        inverted_matrix, linear_dng_correction = load_camera_matrix(args.matrix)
    
    target_colorspace = args.colorspace.lower()
    input_path = args.input_path
    
    # Check if input is a folder or file
    input_path_obj = Path(input_path)
    
    if input_path_obj.is_dir():
        # Process all RAW files in the folder
        # Common RAW extensions from various manufacturers
        raw_extensions = [
            '*.dng', '*.DNG',  # Adobe/Generic
            '*.nef', '*.NEF',  # Nikon
            '*.cr2', '*.CR2', '*.cr3', '*.CR3',  # Canon
            '*.arw', '*.ARW',  # Sony
            '*.orf', '*.ORF',  # Olympus
            '*.rw2', '*.RW2',  # Panasonic
            '*.pef', '*.PEF',  # Pentax
            '*.raf', '*.RAF',  # Fujifilm
            '*.raw', '*.RAW',  # Generic
            '*.rwl', '*.RWL',  # Leica
        ]
        
        raw_files = []
        for pattern in raw_extensions:
            raw_files.extend(input_path_obj.glob(pattern))
        raw_files = sorted(raw_files)
        
        if not raw_files:
            print(f"Error: No RAW files found in folder {input_path}")
            sys.exit(1)
        
        print(f"\nFound {len(raw_files)} RAW file(s) in folder")
        print("="*60)
        
        for i, raw_path in enumerate(raw_files, 1):
            print(f"\n[{i}/{len(raw_files)}] Processing {raw_path.name}")
            output_path = raw_path.parent / f"{raw_path.stem}_{target_colorspace}.tiff"
            
            try:
                use_libraw = (inverted_matrix is None)
                output_image = process_raw_image(str(raw_path), inverted_matrix, target_colorspace, linear_dng_correction, apply_wb=True, use_libraw_xyz=use_libraw)
                print(f"  Saving to {output_path.name}")
                tifffile.imwrite(str(output_path), output_image)
                print(f"  ✓ Completed")
            except Exception as e:
                print(f"  ✗ Error processing {raw_path.name}: {e}")
                continue
        
        print("\n" + "="*60)
        print("All files processed!")
    else:
        # Process single file
        dng_path = input_path
        output_path = dng_path.rsplit('.', 1)[0] + f'_{target_colorspace}.tiff'
        
        print(f"Processing {dng_path} -> {target_colorspace}")
        use_libraw = (inverted_matrix is None)
        output_image = process_raw_image(dng_path, inverted_matrix, target_colorspace, linear_dng_correction, apply_wb=True, use_libraw_xyz=use_libraw)
        
        print(f"Saving to {output_path}")
        tifffile.imwrite(output_path, output_image)
        
        print("Done!")

if __name__ == "__main__":
    main()
    # Force exit to avoid scipy/numpy cleanup hang on macOS
    os._exit(0)
