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
from colour.algebra import table_interpolation_tetrahedral
from colour import CCS_ILLUMINANTS, xy_to_XYZ


def _expand_poly_d3(rgb):
    """
    Degree-3 polynomial expansion of per-pixel RGB into 19 features.
    Inline copy of poly_fm_lib.expand_poly_d3 — no external dependency.

    rgb     : (N, 3) ndarray
    Returns : (N, 19) ndarray

    Term order: R, G, B,
                R², G², B², RG, RB, GB,
                R³, G³, B³, R²G, R²B, RG², G²B, RB², GB², RGB
    """
    R = rgb[:, 0]
    G = rgb[:, 1]
    B = rgb[:, 2]

    return np.column_stack([
        R, G, B,
        R*R, G*G, B*B, R*G, R*B, G*B,
        R*R*R, G*G*G, B*B*B,
        R*R*G, R*R*B, R*G*G, G*G*B, R*B*B, G*B*B,
        R*G*B,
    ])


def _expand_nn(rgb, mode='B', epsilon=0.005):
    """
    Neutral-null basis expansion of per-pixel RGB.
    Inline copy of neutral_null_lib.expand_nn — no external dependency.

    mode A — 17 terms: linear + degree-2 + degree-3 neutral-null
    mode B — 19 terms: mode A + log(R/G), log(R/B)

    rgb     : (N, 3) ndarray
    Returns : (N, 17) or (N, 19) ndarray
    """
    R = rgb[:, 0]
    G = rgb[:, 1]
    B = rgb[:, 2]

    cols = [R, G, B]

    cols += [
        R * (R - G),
        R * (R - B),
        G * (G - B),
        G * (G - R),
        B * (B - R),
    ]

    cols += [
        R * R * (R - G),
        R * R * (R - B),
        G * G * (G - R),
        G * G * (G - B),
        B * B * (B - R),
        B * B * (B - G),
        R * G * (R - B),
        R * B * (R - G),
        G * B * (G - R),
    ]

    if mode == 'B':
        cols += [
            np.log((R + epsilon) / (G + epsilon)),
            np.log((R + epsilon) / (B + epsilon)),
        ]

    return np.column_stack(cols)


def load_profile(matrix_path):
    """
    Load profile JSON. Returns the parsed profile dict.
    Supports three formats:
      - type=spline: new spline format (loads .npz alongside)
      - illuminants=[...]: multi/single illuminant format
      - legacy: {forwardMatrix, jzazlut} wrapped into illuminants format
    """
    matrix_path = Path(matrix_path)

    with open(matrix_path, 'r') as f:
        profile = json.load(f)

    if profile.get('type') == 'spline' and 'mired_breakpoints' in profile:
        if 'jzazlut' in profile:
            npz_path = matrix_path.parent / profile['jzazlut']['spline_data']
            data = np.load(str(npz_path))
            profile['_lut_values'] = data['lut_values']          # (n, 65, 65, 65, 3) float32
        profile['_mired_breakpoints'] = np.array(profile['mired_breakpoints'], dtype=np.float64)
        profile['_fm_values']         = np.array(profile['forward_matrix_values'], dtype=np.float64)  # (n, 9)
        return profile

    if 'illuminants' not in profile:
        # Legacy format: wrap in new structure using D50 CCT (5003K)
        profile = {
            'illuminants': [{
                'cct': 5003,
                'label': 'D50',
                'forwardMatrix': profile['forwardMatrix'],
                'jzazlut': profile.get('jzazlut'),
            }]
        }

    return profile


def load_forward_matrix(matrix_path):
    """Load forward matrix from JSON profile (backward-compatible, uses first entry)."""
    profile = load_profile(matrix_path)
    fm = np.array(profile['illuminants'][0]['forwardMatrix'])
    return fm


# DNG CalibrationIlluminant tag values → CCT in Kelvin
_DNG_ILLUMINANT_CCT = {
    1: 5503,   # Daylight (approximate D55)
    2: 3800,   # Fluorescent (approximate)
    3: 2856,   # Tungsten
    17: 2856,  # Standard Light A
    18: 4874,  # Standard Light B
    19: 6774,  # Standard Light C
    20: 5503,  # D55
    21: 6504,  # D65
    22: 7504,  # D75
    23: 5003,  # D50
    24: 3200,  # ISO studio tungsten
}


def _parse_dng_rationals(values, cols=3):
    """Parse DNG SRATIONAL tag (flat num/den pairs) into a numpy array."""
    floats = [values[i] / values[i + 1] for i in range(0, len(values), 2)]
    rows = len(floats) // cols
    return np.array(floats).reshape(rows, cols)


def _estimate_cct_dng(raw_path):
    """
    Estimate scene CCT from DNG ColorMatrix1/2 + AsShotNeutral tags.
    Uses iterative matrix interpolation in mired space (per DNG spec).
    Returns CCT in Kelvin, or None if DNG tags are missing.
    """
    try:
        with tifffile.TiffFile(str(raw_path)) as tif:
            tags = tif.pages[0].tags

            if 50721 not in tags or 50728 not in tags:
                return None

            cm1 = _parse_dng_rationals(tags[50721].value)
            neutral = _parse_dng_rationals(tags[50728].value, cols=1).flatten()

            # Single-matrix DNG: solve directly
            if 50722 not in tags:
                xyz = np.linalg.solve(cm1, neutral)
                xy = xyz[:2] / xyz.sum()
                return float(colour.xy_to_CCT(xy, method='Hernandez 1999'))

            cm2 = _parse_dng_rationals(tags[50722].value)

            ci1_tag = tags.get(50778)
            ci2_tag = tags.get(50779)
            cct_lo = _DNG_ILLUMINANT_CCT.get(ci1_tag.value if ci1_tag else 17, 2856)
            cct_hi = _DNG_ILLUMINANT_CCT.get(ci2_tag.value if ci2_tag else 21, 6504)

            mired_lo = 1e6 / cct_lo
            mired_hi = 1e6 / cct_hi

            # Seed from CM1
            xyz = np.linalg.solve(cm1, neutral)
            xy = xyz[:2] / xyz.sum()
            cct = float(colour.xy_to_CCT(xy, method='Hernandez 1999'))

            # Iterate: blend matrices in mired space until CCT converges
            for _ in range(4):
                mired = 1e6 / max(cct, 1000)
                w = np.clip((mired - mired_lo) / (mired_hi - mired_lo), 0, 1)
                cm = (1 - w) * cm1 + w * cm2

                xyz = np.linalg.solve(cm, neutral)
                xy = xyz[:2] / xyz.sum()
                cct = float(colour.xy_to_CCT(xy, method='Hernandez 1999'))

            return cct

    except Exception:
        return None


def _estimate_cct_rawpy(raw):
    """
    Fallback CCT estimation for non-DNG raws using rawpy.
    Less accurate than the DNG approach (rawpy's color_matrix is libraw's
    internal matrix, not the original DNG ColorMatrix).
    """
    try:
        wb = np.array(raw.camera_whitebalance[:3], dtype=float)
        cm = raw.color_matrix[:3, :3].astype(float)

        neutral = 1.0 / (wb / wb[1])

        xyz_white = np.linalg.solve(cm, neutral)
        xy = xyz_white[:2] / xyz_white.sum()

        return float(colour.xy_to_CCT(xy, method='Hernandez 1999'))

    except Exception:
        return None


def estimate_scene_cct(raw, raw_path=None):
    """
    Estimate scene CCT. Tries DNG tags via tifffile first (accurate),
    falls back to rawpy (less accurate for non-DNG raws).
    Returns CCT in Kelvin, or None if estimation fails.
    """
    if raw_path is not None:
        cct = _estimate_cct_dng(raw_path)
        if cct is not None:
            return cct

    return _estimate_cct_rawpy(raw)


def _interpolate_spline(profile, cct):
    """
    Evaluate the spline profile at the given scene CCT.
    Fits CubicSpline on the stored raw values and evaluates at the scene mired.
    Returns (forward_matrix, jzazlut_config).
    jzazlut_config contains a pre-built '_table' key (no .cube file needed).
    """
    from scipy.interpolate import CubicSpline

    mired_bps = profile['_mired_breakpoints']
    fm_values = profile['_fm_values']          # (n, 9)
    lut_values = profile.get('_lut_values')    # (n, 65, 65, 65, 3) float32, or None

    mired = np.clip(1e6 / max(cct, 100.0), mired_bps[0], mired_bps[-1])

    # Forward matrix: 9 splines (3x3), 57 splines (3x19 poly), or nn variant
    is_nn   = profile.get('nn', False)
    is_poly = profile.get('poly', False)
    if is_nn:
        n_terms = profile.get('nn_terms', 17)
    elif is_poly:
        n_terms = 19
    else:
        n_terms = 3
    cs_fm = CubicSpline(mired_bps, fm_values)
    fm = cs_fm(mired).reshape(3, n_terms)

    # LUT: only interpolate if the profile includes a jzazlut
    if lut_values is not None:
        n = lut_values.shape[0]
        lut_flat = lut_values.reshape(n, -1).astype(np.float64)
        cs_lut = CubicSpline(mired_bps, lut_flat)
        table = cs_lut(mired).reshape(lut_values.shape[1:])
        table = np.clip(table, 0.0, 1.0)

        jzazlut_config = {
            'reference_luminance': profile['jzazlut']['reference_luminance'],
            'domain':              profile['jzazlut']['domain'],
            '_table':              table,
        }
    else:
        jzazlut_config = None

    return fm, jzazlut_config


def interpolate_for_cct(illuminants, cct):
    """
    Given a sorted list of illuminant dicts and a scene CCT,
    return (fm, jzazlut_lo, jzazlut_hi, weight) where weight blends lo→hi.
    weight=0 means use lo only, weight=1 means use hi only.
    Clamps to nearest if CCT is outside the profile range.
    """
    ccts = [e['cct'] for e in illuminants]

    if cct <= ccts[0]:
        e = illuminants[0]
        return np.array(e['forwardMatrix']), e.get('jzazlut'), None, 0.0

    if cct >= ccts[-1]:
        e = illuminants[-1]
        return np.array(e['forwardMatrix']), e.get('jzazlut'), None, 0.0

    # Find bracketing entries
    hi_idx = next(i for i, c in enumerate(ccts) if c >= cct)
    lo_idx = hi_idx - 1

    lo = illuminants[lo_idx]
    hi = illuminants[hi_idx]

    # Inverse-CCT (mired) interpolation weight
    inv_lo  = 1.0 / lo['cct']
    inv_hi  = 1.0 / hi['cct']
    inv_cct = 1.0 / cct

    weight = (inv_cct - inv_lo) / (inv_hi - inv_lo)
    weight = float(np.clip(weight, 0.0, 1.0))

    fm_lo = np.array(lo['forwardMatrix'])
    fm_hi = np.array(hi['forwardMatrix'])
    fm = (1.0 - weight) * fm_lo + weight * fm_hi

    return fm, lo.get('jzazlut'), hi.get('jzazlut'), weight


def _normalize_jab(jab, domain):
    """Map JzAzBz to [0, 1]³ using domain bounds. Accepts (N, 3) or (H, W, 3)."""

    norm = np.empty_like(jab)

    norm[..., 0] = jab[..., 0] / domain['jz'][1]

    az_range = domain['az'][1] - domain['az'][0]
    norm[..., 1] = (jab[..., 1] - domain['az'][0]) / az_range

    bz_range = domain['bz'][1] - domain['bz'][0]
    norm[..., 2] = (jab[..., 2] - domain['bz'][0]) / bz_range

    return norm


def _denormalize_jab(norm, domain):
    """Map [0, 1]³ back to JzAzBz using domain bounds. Accepts (N, 3) or (H, W, 3)."""

    jab = np.empty_like(norm)

    jab[..., 0] = norm[..., 0] * domain['jz'][1]

    az_range = domain['az'][1] - domain['az'][0]
    jab[..., 1] = norm[..., 1] * az_range + domain['az'][0]

    bz_range = domain['bz'][1] - domain['bz'][0]
    jab[..., 2] = norm[..., 2] * bz_range + domain['bz'][0]

    return jab


# ─── Parallel LUT worker ─────────────────────────────────────────────

def _lut_apply_worker(args):
    """
    Apply full JzAzBz LUT pipeline to a pixel chunk in a subprocess.
    Module-level for macOS spawn compatibility.
    Returns corrected (N, 3) XYZ array.
    """
    import numpy as _np

    chunk, lut_table, ref_lum, domain, cat_d50_to_d65, cat_d65_to_d50 = args

    from colour import XYZ_to_Jzazbz, Jzazbz_to_XYZ, LUT3D
    from colour.algebra import table_interpolation_tetrahedral

    lut = LUT3D(table=lut_table)

    xyz_d65 = chunk @ cat_d50_to_d65.T
    jab = XYZ_to_Jzazbz(xyz_d65 * ref_lum)

    norm = _normalize_jab(jab, domain)
    norm = _np.clip(norm, 0.0, 1.0)

    corrected_norm = lut.apply(norm, interpolator=table_interpolation_tetrahedral)
    corrected_norm = _np.clip(corrected_norm, 0.0, 1.0)

    corrected_jab = _denormalize_jab(corrected_norm, domain)
    corrected_d65 = Jzazbz_to_XYZ(corrected_jab) / ref_lum

    return corrected_d65 @ cat_d65_to_d50.T


def apply_jzazbz_lut_blended(xyz_image, jzazlut_lo, jzazlut_hi, weight, profile_dir=None):
    """
    Apply JzAzBz LUT correction with optional blending between two illuminant LUTs.

    If jzazlut_hi is None or weight==0, applies jzazlut_lo only.
    Otherwise applies both and blends: (1-w)*lo + w*hi.
    """
    corrected_lo = apply_jzazbz_lut(xyz_image, jzazlut_lo, profile_dir)

    if jzazlut_hi is None or weight == 0.0:
        return corrected_lo

    if weight == 1.0:
        return apply_jzazbz_lut(xyz_image, jzazlut_hi, profile_dir)

    corrected_hi = apply_jzazbz_lut(xyz_image, jzazlut_hi, profile_dir)

    return (1.0 - weight) * corrected_lo + weight * corrected_hi


def apply_jzazbz_lut(xyz_image, jzazlut_config, profile_dir=None):
    """
    Apply JzAzBz LUT correction to an XYZ image (Y=1 for white).
    Pixel array is split across all available CPU cores.

    xyz_image:      ndarray shape (H, W, 3) or (N, 3)
    jzazlut_config: dict with 'reference_luminance', 'domain', and either
                    '_table' (pre-built ndarray from spline eval) or
                    'file' + profile_dir (path to .cube file)
    profile_dir:    Path — directory containing the .cube file (not needed if '_table' present)
    """
    import multiprocessing

    ref_lum = jzazlut_config['reference_luminance']
    domain  = jzazlut_config['domain']

    if '_table' in jzazlut_config:
        lut_table = jzazlut_config['_table'].astype(np.float64)
    else:
        cube_path = Path(profile_dir) / jzazlut_config['file']
        lut_table = colour.read_LUT(str(cube_path)).table

    _obs = 'CIE 1931 2 Degree Standard Observer'
    _d50 = xy_to_XYZ(CCS_ILLUMINANTS[_obs]['D50'])
    _d65 = xy_to_XYZ(CCS_ILLUMINANTS[_obs]['D65'])
    cat_d50_to_d65 = matrix_chromatic_adaptation_VonKries(_d50, _d65, transform='Bradford')
    cat_d65_to_d50 = matrix_chromatic_adaptation_VonKries(_d65, _d50, transform='Bradford')

    orig_shape = xyz_image.shape
    xyz_flat = xyz_image.reshape(-1, 3)

    # Use one worker per 100K pixels, capped at cpu_count
    n_workers = min(multiprocessing.cpu_count(), max(1, len(xyz_flat) // 100_000))
    chunks = np.array_split(xyz_flat, n_workers, axis=0)

    worker_args = [
        (chunk, lut_table, ref_lum, domain, cat_d50_to_d65, cat_d65_to_d50)
        for chunk in chunks
    ]

    with multiprocessing.Pool(n_workers) as pool:
        results = pool.map(_lut_apply_worker, worker_args)

    return np.concatenate(results, axis=0).reshape(orig_shape)


def _print_matrix(label, matrix):
    """Print a 3x3 matrix in a readable CLI format."""
    print(f"  {label}:")
    for row in matrix:
        print(f"    [{row[0]:+.6f}  {row[1]:+.6f}  {row[2]:+.6f}]")


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


def process_raw(raw_path, profile=None, scene_cct=None,
                colorspace='xyz', output_dir=None, matrix_name='default'):
    """
    Process a single RAW file.

    Args:
        raw_path:    Path to RAW file
        profile:     Parsed profile dict with 'illuminants' list, or None for default LibRaw
        scene_cct:   Override scene CCT in Kelvin. If None, estimated from raw metadata.
        colorspace:  'xyz' or 'aces'
        output_dir:  Output directory (if None, use same as input)
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

        # Resolve FM and LUT from profile
        if profile is not None:
            profile_dir = profile.get('_dir')

            if scene_cct is None:
                scene_cct = estimate_scene_cct(raw, raw_path)
                if scene_cct is not None:
                    print(f"  Estimated scene CCT: {scene_cct:.0f} K")
                else:
                    if profile.get('type') == 'spline' and '_mired_breakpoints' in profile:
                        scene_cct = float(1e6 / np.mean(profile['_mired_breakpoints']))
                    else:
                        scene_cct = profile['illuminants'][0]['cct']
                    print(f"  Could not estimate CCT, using {scene_cct:.0f} K")
            else:
                print(f"  Scene CCT (manual override): {scene_cct:.0f} K")

            if profile.get('type') == 'spline' and '_mired_breakpoints' in profile:
                forward_matrix, jzazlut_lo = _interpolate_spline(profile, scene_cct)
                jzazlut_hi = None
                lut_weight = 0.0
                print(f"  Spline-interpolated FM + LUT for {scene_cct:.0f} K")
                if forward_matrix.shape[1] == 3:
                    _print_matrix("Interpolated forward matrix", forward_matrix)
                else:
                    print(f"  Interpolated forward matrix: {forward_matrix.shape} (polynomial degree-3)")
            else:
                illuminants = profile['illuminants']
                forward_matrix, jzazlut_lo, jzazlut_hi, lut_weight = interpolate_for_cct(illuminants, scene_cct)
                print(f"  Interpolated FM + LUT for {scene_cct:.0f} K (weight={lut_weight:.3f})")
                _print_matrix("Interpolated forward matrix", forward_matrix)

        else:
            forward_matrix = None
            jzazlut_lo     = None
            jzazlut_hi     = None
            lut_weight     = 0.0
            profile_dir    = None

        # Check if it's a linear DNG (already demosaiced)
        is_linear_dng = False
        try:
            if raw.raw_pattern is None or len(raw.raw_pattern.shape) == 0:
                is_linear_dng = True
            elif raw.num_colors > 3:
                is_linear_dng = False
            elif hasattr(raw, 'sizes') and raw.sizes.raw_width == raw.sizes.width * 3:
                is_linear_dng = True
        except Exception:
            pass

        if is_linear_dng:
            print(f"  Detected linear DNG (pre-demosaiced)")
            try:
                rgb_image = raw.raw_image_visible.astype(np.float64)
                if len(rgb_image.shape) == 2:
                    h, w = rgb_image.shape
                    if w % 3 == 0:
                        rgb_image = rgb_image.reshape((h, w // 3, 3))
            except Exception:
                is_linear_dng = False

        # Process based on matrix mode
        if forward_matrix is not None:
            print(f"  Demosaic + LibRaw WB...")
            rgb16 = raw.postprocess(
                use_camera_wb=True,
                use_auto_wb=False,
                output_color=rawpy.ColorSpace.raw,
                output_bps=16,
                gamma=(1, 1),
                no_auto_bright=True,
                user_flip=0
            )
            rgb_image = rgb16.astype(np.float64) / 65535.0

            print(f"  Using interpolated forward matrix")
            shape    = rgb_image.shape
            rgb_flat = rgb_image.reshape(-1, 3)

            if profile is not None and profile.get('nn', False):
                nn_mode    = profile.get('nn_mode', 'A')
                nn_epsilon = profile.get('nn_epsilon', 0.005)
                xyz_flat = _expand_nn(rgb_flat, mode=nn_mode, epsilon=nn_epsilon) @ forward_matrix.T
            elif forward_matrix.shape[1] == 19:
                xyz_flat = _expand_poly_d3(rgb_flat) @ forward_matrix.T
            else:
                xyz_flat = rgb_flat @ forward_matrix.T

            xyz_image = xyz_flat.reshape(shape)

        else:
            print(f"  LibRaw demosaic + WB + XYZ conversion...")
            xyz16 = raw.postprocess(
                use_camera_wb=True,
                use_auto_wb=False,
                output_color=rawpy.ColorSpace.XYZ,
                output_bps=16,
                gamma=(1, 1),
                no_auto_bright=True,
                user_flip=0
            )
            xyz_image = xyz16.astype(np.float64) / 65535.0
            print(f"  Using LibRaw built-in camera matrix")

        xyz_image = np.clip(xyz_image, 0, None)

        # Optional JzAzBz LUT correction
        if forward_matrix is not None and jzazlut_lo is not None:
            print(f"  Applying JzAzBz LUT correction (weight={lut_weight:.3f})...")
            xyz_image = apply_jzazbz_lut_blended(xyz_image, jzazlut_lo, jzazlut_hi, lut_weight, profile_dir)
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
        # output_scaled = output_image * 65535
        # output_16bit = np.clip(output_scaled, 0, None).astype(np.uint16)
        output_16bit = np.clip(output_image * 65535, 0, 65535).astype(np.uint16)
        
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
  # Use camera built-in matrix
  python convert.py --matrix default --cs xyz -i photo.NEF

  # Use multi-illuminant profile (CCT auto-detected from metadata)
  python convert.py --matrix ref/imx226.json --cs aces -i photo.ORF

  # Override CCT manually
  python convert.py --matrix ref/imx226.json --cs aces --cct 4690 -i photo.ORF

  # Batch process folder
  python convert.py --matrix ref/imx226.json --cs aces -i /path/to/folder/
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

    parser.add_argument(
        '--cct',
        type=float,
        default=None,
        help='Override scene CCT in Kelvin (default: auto-detect from raw metadata)'
    )

    args = parser.parse_args()

    # Load profile
    if args.matrix.lower() == 'default':
        profile     = None
        matrix_name = 'default'
    else:
        matrix_path = Path(args.matrix)
        if not matrix_path.exists():
            print(f"Error: Matrix file not found: {matrix_path}", file=sys.stderr)
            sys.exit(1)
        profile = load_profile(matrix_path)
        # Store the profile directory so process_raw can locate .cube files
        profile['_dir'] = matrix_path.parent
        matrix_name = matrix_path.stem

        if profile.get('type') == 'spline' and 'mired_breakpoints' in profile:
            n = len(profile['mired_breakpoints'])
            ccts_approx = [round(1e6 / m) for m in profile['mired_breakpoints']]
            print(f"Loaded spline profile: {matrix_path}")
            print(f"  {n} breakpoints — CCT range: {max(ccts_approx)}–{min(ccts_approx)} K")
        else:
            n = len(profile['illuminants'])
            ccts = [e['cct'] for e in profile['illuminants']]
            print(f"Loaded profile: {matrix_path}")
            print(f"  {n} illuminant(s): {ccts} K")

    # Process input
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

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

    for raw_file in raw_files:
        try:
            process_raw(
                raw_file,
                profile=profile,
                scene_cct=args.cct,
                colorspace=args.cs,
                output_dir=args.output,
                matrix_name=matrix_name
            )
        except Exception as e:
            print(f"Error processing {raw_file.name}: {e}", file=sys.stderr)
            continue

    print("-" * 60)
    print(f"\u2713 Completed: {len(raw_files)} file(s)")


if __name__ == '__main__':
    main()
