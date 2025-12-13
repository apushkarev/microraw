# Microraw

Barebones RAW converter for research or DaVinci Resolve workflows.

Simple Python tool based on libraw for converting RAW/DNG/Linear DNG files to linear XYZ or ACES AP1/ACEScct 16-bit TIFFs.

Linear DNG support is handy for raw-preprocessors like DxO PureRaw which does optical and AI noise corrections

## Features

- **Minimal dependencies**: Built on rawpy (LibRaw), NumPy, and colour-science
- **Two conversion modes**:
  - LibRaw default matrices
  - Custom DCP ForwardMatrix profiles
- **Two output formats**:
  - Linear XYZ (D50 illuminant)
  - ACES AP1/ACEScct
- **Batch processing**: Process individual files or entire folders

## Requirements

- **Python**: 3.11 or later
- **Operating System**: macOS, Linux, or Windows

## Installation
0. Best to ask AI agent to do this
1. Clone or download this repository
2. Create a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Or if you prefer to be explicit:
   ```bash
   python3 -m pip install -r requirements.txt
   ```

## Usage

**Note**: Make sure your virtual environment is activated before running:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Basic Syntax

```bash
python convert.py --matrix <matrix_source> --cs <colorspace> -i <input>
```

### Arguments

- `--matrix`: Matrix source
  - `default` - Use LibRaw's built-in camera matrices
  - `ref/profile.json` - Use custom DCP ForwardMatrix profile
- `--cs`: Output color space
  - `xyz` - Linear XYZ (D50 illuminant)
  - `aces` - ACES AP1 primaries with ACEScct log encoding
- `-i`: Input file or folder
  - Single RAW file (`.NEF`, `.CR2`, `.ARW`, `.DNG`, etc.)
  - Folder containing RAW files (will process all)

### Examples

#### Convert single file using LibRaw default matrices to XYZ
```bash
python convert.py --matrix default --cs xyz -i image.NEF
```

#### Convert single file using custom profile to ACES
```bash
python convert.py --matrix ref/nz6.json --cs aces -i image.NEF
```

#### Batch convert folder to ACES
```bash
python convert.py --matrix ref/camera_profile.json --cs aces -i /path/to/raw/folder/
```

#### Convert Linear DNG with custom correction
```bash
python convert.py --matrix ref/camera_profile.json --cs xyz -i linear.dng
```

### Output Files

Output files are saved in the same directory as the input with the color space suffix:
- `image_xyz.tiff` - Linear XYZ output
- `image_aces.tiff` - ACES AP1/ACEScct output

All outputs are 16-bit uncompressed TIFFs.

## Custom Camera Profiles

Custom profiles are JSON files containing:

1. **ForwardMatrix** (required): 3x3 matrix for Camera RGB → XYZ (D50) conversion
2. **linearDngCorrection** (optional): 3-element vector `[R, G, B]` for Linear DNG correction

### Example Profile Format

```json
{
  "forwardMatrix": [
    [0.796100, 0.209700, -0.041500],
    [0.313400, 1.002200, -0.315600],
    [0.032300, -0.204400, 0.997300]
  ],
  "linearDngCorrection": [0.958865, 1.941213, 1.630617]
}
```

### Creating Profiles

1. create own DCP profile (lumariver, argyll+dcamprof)
2. Extract ForwardMatrix from the DCP file (dcpTool -d) *
3. (Optional) Calculate linearDngCorrection vector using ColorChecker measurements
4. Save as JSON in the `ref/` folder

* note forward matrix coefficients in dcp are saved in reverse order and in forward order in json profile

## Processing Pipeline

### LibRaw Default Mode (`--matrix default`)

1. Demosaic Bayer → Camera RGB
2. Apply camera white balance
3. Convert using LibRaw's built-in Camera → XYZ matrices
4. (Optional) Convert XYZ → ACES AP1/ACEScct if `--cs aces`

### Custom Matrix Mode (`--matrix path/to/json`)

1. Demosaic Bayer → Camera RGB
2. Apply camera white balance (multiply by WB multipliers)
3. Apply linearDngCorrection (if present in JSON)
4. Apply ForwardMatrix: Camera RGB → XYZ (D50)
5. (Optional) Convert XYZ → ACES AP1/ACEScct:
   - Chromatic adaptation: D50 → D60 (Bradford)
   - Convert to AP1 primaries
   - Apply ACEScct log encoding

### Linear DNG Processing

For pre-demosaiced Linear DNGs:
1. Read RGB data directly (no demosaic needed)
2. Apply linearDngCorrection vector
3. Apply ForwardMatrix: RGB → XYZ (D50)
4. Continue with color space conversion if needed

## Using Output in DaVinci Resolve

### XYZ Files
1. Import TIFF
2. Set **Input Color Space**: `XYZ (CIE)`
3. Set **Input Gamma**: `Linear`

### ACES Files
1. Import TIFF
2. Set **Input Color Space**: `ACES (AP1)`
3. Set **Input Gamma**: `ACEScct`
4. **Important**: Disable "Apply Forward OOTF" in Color Space Transform nodes
5. I prefer to disable "Use White Point Adaptation" option

## Troubleshooting

### Colors look wrong
- Verify you're using the correct `--matrix` for your camera
- Check that Input Color Space settings match in DaVinci Resolve
- Ensure "Apply Forward OOTF" is disabled for log-encoded footage

## Technical Details

- **White Balance**: Applied by multiplying camera RGB by WB multipliers
- **Demosaic Algorithm**: AHD (Adaptive Homogeneity-Directed)
- **Color Primaries**: 
  - XYZ: CIE 1931
  - ACES: AP1 working space
- **Log Encoding**: ACEScct
- **Bit Depth**: 16-bit unsigned integer output

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Built on [rawpy](https://github.com/letmaik/rawpy) (LibRaw Python bindings)
- Color science powered by [colour-science](https://www.colour-science.org/)
