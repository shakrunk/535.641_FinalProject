<div align="center">

# Microscope Mosaic Pipeline

Automated wide-field mosaic and multi-focus fusion pipeline for microscopy images. This project combines z-stack focus stacking with panoramic image stitching to create sharp, seamless microscopy mosaics.

<p align="center">
  <img src="ReadmeAssets/JHU Logo Padding.jpeg" alt="Johns Hopkins University" width="500">
  <img src="ReadmeAssets/RMLEB Logo Padding.png" alt="Rocky Mountain Lions Eye Bank" width="500">
</p>

Developed for course 535.641 Mathematical Methods at Johns Hopkins University, in collaboration with the Rocky Mountain Lion's Eye Bank (RMLEB).

</div>

## Authors

- Krishna A. Kumar
- Nicholas Merten

## Overview

The pipeline processes collections of microscopy z-stacks through two stages:

1. **Focus Stacking**: Combines multiple images at different focal depths into a single all-in-focus image using Laplacian pyramid fusion
2. **Mosaic Stitching**: Assembles the focused images into a seamless panoramic mosaic using feature matching and homography estimation

## Features

- **Advanced Focus Stacking**

  - Sum-Modified-Laplacian (SML) focus measure for accurate sharpness detection
  - Multi-resolution Laplacian pyramid fusion for smooth transitions
  - Handles both grayscale and color images

- **Robust Image Stitching** (In Development)
  - ORB feature detection and matching
  - RANSAC-based homography estimation
  - Seamless blending of overlapping regions

## License

Copyright © 2025 Krishna A. Kumar and Nicholas Merten. All rights reserved.

## Acknowledgments

- Rocky Mountain Lion's Eye Bank (RMLEB) for collaboration and domain expertise
- Johns Hopkins University Mathematical Methods course (535.641)

## Installation

### Prerequisites

- Python 3.8+
- OpenCV (cv2)
- NumPy

### Setup

```bash
# Clone the repository
git clone https://github.com/shakrunk/535.641_FinalProject
cd microscope-mosaic-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install opencv-python numpy pytest
```

## Usage

### Basic Focus Stacking

```python
from microscope_mosaic_pipeline import process_z_stack
import cv2

# Load your z-stack images
z_stack = [
    cv2.imread('image_z1.png'),
    cv2.imread('image_z2.png'),
    cv2.imread('image_z3.png')
]

# Process the z-stack
focused_image = process_z_stack(z_stack)

# Save the result
cv2.imwrite('focused_output.png', focused_image)
```

### Individual Components

```python
from microscope_mosaic_pipeline import (
    compute_sml,
    create_focus_decision_map,
    fuse_pyramids
)

# Compute focus measures
sml_map = compute_sml(image, kernel_size=3)

# Create decision map for z-stack
decision_map = create_focus_decision_map(z_stack)

# Fuse images using pyramids
fused = fuse_pyramids(z_stack, decision_map, levels=6)
```

## Technical Details

### Focus Stacking Algorithm

The focus stacking implementation uses:

- **Sum-Modified-Laplacian (SML)**: An edge-based focus measure that responds to sharp edges characteristic of in-focus regions
- **Laplacian Pyramid Fusion**: Multi-resolution blending that preserves fine details while ensuring smooth transitions between source images
- **Per-pixel Decision Maps**: Determines which image in the z-stack is sharpest at each pixel location

### Parameters

- `kernel_size`: Size of the SML kernel (default: 3)
- `levels`: Number of pyramid levels for fusion (default: 6)

## Project Structure

```
microscope-mosaic-pipeline/
├── microscope_mosaic_pipeline.py  # Main pipeline implementation
├── tests/
│   ├── test_focus_stacking.py     # Focus stacking unit tests
│   ├── test_stitching.py          # Stitching tests (in development)
│   └── test_data/                 # Test images
├── .gitignore
└── README.md
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=microscope_mosaic_pipeline

# Run specific test file
pytest tests/test_focus_stacking.py
```

## Development Status

- ✅ Focus stacking pipeline - Complete
- 🚧 Mosaic stitching - In development
- 📝 Documentation - Ongoing

## References

- Laplacian pyramid fusion technique based on established multi-resolution blending methods
- Sum-Modified-Laplacian focus measure for microscopy applications
