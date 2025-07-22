#!/usr/bin/env python3
"""Automated Wide-Field Mosaic and Multi-Focus Fusion of Microscopy Images.

This module implements a two-stage pipeline to generate a single, all-in-focus,
wide-field mosaic from a collection of microscopy z-stacks.  The process first
creates an all-in-focus image from each z-stack (focus stacking) and then
stitches these focused images together into a final panoramic mosaic.

This script was developed by Krishna A. Kumar and Nicholas Merten for the course
535.641 Mathematical Methods at Johns Hopkins University, in collaboration with
the Rocky Mountain Lion's Eye Bank (RMLEB). Copyright (2025)

Pipeline Stages
---
1. **Focus Stacking**: Uses a Laplacian pyramid fusion technique guided by a
   Sum-Modified-Laplacian (SML) focus measure to combine a z-stack of images
   into a single, sharp composite image.
2. **Mosaic Stitching**: Stitches the resulting all-in-focus images into a 
   seamless panorama using ORB feature detection, feature matching, and robust
   homography estimation with RANSAC.

Key Functions
---
Focus Stacking:
   - process_z_stack(): The complete focus stacking pipeline for a single z-stack
   - compute_sml(): Implements the Sum-Modified-Laplacian focus measure.
   - fuse_pyramids(): Fuses Laplacian pyramids based on a focus decision map.

Usage
---

Parameters
---
Focus Stacking:
   kernel_size : int, optional
      Size of the kernel for the SML focus measure (default is 3).

Dependencies
---
-   cv2 (OpenCV)
-   numpy
"""

import cv2
import numpy as np

# ============================================================================
# Stage 1: Per-location Depth-of-Field Extension (Focus Stacking)
# ============================================================================

def compute_sml(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
   """
   Compute Sum-Modified-Laplacian (SML) focus measure for an image.
   
   The SML is an edge-based focus measure that responds to sharp edges
   which are characteristic of in-focus regions.
   
   Args:
      image: Input grayscale image
      kernel_size: Size of the Laplacian kernel (odd number)
   
   Returns:
      SML map of the same size as input image
   """
   # Convert to grayscale if needed
   if len(image.shape) == 3:
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   else:
      gray = image.copy()

   # Convert ot float for precision
   gray = gray.astype(np.float64)
   
   # Compute modified Laplacian using separate kernels for x and y
   kernel_x = np.array([[0, 0, 0],
                        [1, -2, 1],
                        [0, 0, 0]], dtype=np.float64)
   kernel_y = kernel_x.T
   
   # Apply kernels
   lap_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
   lap_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)

   # Sum of absolute values (modified part of SML)
   sml = np.abs(lap_x) + np.abs(lap_y)

   # Apply averaging filter to reduce noise
   sml = cv2.boxFilter(sml, cv2.CV_64F, (kernel_size, kernel_size))

   return sml