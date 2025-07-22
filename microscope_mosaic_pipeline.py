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
   - build_gaussian_pyramid: Creates a gaussian pyramid of an image.
   - build_laplacian_pyramid: Creates a laplacian pyramid of an image.
   - fuse_pyramids(): Fuses Laplacian pyramids based on a focus decision map.

Usage
---

Parameters
---
Focus Stacking:
   kernel_size : int, optional
      Size of the kernel for the SML focus measure (default is 3).
   levels : int, optional
      Number of levels in the Gaussian/Laplacian pyramids (default is 6).

Dependencies
---
-   cv2 (OpenCV)
-   numpy
-   typing
"""

import cv2
import numpy as np
from typing import List

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

def build_gaussian_pyramid(image: np.ndarray, levels: int = 6) -> List[np.ndarray]:
   """
   Build a Gaussian pyramid by repeatedly blurring and downsampling.

   Args:
      image: Input image
      levels: Number of pyramid levels
   
   Returns
      List of pyramid levels, from original (level 0) to smallest
   """
   pyramid = [image.astype(np.float64)]
   
   for _ in range(levels - 1):
      # Blur with Gaussian filter
      blurred = cv2.GaussianBlur(pyramid[-1], (5, 5), 1.0)

      # Downsample by factor of 2
      downsampled = blurred[::2, ::2]
      pyramid.append(downsampled)

   return pyramid

def build_laplacian_pyramid(image: np.ndarray, levels: int = 6) -> List[np.ndarray]:
   """
   Build a Laplacian pyramid from a Gaussian pyramid.
   
   The Laplacian pyramid contains the high-frequency details at each scale.
   
   Args:
      image: Input image
      levels: Number of pyramid levels
   
   Returns:
      List of Laplacian pyramid levels
   """
   # First build the Gaussian pyramid
   gaussian_pyramid = build_gaussian_pyramid(image, levels)
   laplacian_pyramid = []
   
   # Each level is the difference between the Gaussian level
   # and the upsampled version of the next smaller level
   for i in range(levels - 1):
      # Get the dimensions of current level
      h, w = gaussian_pyramid[i].shape[:2]
      
      # Upsample the next level
      upsampled = cv2.resize(gaussian_pyramid[i + 1], (w, h),
                             interpolation=cv2.INTER_LINEAR)
      
      # Compute difference
      laplacian = gaussian_pyramid[i] - upsampled
      laplacian_pyramid.append(laplacian)

   # The last level is just the smallest Gaussian level
   laplacian_pyramid.append(gaussian_pyramid[-1])
   
   return laplacian_pyramid

def fuse_pyramids(z_stack: List[np.ndarray], decision_map: np.ndarray,
                  levels: int = 6) -> np.ndarray:
   """
   Fuse multiple images using Laplacian pyramid blending based on decision map.
   
   Args:
      z_stack: List of images to fuse
      decision_map: Map indication which image to use at each pixel
      levels: Number of pyramid levels

   Returns:
      Fused image combining in-focus regions from all images
   """
   # Ensure that the z_stack was passed to the function
   if not z_stack:
      raise ValueError("Z-stack cannot be empty")

   # Get info on the stack
   h, w = z_stack[0].shape[:2]
   n_channels = z_stack[0].shape[2] if len(z_stack[0].shape) == 3 else 1
   
   # Build Laplacian pyramids for all images
   pyramids = []
   for img in z_stack:
      if n_channels == 1 and len(img.shape) == 3:
         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      pyramids.append(build_laplacian_pyramid(img, levels))

   # Build Gaussian pyramid for decision map to match pyramid levels
   decision_pyramid = build_gaussian_pyramid(decision_map.astype(np.float64), levels)
   
   # Create fused pyramid by selecting from source pyramids based on decision map
   fused_pyramid = []
   for level in range(levels):
      # Get the dimensions of current level
      level_h, level_w = pyramids[0][level].shape[:2]
      
      # Initialize fused level
      if n_channels == 3:
         fused_level = np.zeros((level_h, level_w, n_channels), dtype=np.float64)
      else:
         fused_level = np.zeros((level_h, level_w), dtype=np.float64)
      
      # Round and clip decision values to valid indices
      decision_level = np.round(decision_pyramid[level]).astype(int)
      decision_level = np.clip(decision_level, 0, len(z_stack) - 1)

      # Copy pixels from appropriate source images
      for idx in range(len(z_stack)):
         mask = (decision_level == idx)
         if n_channels == 3:
            for c in range(n_channels):
               fused_level[mask, c] = pyramids[idx][level][mask, c]
         else:
            fused_level[mask] = pyramids[idx][level][mask]
      
      fused_pyramid.append(fused_level)
      
   # Reconstruct image from fused pyramid
   reconstructed = fused_pyramid[-1]
   for level in range(levels - 2, -1, -1):
      # Get dimensions of target level
      h, w = fused_pyramid[level].shape[:2]
      
      # Upsample and add
      upsampled = cv2.resize(reconstructed, (w, h), interpolation=cv2.INTER_LINEAR)
      reconstructed = upsampled + fused_pyramid[level]
   
   # Clip to valid range
   reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

   return reconstructed