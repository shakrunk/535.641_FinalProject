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
    - create_focus_decision_map(): Creates a map of which image in a z-stack is sharpest
    - build_gaussian_pyramid(): Creates a gaussian pyramid of an image.
    - build_laplacian_pyramid(): Creates a laplacian pyramid of an image.
    - fuse_pyramids(): Fuses Laplacian pyramids based on a focus decision map.
Mosaic Stitching:
    - create_mosaic(): Sequentially stitches a list of multiple images.
    - detect_features_orb(): Detects ORB features in an image.
    - match_features(): Matches keypoint descriptors between two images.
    - estimate_homography(): Estimates image transformation with RANSAC.
    - warp_image(): Applies a perspective transformation to an image.
    - blend_images(): Blends multiple overlapping images into one.

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
from typing import List, Tuple, Optional, Union

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
    kernel_x = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float64)
    kernel_y = kernel_x.T

    # Apply kernels
    lap_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
    lap_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)

    # Sum of absolute values (modified part of SML)
    sml = np.abs(lap_x) + np.abs(lap_y)

    # Apply averaging filter to reduce noise
    sml = cv2.boxFilter(sml, cv2.CV_64F, (kernel_size, kernel_size))

    return sml


def create_focus_decision_map(z_stack: List[np.ndarray]) -> np.ndarray:
    """
    Create a decision map indicating which image in the z-stack
    has the highest focus measure at each pixel location.

    Args:
        z_stack: List of images from the same location at different focal depth

    Returns:
        Decision map where each pixel contains the index of the sharpest image
    """
    # Ensure the function receives a z-stack
    if not z_stack:
        raise ValueError("Z-stack cannot be empty")

    # Get stack info
    h, w = z_stack[0].shape[:2]

    # Initialize arrays to track maximum SML and corresponding image index
    max_sml = np.zeros((h, w), dtype=np.float64)
    decision_map = np.zeros((h, w), dtype=np.uint8)

    # Process each image in the z-stack
    for idx, image in enumerate(z_stack):
        sml = compute_sml(image)

        # Update decision map where current image his higher SML
        mask = sml > max_sml
        max_sml[mask] = sml[mask]
        decision_map[mask] = idx

    return decision_map


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
        upsampled = cv2.resize(
            gaussian_pyramid[i + 1], (w, h), interpolation=cv2.INTER_LINEAR
        )

        # Compute difference
        laplacian = gaussian_pyramid[i] - upsampled
        laplacian_pyramid.append(laplacian)

    # The last level is just the smallest Gaussian level
    laplacian_pyramid.append(gaussian_pyramid[-1])

    return laplacian_pyramid


def fuse_pyramids(
    z_stack: List[np.ndarray], decision_map: np.ndarray, levels: int = 6
) -> np.ndarray:
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
            mask = decision_level == idx
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


def process_z_stack(z_stack: List[np.ndarray]) -> np.ndarray:
    """
    Process a complete z-stack to create a single all-in-focus image.

    Args:
        z_stack: List of images from the same location at different focal depths

    Returns:
        Single composite image with extended depth of field
    """
    print(f"Processing z-stack with {len(z_stack)} images...")

    # Create focus decision map
    decision_map = create_focus_decision_map(z_stack)

    # Fuse images using Laplacian pyramid
    fused_image = fuse_pyramids(z_stack, decision_map)

    return fused_image


# ============================================================================
# Stage 2: Spacial Mosaic Stitching
# ============================================================================


def detect_features_orb(
    image: np.ndarray, n_features: int = 5000
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Detect ORB (Oriented FAST and Rotated BRIEF) features in an image.

    Args:
        image: Input image
        n_features: Maximum number of features to detect

    Returns:
        Tuple of (keypoints, descriptors)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Create ORB detector
    orb = cv2.ORB_create(
        nfeatures=n_features,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
    )

    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    return keypoints, descriptors


def match_features(
    desc1: np.ndarray, desc2: np.ndarray, ratio_threshold: float = 0.7
) -> List:
    """
    Match ORB features between two sets of descriptors using brute force matching.

    Args:
        desc1: Descriptors from first image
        desc2: Descriptors from second image
        ratio_threshold: Lowe's ratio test threshold

    Returns:
        List of good matches
    """
    # Create brute force matcher with Hamming distance (for binary descriptors)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Match descriptors
    matches = bf.knnMatch(desc1, desc2, k=2)

    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

    return good_matches


def estimate_homography(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    ransac_threshold: float = 5.0,
    confidence: float = 0.99,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Estimate homography between two images using RANSAC.

    Args:
        src_pts: NumPy array of source points (N, 2)
        dst_pts: NumPy array of destination points (N, 2)
        ransac_threshold: RANSAC re-projection error threshold in pixels
        confidence: Desired confidence level

    Returns:
        A tuple containing:
        - 3x3 homography matrix (or None if not found)
        - Inlier mask (or None if not found)
    """
    if len(src_pts) < 4:
        return None, None

    # Find homography using RANSAC
    # Note: cv2.findHomography expects points in shape (N, 1, 2)
    homography, mask = cv2.findHomography(
        src_pts.reshape(-1, 1, 2),
        dst_pts.reshape(-1, 1, 2),
        cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        confidence=confidence,
    )

    # Count inliers
    if mask is not None:
        inliers = np.sum(mask)
        print(f"RANSAC found {inliers}/{len(src_pts)} inliers")

    # The mask from findHomography is a column vector (N, 1) of uint8.
    return homography, mask


def warp_image(
    img: np.ndarray,
    homography: np.ndarray,
    output_shape: Tuple[int, int],
    return_mask: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Warps an image using a given homography matrix.

    Args:
        img: The image to be warped.
        homography: The 3x3 transformation matrix.
        output_shape: A tuple (height, width) for the output canvas.
        return_mask: If True, also returns a mask of the valid warped pixels.

    Returns:
        The warped image. If return_mask is True, returns a tuple of
        (warped_image, mask).
    """
    # Note: cv2.warpPerspective expects (width, height)
    output_wh = (output_shape[1], output_shape[0])

    warped_img = cv2.warpPerspective(img, homography, output_wh)

    if return_mask:
        # Create a mask of the same size as the input image
        if img.ndim > 2:  # Handle color or grayscale images
            mask_in = np.ones(img.shape[:2], dtype=np.uint8) * 255
        else:
            mask_in = np.ones_like(img, dtype=np.uint8) * 255

        # Warp the mask to find the valid pixel area in the output
        warped_mask = cv2.warpPerspective(
            mask_in, homography, output_wh, flags=cv2.INTER_NEAREST
        )
        return warped_img, warped_mask

    return warped_img


def blend_images(images: List[np.ndarray], masks: List[np.ndarray]) -> np.ndarray:
    """
    Blends multiple images together using feathered weighting.

    Args:
        images: A list of images to blend.
        masks: A list of corresponding masks (255 for valid pixels).

    Returns:
        A single blended image.
    """
    output_shape = images[0].shape
    blended_image = np.zeros(output_shape, dtype=np.float32)
    weight_sum = np.zeros(output_shape[:2], dtype=np.float32)

    for img, mask in zip(images, masks):
        # Use distance transform for smooth feathering at the edges
        # This creates a weight map that falls off towards the mask boundaries
        weights = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)

        # Add weights to the total weight map, handling multiple channels if necessary
        if img.ndim > 2:
            weights = cv2.cvtColor(weights, cv2.COLOR_GRAY2BGR)

        blended_image += img.astype(np.float32) * weights
        weight_sum += weights

    # Avoid division by zero
    # np.where is faster than creating a mask and indexing
    result = np.where(weight_sum > 0, blended_image / weight_sum, 0)

    return result.astype(np.uint8)

def create_mosaic(images: List[np.ndarray]) -> np.ndarray:
    """
    Stitch multiple images into a panoramic mosaic.
    
    Args:
        images: List of images to stitch
    
    Returns:
        Stitched panoramic image
    """
    if len(images) < 2:
        return images[0] if images else None
    
    print(f"Stitching {len(images)} images...")
    
    # Start with the first image
    result = images[0].copy()
    
    # Sequentially stitch each image
    for i in range(1, len(images)):
        print(f"Processing image {i+1}/{len(images)}")
        
        # Detect features
        kp1, desc1 = detect_features_orb(result)
        kp2, desc2 = detect_features_orb(images[i])
        
        if desc1 is None or desc2 is None:
            print(f"Warning: No features detected in image pair {i}")
            continue
        
        # Match features
        matches = match_features(desc1, desc2)
        print(f"Found {len(matches)} feature matches")
        
        if len(matches) < 10:
            print(f"Warning: Not enough matches for image {i+1}")
            continue
        
        # Estimate homography
        homography = estimate_homography(kp1, kp2, matches)
        
        if homography is None:
            print(f"Warning: Could not compute homography for image {i+1}")
            continue
        
    return result
