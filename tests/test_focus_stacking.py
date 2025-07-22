import pytest
import numpy as np
import cv2
from pathlib import Path
try:
  from microscope_mosaic_pipeline import (
    compute_sml,
    create_focus_decision_map,
    fuse_pyramids,
    process_z_stack
  )
except ImportError:
  # Define dummy functions if the source is not available (allows test file to be parsed w/o error)
  def compute_sml(image, kernel_size=3): return np.zeros_like(image, dtype=np.float64)
  def create_focus_decision_map(sml_maps): return np.zeros_like(sml_maps[0], dtype=int)
  def fuse_pyramids(laplacian_pyramids, decision_map): return laplacian_pyramids[0][0]
  
# Path to test data (real and synthetic)
TEST_DATA_DIR = Path(__file__).parent / "test_data"

## -------------------------------- ##
## Fixtures                         ##
## -------------------------------- ##

@pytest.fixture
def synthetic_images():
  """Creates a sharp and blurred synthetic image for testing."""
  # Create a sharp synthetic image with high-frequency details (checkerboard)
  sharp_img = np.zeros((128, 128), dtype=np.uint8)
  # Make a checkerboard pattern
  for i in range(0, 128, 16):
    for j in range(0, 128, 16):
      if (i // 16+j // 16) % 2 == 0:
        sharp_img[i:i+16,j:j+16] = 255

  # Create a programmatically blurred version of the sharp image
  blurred_img = cv2.GaussianBlur(sharp_img, (15, 15), 0)

  return sharp_img, blurred_img

## -------------------------------- ##
## Unit Tests                       ##
## -------------------------------- ##

def test_compute_sml(synthetic_images):
  """
  Tests compute_sml() for correct focus measurement.
  - Compute the SML score for both the sharp and blurred images.
  - Assert that the sharp image's SML score is higher than the blurred one.
  - Assert that the output SML map has the same dimensions as the input image.
  - Assert that the output SML map is a floating-point array.
  """
  sharp_img, blurred_img = synthetic_images
  
  # Compute SML score for sharp and blurred images
  sml_sharp = compute_sml(sharp_img)
  sml_blurred = compute_sml(blurred_img)

  # Assert SML score of sharp image is higher than blurred one
  assert np.sum(sml_sharp) > np.sum(sml_blurred)

  # Assert output SML map has the same dimensions as the input
  assert sml_sharp.shape == sharp_img.shape

  # assert output SML map is a floating-point array
  assert sml_sharp.dtype == np.float64

def test_create_focus_decision_map():
  """
  Tests create_focus_decision_map() for correct index mapping.
  - Create a list of three 2x2 SML maps; max value is different location for each
  - Define the expected 2x2 decision map where each pixel's value is the SML Value's index
  - Generate the actual decision map using the function.
  - Assert that the generated decision map is identical to the expected decision map.
  """
  # Create a list of three simple, known 2x2 SML maps
  sml_maps = [
    np.array([[10, 2], [3, 4]], dtype=np.float64),  # Max at (0,0) is 10 (index 0)
    np.array([[5, 20], [8, 1]], dtype=np.float64),  # Max at (0,1) is 20 (index 1)
    np.array([[1,2], [30, 40]], dtype=np.float64)   # Max at (1,0) is 30, (1,1) is 40 (index 2)
  ]

  # Define the expected 2x2 decision map
  expected_decision_map = np.array([[0, 1], [2, 2]], dtype=int)

  # Generate the actual decision map using the function
  actual_decision_map = create_focus_decision_map(sml_maps)

  # Assert that the generated map is identical to expected map
  assert np.array_equal(actual_decision_map, expected_decision_map)

def test_fuse_pyramids():
  """
  Test fuse_pyramids() for correct pixel fusion logic
  - Create two simple, known Laplacian pyramids.
  - Create a known 2x2 decision map.
  - Call fuse_pyramids() with known pyramids and decision map.
  - Manually calculate the expected fused result.
  - Assert that the function's output is the expected result.
  """
  # Create two simple, known Laplacian pyramids (e.g., one pyramid of all 1s, one of all 10s)
  pyramid_a = [np.ones((2, 2), dtype=np.float64)]       # Pyramid for image A
  pyramid_b = [np.full((2, 2), 10.0, dtype=np.float64)] # Pyramid for image B
  laplacian_pyramids = [pyramid_a, pyramid_b]

  # Create a known 2x2 decision map 
  decision_map = np.array([[0, 1], [1, 0]])             # Use pyramid A then B, then B then A

  # Call the function to fuse the pyramids
  fused_level = fuse_pyramids(laplacian_pyramids, decision_map)

  # Manually calculate the expected result
  expected_result = np.array([[1.0, 10.0], [10.0, 1.0]])
  
  # Assert that the function's output is the expected result
  assert np.array_equal(fused_level, expected_result)

# Test build_gaussian_pyramid() and build_laplacian_pyramid() for correct structure
# Create a sample 128x128 grayscale image
# Build a pyramid with a specific number of levels (e.g., 4)
# Assert that the returned pyramid is a list containing the correct number of images (e.g., 4)
# Assert that the dimensions of each image in the pyramid are halved at each level (128x128 -> 64x64 -> 32x32 -> 16x16)

## -------------------------------- ##
## Integration Tests                ##
## -------------------------------- ##

# Test the full process_z_stack() pipeline on predictable synthetic data
# Create a synthetic z-stack of two images:
#  - Image 1: Sharp on the left half, blurry on the right
#  - Image 2: Blurry on the left half, sharp on the right
# Process the synthetic z-stack using the main pipeline function
# Assert that the output fused image has the same dimensions as the input images
# Assert that the output data type is uint8
# Assert that the left half of the output image is primarily sourced from Image 1
# Assert that the right half of the output image is primarily sourced from Image 2

# Create a smoke test for the full process_z_stack() pipeline on real data
# Load a small, real-world z-stack from the TEST_DATA_DIR
# Run the complete process_z_stack() function on this data
# Assert that the function executes without raising an exception
# Assert that the returned image is a non-empty NumPy array with the expected dimensions and uint8 data type

## -------------------------------- ##
## Edge Case & Error Handling Tests ##
## -------------------------------- ##

# Test pipeline behavior with invalid inputs
# Assert that calling process_z_stack() with an empty list of images raises a ValueError
# Assert that calling process_z_stack() with only one image returns that same image without processing
# Assert that calling process_z_stack() with images of different dimensions raises a ValueError

# Test pipeline behavior with invalid parameters
# Assert that calling compute_sml() with an invalid kernel_size (e.g., 0, even number) raises a ValueError
# Assert that calling pyramid functions with an invalid number of levels (e.g., 0, negative) raises a ValueError