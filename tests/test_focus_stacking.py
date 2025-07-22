import pytest
import numpy as np
import cv2
from pathlib import Path
from microscope_mosaic_pipeline import create_focus_decision_map

# Define dummy functions if the source is not available (allows test file to be parsed w/o error)
try:
    from microscope_mosaic_pipeline import compute_sml
except ImportError:

    def compute_sml(image, kernel_size=3):
        return np.zeros_like(image, dtype=np.float64)


try:
    from microscope_mosaic_pipeline import create_focus_decision_map
except ImportError:

    def create_focus_decision_map(sml_maps):
        return np.zeros_like(sml_maps[0], dtype=int)


try:
    from microscope_mosaic_pipeline import fuse_pyramids
except ImportError:

    def fuse_pyramids(laplacian_pyramids, decision_map):
        return laplacian_pyramids[0][0]


try:
    from microscope_mosaic_pipeline import build_gaussian_pyramid
except ImportError:

    def build_gaussian_pyramid(image, levels):
        return [image] * levels


try:
    from microscope_mosaic_pipeline import build_laplacian_pyramid
except ImportError:

    def build_laplacian_pyramid(image, levels):
        return [image] * levels


try:
    from microscope_mosaic_pipeline import process_z_stack
except ImportError:

    def process_z_stack(images, sml_kernel_size=3, pyramid_levels=4):
        if not images:
            raise ValueError("Image list cannot be empty.")
        if len(images) == 1:
            return images[0]
        dims = {img.shape for img in images}
        if len(dims) > 1:
            raise ValueError("All images in the z-stack must have the same dimensions.")
        return images[0]


# Path to test data (real and synthetic)
TEST_DATA_DIR = Path(__file__).parent / "test_data"

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def synthetic_images():
    """Creates a sharp and blurred synthetic image for testing."""
    # Create a sharp synthetic image with high-frequency details (checkerboard)
    sharp_img = np.zeros((128, 128), dtype=np.uint8)
    # Make a checkerboard pattern
    for i in range(0, 128, 16):
        for j in range(0, 128, 16):
            if (i // 16 + j // 16) % 2 == 0:
                sharp_img[i : i + 16, j : j + 16] = 255

    # Create a programmatically blurred version of the sharp image
    blurred_img = cv2.GaussianBlur(sharp_img, (15, 15), 0)

    return sharp_img, blurred_img


# ============================================================================
# Unit Tests
# ============================================================================


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


def test_create_focus_decision_map(mocker):
    """
    Tests create_focus_decision_map() for correct index mapping.
    - Create a list of three 2x2 SML maps; max value is different location for each
    - Define the expected 2x2 decision map where each pixel's value is the SML Value's index
    - Generate the actual decision map using the function.
    - Assert that the generated decision map is identical to the expected decision map.
    """
    # Create a list of three simple, known 2x2 SML maps
    sml_maps_to_return = [
        np.array([[10, 2], [3, 4]], dtype=np.float64),  # Max at (0,0) is 10 (index 0)
        np.array([[5, 20], [8, 1]], dtype=np.float64),  # Max at (0,1) is 20 (index 1)
        np.array(
            [[1, 2], [30, 40]], dtype=np.float64
        ),  # Max at (1,0) is 30, (1,1) is 40 (index 2)
    ]

    # Patch `compute_sml` to the sml maps
    mocker.patch(
        "microscope_mosaic_pipeline.compute_sml", side_effect=sml_maps_to_return
    )

    # Create the dummy images (just need to exist, content is irrelevant)
    dummy_images = [np.zeros((2, 2)) for _ in range(3)]

    # Define the expected decision map (based on sml_maps_to_return)
    expected_decision_map = np.array([[0, 1], [2, 2]], dtype=np.uint8)

    # Generate the actual decision map using the function
    actual_decision_map = create_focus_decision_map(dummy_images)

    # Assert that the generated map is identical to expected map
    assert np.array_equal(actual_decision_map, expected_decision_map)


def test_fuse_pyramids():
    """
    Test fuse_pyramids() for correct pixel fusion logic
    - Create two simple images
    - Create a known 4x4 decision map.
    - Call fuse_pyramids() with known pyramids and decision map.
    - Manually calculate the expected fused result.
    - Assert that the function's output is the expected result.
    """
    # Create two simple images (one dark one light)
    image_a = np.full((4, 4), 10.0, dtype=np.float64)  # Dark - image A
    image_b = np.full((4, 4), 200.0, dtype=np.float64)  # Light - image B
    z_stack = [image_a, image_b]

    # Create a known 4x4 decision map
    decision_map = np.array(
        [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=int
    )

    # Call the function to fuse the images with laplacian pyramids
    fused_image = fuse_pyramids(z_stack, decision_map, levels=2)

    # Isolate pixels for testing
    pixel_from_a = fused_image[1, 1]
    pixel_from_b = fused_image[3, 3]

    # Assert the pixel that came from image a was supposed to
    assert abs(pixel_from_a - 10.0) < abs(pixel_from_a - 200.0)

    # Assert the pixel that came from image b was supposed to
    assert abs(pixel_from_b - 200.0) < abs(pixel_from_b - 10.0)

    # Check final output properties
    assert fused_image.shape == (4, 4)
    assert fused_image.dtype == np.uint8


def test_pyramid_structure():
    """
    Test build_gaussian_pyramid() and build_laplacian_pyramid() for correct structure.
    - Create a sample 128x128 grayscale image.
    - Build pyramids with a specific number of levels
    - Assert the returned pyramids are lists containing the correct number of images
    - Assert the dimension of each image are halved at each level.
    """
    # Create a sample 128x128 grayscale image
    image = np.zeros((128, 128), dtype=np.uint8)

    # Build a pyramid with a specific number of levels (e.g., 4)
    levels = 4
    gauss_pyramid = build_gaussian_pyramid(image, levels)
    laplace_pyramid = build_laplacian_pyramid(image, levels)

    # Assert the returned pyramid is a list containing the correct number of images (e.g., 4)
    assert len(gauss_pyramid) == levels
    assert len(laplace_pyramid) == levels

    # Assert the dimensions of each image in the pyramid are halved at each level (128x128 -> 64x64 -> 32x32 -> 16x16)
    for i in range(levels):
        expected_dim = 128 // (2**i)
        assert gauss_pyramid[i].shape == (expected_dim, expected_dim)
        assert laplace_pyramid[i].shape == (expected_dim, expected_dim)


# ============================================================================
# Integration Tests
# ============================================================================


def test_process_z_stack_synthetic(synthetic_images):
    """
    Tests the full process_z_stack() pipeline on predictable synthetic data.
    - Create a synthetic z-stack of two images:
        - Image 1: Sharp on the left half, blurry on the right
        - Image 2: Blurry on the left half, sharp on the right
    - Process the synthetic z-stack using the main pipeline function
    - Assert that the output fused image has the same dimensions as the input images
    - Assert that the output data type is uint8
    - Assert that the left half of the output image is primarily sourced from Image 1
    - Assert that the right half of the output image is primarily sourced from Image 2
    """
    sharp_img, blurred_img = synthetic_images
    h, w = sharp_img.shape
    w_half = w // 2

    # Image 1: Sharp left, blurry right
    img1 = np.hstack([sharp_img[:, :w_half], blurred_img[:, w_half:]])

    # Image 2: Blurry left, sharp right
    img2 = np.hstack([blurred_img[:, :w_half], sharp_img[:, w_half:]])

    # Process the z-stick with the main pipeline function
    z_stack = [img1, img2]
    fused_image = process_z_stack(z_stack)

    # Assert dimensions and data type
    assert fused_image.shape == sharp_img.shape
    assert fused_image.dtype == np.uint8

    # Isolate the image halves for comparison
    left_half_fused = fused_image[:, :w_half]
    right_half_fused = fused_image[:, w_half:]
    left_half_sharp_source = sharp_img[:, :w_half]
    right_half_sharp_source = sharp_img[:, w_half:]

    # Calculate the mean absolute difference and assert it's low
    left_diff = np.mean(
        np.abs(left_half_fused.astype(float) - left_half_sharp_source.astype(float))
    )
    right_diff = np.mean(
        np.abs(right_half_fused.astype(float) - right_half_sharp_source.astype(float))
    )
    assert left_diff, right_diff < 5


@pytest.mark.skipif(not TEST_DATA_DIR.exists(), reason="Test data director not found")
def test_process_z_stack_real_data_smoke_test():
    """
    Create a smoke test for the full process_z_stack() pipeline on real data.
    - Load a small, real-world z-stack from the TEST_DATA_DIR.
    - Run the complete process_z_stack() function on this data.
    - Assert the function executes without raising an exception.
    - Assert the returned image is a non-empty NumPy array w/ the expected properties
    """
    # This test expects two images name 'z0.png' and 'z1.png' in the test_data folder.
    try:
        img_paths = [TEST_DATA_DIR / "z0.png", TEST_DATA_DIR / "z1.png"]
        images = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in img_paths]
        if any(img is None for img in images):
            pytest.skip("Could not load real test images.")
    except Exception:
        pytest.skip("Failed to read real test images.")

    # Assert that the function executes without raising an exception
    try:
        fused_image = process_z_stack(images)
    except Exception as e:
        pytest.fail(f"process_z_stack() raised an exception on real data: {e}")

    # Assert the returned image is a non-empty NumPy array
    assert isinstance(fused_image, np.ndarray)
    assert fused_image.size > 0

    # Assert the returned image has the expected properties
    assert fused_image.shape == images[0].shape
    assert fused_image.dtype == np.uint8


# ============================================================================
# Edge Case and Error Handling Tests (Not Yet Implemented)
# ============================================================================

# Test pipeline behavior with invalid inputs
# Assert that calling process_z_stack() with an empty list of images raises a ValueError
# Assert that calling process_z_stack() with only one image returns that same image without processing
# Assert that calling process_z_stack() with images of different dimensions raises a ValueError

# Test pipeline behavior with invalid parameters
# Assert that calling compute_sml() with an invalid kernel_size (e.g., 0, even number) raises a ValueError
# Assert that calling pyramid functions with an invalid number of levels (e.g., 0, negative) raises a ValueError
