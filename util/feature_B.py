import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure
from typing import Dict, Optional

def extract_border_features(
    image_path: str,
    visualize: bool = False,
    block_size: int = 11,
    morph_kernel_size: int = 3,
    hog_orientations: int = 9,
    hog_pixels_per_cell: tuple = (16, 16),
    hog_cells_per_block: tuple = (2, 2)
) -> Dict[str, float]:
    """
    Enhanced border feature extraction with configurable parameters and robust error handling.

    Args:
        image_path: Path to the input image.
        visualize: Whether to display intermediate results.
        block_size: Adaptive thresholding block size (must be odd).
        morph_kernel_size: Size of morphological operation kernel.
        hog_orientations: Number of HOG orientation bins.
        hog_pixels_per_cell: HOG cell size (width, height).
        hog_cells_per_block: HOG block normalization size.

    Returns:
        Dictionary of border features.
    """
    # --- Image Loading with Validation ---
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found or corrupted: {image_path}")
    
    # --- Adaptive Thresholding ---
    if block_size % 2 == 0:
        block_size += 1  # Ensure odd block size
    img_adapt = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, 2
    )
    
    # --- Morphological Processing ---
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    img_clean = cv2.morphologyEx(img_adapt, cv2.MORPH_CLOSE, kernel)

    # --- Edge Detection on Original Image ---
    sobel_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.hypot(sobel_x, sobel_y)
    
    laplacian = cv2.Laplacian(img, cv2.CV_32F)
    
    # --- Contour Analysis (All Contours) ---
    _, fused_edges = cv2.threshold(
        cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        50, 255, cv2.THRESH_BINARY
    )
    contours, _ = cv2.findContours(
        fused_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Handle empty contours
    contour_count = len(contours)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    perimeters = [cv2.arcLength(cnt, True) for cnt in contours]
    
    # --- HOG on Original Grayscale Image ---
    hog_features, hog_img = hog(
        img,
        orientations=hog_orientations,
        pixels_per_cell=hog_pixels_per_cell,
        cells_per_block=hog_cells_per_block,
        visualize=True,
        channel_axis=None
    )
    
    # --- Visualization ---
    if visualize:
        viz_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(viz_img, contours, -1, (0, 255, 0), 1)
        cv2.imshow("Contours", viz_img)
        cv2.imshow("HOG", exposure.rescale_intensity(hog_img, out_range=(0, 255)).astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # --- Feature Engineering ---
    features = {
        # Contour Features
        "contour_count": contour_count,
        "avg_contour_area": np.nanmean(areas) if areas else 0.0,
        "contour_area_std": np.nanstd(areas) if areas else 0.0,
        # Edge Features
        "sobel_mean": np.nanmean(sobel_mag),
        "sobel_std": np.nanstd(sobel_mag),
        "laplacian_mean": np.nanmean(laplacian),
        # HOG Features (first 5 components as example)
        **{f"hog_{i}": val for i, val in enumerate(hog_features[:5])}
    }
    
    return features
