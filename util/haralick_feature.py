#Haralick

import cv2
import mahotas
import numpy as np

def extract_haralick_features(image_path, mask_path=None):
    """
    Extracts Haralick texture features from a skin lesion image.

    Parameters:
    - image_path: str, path to image
    - mask_path: str or None, binary mask to focus on lesion only (optional)

    Returns:
    - haralick_mean: numpy array, mean of 13 Haralick features over all directions
    """
    # Load image and convert to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gray = cv2.bitwise_and(gray, gray, mask=mask)

    # Compute Haralick features (returns a 4x13 matrix: 4 directions, 13 features)
    haralick_features = mahotas.features.haralick(gray)
    haralick_mean = haralick_features.mean(axis=0)

    return haralick_mean

# Example usage
features = extract_haralick_features("lesion.jpg", "mask.png")
print("Haralick Features (mean):", features)
