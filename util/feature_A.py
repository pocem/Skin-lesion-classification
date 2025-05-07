#Feature A
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.transform import rotate
from sklearn.decomposition import PCA
import numpy as np
import os

def compute_asymmetry_from_rgb(img_rgb):
    # Convert to grayscale
    gray = rgb2gray(img_rgb)

    # Threshold the image to create a binary mask of the lesion
    thresh = threshold_otsu(gray)
    binary = gray < thresh  # dark regions are lesions

    # Label connected regions
    labeled = label(binary)
    regions = regionprops(labeled)

    if not regions:
        return 1.0  # Fully asymmetric if no lesion found

    # Focus on the largest region
    lesion = max(regions, key=lambda r: r.area)
    minr, minc, maxr, maxc = lesion.bbox
    cropped = binary[minr:maxr, minc:maxc]

    # Mirror comparisons
    vertical_flip = np.fliplr(cropped)
    horizontal_flip = np.flipud(cropped)

    vert_diff = np.sum(np.abs(cropped.astype(int) - vertical_flip.astype(int)))
    horiz_diff = np.sum(np.abs(cropped.astype(int) - horizontal_flip.astype(int)))

    total_area = cropped.size
    asymmetry_score = (vert_diff + horiz_diff) / (2 * total_area)
    return asymmetry_score

def compute_asymmetry_from_mask(mask):
    """
    Compute asymmetry score for a pre-masked binary image (lesion = 1/True, background = 0/False).
    """
    # Ensure input is a binary mask (0s and 1s)
    if mask.dtype != bool:
        mask = mask.astype(bool)
    
    # Label connected regions
    labeled = label(mask)
    regions = regionprops(labeled)
    
    if not regions:
        return 1.0  # No lesion found
    
    # Focus on the largest region (assumed to be the lesion)
    lesion = max(regions, key=lambda r: r.area)
    minr, minc, maxr, maxc = lesion.bbox
    cropped = mask[minr:maxr, minc:maxc]
    
    # Calculate asymmetry via flipping
    vertical_flip = np.fliplr(cropped)
    horizontal_flip = np.flipud(cropped)
    
    vert_diff = np.sum(np.abs(cropped.astype(int) - vertical_flip.astype(int)))
    horiz_diff = np.sum(np.abs(cropped.astype(int) - horizontal_flip.astype(int)))
    
    total_area = lesion.area
    if total_area == 0:
        return 1.0  # Avoid division by zero
    
    asymmetry_score = (vert_diff + horiz_diff) / (2 * total_area)
    return asymmetry_score

def pca_align(mask):
    # Get lesion coordinates and center
    y, x = np.where(mask)
    center = np.array([x.mean(), y.mean()])
    
    # PCA to find principal axis
    pca = PCA(n_components=2)
    pca.fit(np.column_stack([x, y]))
    angle = np.arctan2(pca.components_[0][1], pca.components_[0][0]) * 180 / np.pi
    
    # Rotate to align principal axis horizontally
    rotated = rotate(mask.astype(float), angle, resize=True, order=0)
    return rotated
