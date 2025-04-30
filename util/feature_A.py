#Feature A
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
import numpy as np

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
