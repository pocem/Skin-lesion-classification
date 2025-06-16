import cv2
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
from scipy.ndimage import distance_transform_edt
from skimage import measure, transform, morphology
from tqdm import tqdm

# This function no longer needs to do its own segmentation.
# It will receive a path to a folder of masks and work on them directly.
def extract_asymmetry_features(folder_path, output_csv=None, visualize=False, **kwargs):
    """
    Extracts asymmetry features directly from lesion MASK images in a folder.

    Parameters:
    - folder_path (str): Path to the folder containing pre-segmented binary mask images.
    - output_csv (str, optional): Path to save the output CSV file.
    - visualize (bool, optional): Whether to show visualizations.
    - **kwargs: Accepts extra keyword arguments (like 'mask_folder_path') and ignores them
                to prevent crashes when called from different pipelines.
    Returns:
    - pd.DataFrame: DataFrame containing asymmetry features for all mask images.
    """
    results = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_extensions]

    for filename in tqdm(image_files, desc="Extracting Asymmetry Features from Masks"):
        image_path = os.path.join(folder_path, filename)
        base_filename = os.path.splitext(filename)[0].replace('_mask', '')
        features = {'filename': base_filename}

        try:
            # Read the mask image in grayscale
            mask_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                print(f"Warning: Could not read mask {filename}, skipping.")
                continue

            # Binarize the mask: ensure it's 0s and 1s.
            # Anything > 0 becomes 1.
            _, binary_mask = cv2.threshold(mask_img, 1, 1, cv2.THRESH_BINARY)
            binary_mask = binary_mask.astype(np.uint8)

            # --- REMOVED: All the complex color segmentation logic ---
            # We now use the provided binary_mask directly.

            # Calculate all asymmetry features
            basic_score = compute_basic_asymmetry(binary_mask)
            features['a_basic'] = basic_score

            pca_score = compute_pca_asymmetry(binary_mask)
            features['a_pca'] = pca_score

            boundary_score = compute_boundary_asymmetry(binary_mask)
            features['a_boundary'] = boundary_score

            # We can remove the combined score, as the model can learn the weights itself.
            # Keeping the fundamental features is better.
            # combined_score = 0.4*basic_score + 0.3*pca_score + 0.3*boundary_score
            # features['a_combined'] = min(combined_score, 1.0)

            results.append(features)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            base_filename = os.path.splitext(filename)[0].replace('_mask', '')
            features = {'filename': base_filename, 'a_basic': np.nan, 'a_pca': np.nan, 'a_boundary': np.nan}
            results.append(features)

    df = pd.DataFrame(results)

    if output_csv and not df.empty:
        df.to_csv(output_csv, index=False)
        print(f"Asymmetry features saved to {output_csv}")

    return df

# Helper functions remain the same but are now more reliable
# because they receive clean binary masks.

def compute_basic_asymmetry(mask):
    """Compute basic vertical/horizontal mirror asymmetry"""
    labeled = measure.label(mask)
    regions = measure.regionprops(labeled)

    if not regions:
        return 1.0  # Max asymmetry if no lesion found

    lesion = max(regions, key=lambda r: r.area)
    minr, minc, maxr, maxc = lesion.bbox
    cropped = mask[minr:maxr, minc:maxc]

    if cropped.size == 0:
        return 1.0

    vert_flip = np.fliplr(cropped)
    horiz_flip = np.flipud(cropped)

    vert_diff = np.sum(np.abs(cropped - vert_flip))
    horiz_diff = np.sum(np.abs(cropped - horiz_flip))

    total_area = lesion.area
    if total_area == 0:
        return 1.0

    return (vert_diff + horiz_diff) / (2 * total_area)

def compute_pca_asymmetry(mask):
    """Compute rotation-invariant asymmetry using PCA alignment"""
    y, x = np.where(mask)
    if len(x) < 2:
        return 1.0

    try:
        pca = PCA(n_components=2)
        coords = np.column_stack([x, y])
        pca.fit(coords - np.mean(coords, axis=0))
        angle = np.arctan2(pca.components_[0][1], pca.components_[0][0]) * 180 / np.pi

        # Rotate the original mask (not a cropped version)
        rotated = transform.rotate(mask.astype(float), angle, resize=True, order=0, preserve_range=True).astype(np.uint8)

        if np.sum(rotated) == 0:
            return 1.0

        # Now flip and compare
        return np.sum(np.abs(rotated - np.fliplr(rotated))) / (2 * np.sum(rotated))
    except Exception:
        # Fallback to a simpler method if PCA fails
        return compute_basic_asymmetry(mask)

def compute_boundary_asymmetry(mask):
    """Compute boundary-weighted asymmetry"""
    if np.sum(mask) == 0:
        return 1.0

    boundary = mask - morphology.binary_erosion(mask)
    if np.sum(boundary) == 0: # Handle single-pixel masks
        return 1.0
        
    weights = distance_transform_edt(~boundary)
    vert_flip = np.fliplr(mask)
    vert_diff = np.sum(weights * np.abs(mask - vert_flip))
    
    return vert_diff / max(np.sum(weights), 1)

if __name__ == "__main__":
    # This should point to your folder of MASKS
    mask_folder_path = r"C:\Users\misog\portfolio\Machine learning skin lesion project\matched_data\masks"
    output_csv_path = r"./results_refined_model/asymmetry_features.csv"

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # Call the function correctly now
    df = extract_asymmetry_features(folder_path=mask_folder_path, output_csv=output_csv_path)

    if not df.empty:
        print("\nAsymmetry features extracted successfully:")
        print(df.head())
    else:
        print("\nNo asymmetry features were extracted.")