import os
import cv2
import numpy as np
import pandas as pd
from typing import Dict, Optional
from tqdm import tqdm

def extract_border_features_from_folder(
    folder_path: str,
    output_csv: Optional[str] = None,
    visualize: bool = False,
    block_size: int = 11,
    morph_kernel_size: int = 3
) -> pd.DataFrame:
    """
    Extracts a minimal set of informative border features from all images in a folder.

    This function iterates through images, calls extract_border_features for each,
    and compiles the results into a pandas DataFrame.

    Args:
        folder_path: Path to the folder containing lesion images.
        output_csv: Optional path to save the resulting features to a CSV file.
        visualize: If True, displays intermediate processing steps for debugging.
        block_size: The size of the neighborhood for adaptive thresholding (must be odd).
        morph_kernel_size: The size of the kernel for morphological operations.

    Returns:
        A pandas DataFrame where each row corresponds to an image and columns
        are the extracted border features.
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(valid_extensions)]

    features_list = []

    for filename in tqdm(image_files, desc="Extracting Border Features"):
        try:
            image_path = os.path.join(folder_path, filename)
            features = extract_border_features(
                image_path=image_path,
                visualize=visualize,
                block_size=block_size,
                morph_kernel_size=morph_kernel_size
            )
            base_filename = os.path.splitext(os.path.basename(filename))[0]
            features['filename'] = base_filename# Use basename for safety
            features_list.append(features)

        except Exception as e:
            print(f"\nError processing {filename}: {str(e)}")
            continue

    if not features_list:
        return pd.DataFrame()

    df = pd.DataFrame(features_list)

    # Reorder columns to have 'filename' first
    cols = ['filename'] + [col for col in df.columns if col != 'filename']
    df = df[cols]

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\nSaved border features to {output_csv}")

    return df

def extract_border_features(
    image_path: str,
    visualize: bool = False,
    block_size: int = 11,
    morph_kernel_size: int = 3
) -> Dict[str, float]:
    """
    Extracts a refined set of border features to minimize redundancy.

    This version focuses on the variability (standard deviation) of contour
    and edge properties, which are good indicators of irregularity.

    Features Extracted:
    - contour_count: Number of distinct contours detected.
    - contour_perimeter_std: Standard deviation of contour perimeters (measures size variation).
    - sobel_std: Standard deviation of Sobel edge magnitude (measures edge texture/consistency).
    - laplacian_std: Standard deviation of Laplacian response (measures edge "spikiness" variation).

    Args:
        (Same as above)

    Returns:
        A dictionary containing the calculated feature values for one image.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found or corrupted: {image_path}")

        # Standardize image size
        img = cv2.resize(img, (256, 256))

        # --- Edge Detection on Original Grayscale Image ---
        sobel_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        sobel_mag = np.hypot(sobel_x, sobel_y)
        laplacian = cv2.Laplacian(img, cv2.CV_32F)

        # --- Contour Analysis ---
        # We find contours on a binarized version of the edge magnitude map
        # to capture the main border of the lesion.
        _, binary_edges = cv2.threshold(
            cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            50, 255, cv2.THRESH_BINARY
        )
        contours, _ = cv2.findContours(
            binary_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        contour_count = len(contours)
        perimeters = [cv2.arcLength(cnt, True) for cnt in contours] if contours else []

        # --- Visualization ---
        if visualize:
            viz_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if contours:
                cv2.drawContours(viz_img, contours, -1, (0, 255, 0), 1)
            cv2.imshow("Extracted Contours", viz_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # --- Feature Engineering (Minimal Set) ---
        features = {
            "contour_count": float(contour_count),
            "contour_perimeter_std": np.std(perimeters) if perimeters else 0.0,
            "sobel_std": np.std(sobel_mag),
            "laplacian_std": np.std(laplacian)
        }

        return features

    except Exception as e:
        print(f"Error during feature extraction for {image_path}: {str(e)}")
        # Return a dictionary with default values to prevent pipeline failure
        return {
            "contour_count": 0.0,
            "contour_perimeter_std": 0.0,
            "sobel_std": 0.0,
            "laplacian_std": 0.0
        }

if __name__ == "__main__":
    # IMPORTANT: This should ideally point to your pre-processed, matched MASK directory
    # as masks provide the cleanest definition of a lesion's border.
    mask_folder_path = r''
    output_csv_path = os.path.join(mask_folder_path, "border_features.csv") 

    print(f"Starting border feature extraction from: {mask_folder_path}")

    # The function now directly returns the final DataFrame of features.
    df_features = extract_border_features_from_folder(
        folder_path=mask_folder_path,
        output_csv=output_csv_path,
        visualize=False
    )

    if not df_features.empty:
        print("\nSuccessfully extracted features. First 5 rows:")
        print(df_features.head())
        print(f"\nTotal features extracted for {len(df_features)} images.")
    else:
        print("\nNo features were extracted. Please check the input folder and image files.")