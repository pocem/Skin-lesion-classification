import os
import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from tqdm import tqdm  # for progress bar

def extract_border_features_from_folder(
    folder_path: str,
    output_csv: Optional[str] = None,
    visualize: bool = False,
    block_size: int = 11,
    morph_kernel_size: int = 3
) -> pd.DataFrame:
    """
    Extract border features from all images in a folder and return as DataFrame.
    
    Args:
        folder_path: Path to folder containing images
        output_csv: Optional path to save results as CSV
        visualize: Whether to display intermediate results
        block_size: Adaptive thresholding block size (must be odd)
        morph_kernel_size: Size of morphological operation kernel
        
    Returns:
        DataFrame containing border features for all images
    """
    # Supported image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(folder_path) 
                  if f.lower().endswith(valid_extensions)]
    
    features_list = []
    
    for filename in tqdm(image_files, desc="Processing images"):
        try:
            image_path = os.path.join(folder_path, filename)
            features = extract_border_features(
                image_path=image_path,
                visualize=visualize,
                block_size=block_size,
                morph_kernel_size=morph_kernel_size
            )
            features['filename'] = filename
            features_list.append(features)
            
        except Exception as e:
            print(f"\nError processing {filename}: {str(e)}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(features_list)
    
    # Reorder columns to have filename first
    cols = ['filename'] + [col for col in df.columns if col != 'filename']
    df = df[cols]
    
    # Save to CSV if requested
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\nSaved features to {output_csv}")
    
    return df

def extract_border_features(
    image_path: str,
    visualize: bool = False,
    block_size: int = 11,
    morph_kernel_size: int = 3
) -> Dict[str, float]:
    """
    Enhanced border feature extraction focused on essential border characteristics.
    """
    try:
        # --- Image Loading with Validation ---
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found or corrupted: {image_path}")
        
        # Resize for consistency (optional)
        img = cv2.resize(img, (256, 256))
        
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
        
        # --- Visualization ---
        if visualize:
            viz_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(viz_img, contours, -1, (0, 255, 0), 1)
            cv2.imshow("Contours", viz_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # --- Feature Engineering ---
        features = {
            # Contour Features
            "contour_count": contour_count,
            "avg_contour_area": np.nanmean(areas) if areas else 0.0,
            "contour_area_std": np.nanstd(areas) if areas else 0.0,
            "avg_contour_perimeter": np.nanmean(perimeters) if perimeters else 0.0,
            "contour_perimeter_std": np.nanstd(perimeters) if perimeters else 0.0,
            # Edge Features
            "sobel_mean": np.nanmean(sobel_mag),
            "sobel_std": np.nanstd(sobel_mag),
            "laplacian_mean": np.nanmean(laplacian),
            "laplacian_std": np.nanstd(laplacian)
        }
        
        return features
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        # Return empty features with same structure
        return {
            "contour_count": 0,
            "avg_contour_area": 0.0,
            "contour_area_std": 0.0,
            "avg_contour_perimeter": 0.0,
            "contour_perimeter_std": 0.0,
            "sobel_mean": 0.0,
            "sobel_std": 0.0,
            "laplacian_mean": 0.0,
            "laplacian_std": 0.0
        }

def calculate_border_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a single border irregularity score from extracted features.
    Higher scores indicate more irregular borders (potentially malignant).
    """
    df_copy = df.copy()
    
    # Avoid division by zero
    df_copy['sobel_mean_safe'] = df_copy['sobel_mean'].replace(0, 1)
    df_copy['avg_contour_perimeter_safe'] = df_copy['avg_contour_perimeter'].replace(0, 1)
    df_copy['laplacian_mean_safe'] = df_copy['laplacian_mean'].abs().replace(0, 1)
    df_copy['avg_contour_area_safe'] = df_copy['avg_contour_area'].replace(0, 1)
    
    # Calculate individual irregularity components
    perimeter_irregularity = df_copy['contour_perimeter_std'] / df_copy['avg_contour_perimeter_safe']
    edge_irregularity = df_copy['sobel_std'] / df_copy['sobel_mean_safe']
    laplacian_irregularity = df_copy['laplacian_std'] / df_copy['laplacian_mean_safe']
    compactness = df_copy['avg_contour_perimeter'] / np.sqrt(df_copy['avg_contour_area_safe'])
    
    # Combine into border score (normalized)
    df_copy['border_score'] = (
        0.3 * perimeter_irregularity +
        0.3 * edge_irregularity + 
        0.2 * laplacian_irregularity +
        0.2 * (compactness / compactness.mean())  # Normalize compactness
    )
    
    return df_copy

# Example usage
if __name__ == "__main__":
    # Process all images in a folder
    folder_path = "masks"
    output_csv = "border_features.csv"
    
    df = extract_border_features_from_folder(
        folder_path=folder_path,
        output_csv=output_csv,
        visualize=False  # Set to True to see processing steps
    )
    
    print("\nExtracted features:")
    print(df.head())
    
    # Calculate border scores
    df_with_scores = calculate_border_score(df)
    
    print("\nBorder scores:")
    print(df_with_scores[['filename', 'border_score']].head())
    
    # Save with border scores
    df_with_scores.to_csv(output_csv.replace('.csv', '_with_scores.csv'), index=False)
    print(f"\nSaved features with border scores to {output_csv.replace('.csv', '_with_scores.csv')}")