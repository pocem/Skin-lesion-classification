

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage import segmentation, color
from sklearn.cluster import KMeans
from tqdm import tqdm # Import tqdm

def extract_feature_C(folder_path, output_csv=None, normalize_colors=True, visualize=False):
    """
    Function to extract color features from skin lesion images in a folder
    
    Parameters:
    folder_path (str): Path to the folder containing skin lesion images
    output_csv (str): Path to output CSV file. If the file exists, features will be added to it
    normalize_colors (bool): Whether to normalize color values to range [0,1]
    visualize (bool): Whether to visualize the segmentation results
    
    Returns:
    pd.DataFrame: DataFrame containing color features for all images
    """
    # List to store results
    results = []
    
    # Supported image extensions
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Load existing CSV if specified and exists
    existing_df = None
    if output_csv and os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            print(f"Loaded existing features from {output_csv}")
        except Exception as e:
            print(f"Error loading existing CSV: {str(e)}")
    
    # Get list of image files
    image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_extensions]

    # Iterate through all files in the folder with a progress bar
    for filename in tqdm(image_files, desc="Extracting Color Features (C)"): # Wrap loop with tqdm
        # Skip if the image is already in the existing dataframe
        if existing_df is not None and filename in existing_df['filename'].values:
            # print(f"Skipping {filename} - already processed") # Suppress per-file skip message
            continue
            
        image_path = os.path.join(folder_path, filename)
        # print(f"Processing {filename}...") # Suppress per-file message
        
        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error reading {filename}, skipping...")
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize for consistency (optional)
            img = cv2.resize(img, (256, 256))
            
            # Step 1: Segment the lesion from the background
            # Convert to LAB color space for better segmentation
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            
            # Apply SLIC segmentation to get superpixels
            segments = segmentation.slic(img, n_segments=100, compactness=10, sigma=1)
            
            # Create a mask for the lesion area
            h, w = img.shape[:2]
            center_y, center_x = h // 2, w // 2
            
            # Create a circular mask around the center
            y, x = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            mask = dist_from_center <= min(h, w) // 3
            
            # Refine mask using color information
            pixels = img.reshape(-1, 3)
            kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(pixels) # Added n_init='auto' for KMeans
            labels = kmeans.labels_.reshape(h, w)
            
            # Determine which label corresponds to the lesion
            center_label = labels[center_y, center_x]
            refined_mask = (labels == center_label)
            
            # Combine masks
            final_mask = np.logical_and(mask, refined_mask)
            
            # Step 2: Extract color features from the lesion area
            lesion_pixels = img[final_mask]
            
            if len(lesion_pixels) == 0:
                print(f"No lesion detected in {filename}, skipping...")
                # Initialize features with default values for skipped files
                features = {'filename': filename}
                default_color_features = [
                    'c_mean_red', 'c_mean_green', 'c_mean_blue', 'c_std_red', 'c_std_green', 'c_std_blue',
                    'c_mean_hue', 'c_mean_saturation', 'c_mean_value', 'c_std_hue', 'c_std_saturation', 'c_std_value',
                    'c_red_asymmetry', 'c_green_asymmetry', 'c_blue_asymmetry', 'c_color_variance',
                    'c_red_green_ratio', 'c_red_blue_ratio', 'c_green_blue_ratio'
                ]
                for f in default_color_features:
                    features[f] = 0.0
                features['c_dominant_channel'] = 'none' # Or NaN, depending on preference
                results.append(features)
                continue
            
            # Calculate color features
            features = {'filename': filename}
            
            # Apply normalization if requested
            if normalize_colors:
                lesion_pixels = lesion_pixels / 255.0
                divisor = 1.0  # For normalized ratios
            else:
                divisor = 1.0  # For non-normalized values
            
            # RGB color space features
            features['c_mean_red'] = np.mean(lesion_pixels[:, 0])
            features['c_mean_green'] = np.mean(lesion_pixels[:, 1])
            features['c_mean_blue'] = np.mean(lesion_pixels[:, 2])
            features['c_std_red'] = np.std(lesion_pixels[:, 0])
            features['c_std_green'] = np.std(lesion_pixels[:, 1])
            features['c_std_blue'] = np.std(lesion_pixels[:, 2])
            
            # Convert to HSV for additional features
            # If already normalized [0,1], no need to divide by 255
            if normalize_colors:
                hsv_pixels = color.rgb2hsv(lesion_pixels)
            else:
                hsv_pixels = color.rgb2hsv(lesion_pixels / 255.0)
                
            features['c_mean_hue'] = np.mean(hsv_pixels[:, 0])
            features['c_mean_saturation'] = np.mean(hsv_pixels[:, 1])
            features['c_mean_value'] = np.mean(hsv_pixels[:, 2])
            features['c_std_hue'] = np.std(hsv_pixels[:, 0])
            features['c_std_saturation'] = np.std(hsv_pixels[:, 1])
            features['c_std_value'] = np.std(hsv_pixels[:, 2])
            
            # Color asymmetry features
            left_mask = np.zeros_like(final_mask)
            left_mask[:, :w//2] = final_mask[:, :w//2]
            right_mask = np.zeros_like(final_mask)
            right_mask[:, w//2:] = final_mask[:, w//2:]
            
            left_pixels = img[left_mask]
            right_pixels = img[right_mask]
            
            if len(left_pixels) > 0 and len(right_pixels) > 0:
                if normalize_colors:
                    left_pixels = left_pixels / 255.0
                    right_pixels = right_pixels / 255.0
                    
                features['c_red_asymmetry'] = abs(np.mean(left_pixels[:, 0]) - np.mean(right_pixels[:, 0]))
                features['c_green_asymmetry'] = abs(np.mean(left_pixels[:, 1]) - np.mean(right_pixels[:, 1]))
                features['c_blue_asymmetry'] = abs(np.mean(left_pixels[:, 2]) - np.mean(right_pixels[:, 2]))
            else:
                features['c_red_asymmetry'] = 0
                features['c_green_asymmetry'] = 0
                features['c_blue_asymmetry'] = 0
            
            # Color variance (indicates color homogeneity/heterogeneity)
            features['c_color_variance'] = np.sum(np.var(lesion_pixels, axis=0))
            
            # Additional color features
            # Color ratio features (avoid division by zero)
            features['c_red_green_ratio'] = features['c_mean_red'] / max(features['c_mean_green'], divisor)
            features['c_red_blue_ratio'] = features['c_mean_red'] / max(features['c_mean_blue'], divisor)
            features['c_green_blue_ratio'] = features['c_mean_green'] / max(features['c_mean_blue'], divisor)
            
            # Color dominance
            rgb_means = [features['c_mean_red'], features['c_mean_green'], features['c_mean_blue']]
            features['c_dominant_channel'] = ['red', 'green', 'blue'][np.argmax(rgb_means)]
            
            # Visualization (if enabled)
            if visualize:
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(img)
                plt.title("Original Image")
                
                plt.subplot(1, 3, 2)
                plt.imshow(final_mask, cmap='gray')
                plt.title("Lesion Mask")
                
                plt.subplot(1, 3, 3)
                masked_img = img.copy()
                masked_img[~final_mask] = [0, 0, 0]
                plt.imshow(masked_img)
                plt.title("Extracted Lesion")
                
                plt.tight_layout()
                
                # Save visualization to a subdirectory
                vis_dir = os.path.join(folder_path, 'visualizations')
                os.makedirs(vis_dir, exist_ok=True)
                plt.savefig(os.path.join(vis_dir, f"vis_{filename}"))
                plt.close()
            
            results.append(features)
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            # Ensure skipped files still have a row in the results DataFrame,
            # even if features are defaulted to 0 or NaN.
            # Initialize features with default values for skipped files
            features = {'filename': filename}
            default_color_features = [
                'c_mean_red', 'c_mean_green', 'c_mean_blue', 'c_std_red', 'c_std_green', 'c_std_blue',
                'c_mean_hue', 'c_mean_saturation', 'c_mean_value', 'c_std_hue', 'c_std_saturation', 'c_std_value',
                'c_red_asymmetry', 'c_green_asymmetry', 'c_blue_asymmetry', 'c_color_variance',
                'c_red_green_ratio', 'c_red_blue_ratio', 'c_green_blue_ratio'
            ]
            for f in default_color_features:
                features[f] = 0.0
            features['c_dominant_channel'] = 'none' # Or NaN, depending on preference
            results.append(features)

    # Convert to DataFrame
    new_df = pd.DataFrame(results)
    
    # Combine with existing data if available
    if existing_df is not None and not new_df.empty:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    elif not new_df.empty:
        combined_df = new_df
    else:
        combined_df = existing_df if existing_df is not None else pd.DataFrame()
    

    
    return combined_df

# Example usage with all features together in a single CSV:
"""
# First extraction (e.g., color features)
df = extract_feature_C('path_to_images', output_csv='all_features.csv', normalize_colors=True)

# Second extraction (from another function - texture features)
df = extract_feature_T('path_to_images', output_csv='all_features.csv')

# Third extraction (e.g., shape features)
df = extract_feature_S('path_to_images', output_csv='all_features.csv')
"""

if __name__ == "__main__":
    image_folder = r"C:\Users\Erik\OneDrive - ITU\Escritorio\2 semester\Semester project\Introduction to final project\matched_pairs\images" # Example path
    # This output_csv variable is for the standalone run of feature_C.py
    output_csv_for_standalone_run = r"C:\Users\Erik\OneDrive - ITU\Escritorio\2 semester\Semester project\Introduction to final project\2025-FYP-Final\result\color_features_standalone.csv"

    df = extract_feature_C(
        folder_path=image_folder,
        output_csv=None,  # Pass None here if you want to handle saving manually after getting the df
        normalize_colors=True,
        visualize=False # Set to False for faster processing unless debugging
    )

    # Save to CSV separately if df is not empty
    if not df.empty:
        # Ensure the directory for output_csv_for_standalone_run exists
        os.makedirs(os.path.dirname(output_csv_for_standalone_run), exist_ok=True)
        df.to_csv(output_csv_for_standalone_run, index=False) # Use the defined path
        print(f"Saved extracted color features (standalone run) to: {output_csv_for_standalone_run}")
        print(df.head())
    else:
        print("No color features were extracted (standalone run).")