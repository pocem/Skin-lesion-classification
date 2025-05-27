

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage import color # Using skimage.color for HSV conversion
from sklearn.cluster import KMeans
from tqdm import tqdm # Import tqdm

def extract_feature_BV(folder_path, output_csv=None, normalize_colors=True, visualize=False):
    """
    Function to extract blue veil features from skin lesion images in a folder.
    Blue veil is characterized by blue-to-whitish or blue-to-gray areas.
    
    Parameters:
    folder_path (str): Path to the folder containing skin lesion images.
    output_csv (str): Path to output CSV file. If the file exists, features for new images
                      will be added to it (assumes it's a CSV for blue veil features).
    normalize_colors (bool): Whether to normalize output RGB color mean/std values to range [0,1].
    visualize (bool): Whether to visualize the segmentation and blue veil detection results.
    
    Returns:
    pd.DataFrame: DataFrame containing blue veil features.
    """
    results = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    existing_df = None
    if output_csv and os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            print(f"Loaded existing Blue Veil features from {output_csv}")
        except Exception as e:
            print(f"Error loading existing CSV {output_csv}: {str(e)}")
            existing_df = None

    image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_extensions]

    for filename in tqdm(image_files, desc="Extracting Blue Veil Features"): # Wrap loop with tqdm
        if existing_df is not None and filename in existing_df['filename'].values:
            # print(f"Skipping {filename} - already processed for Blue Veil features in existing CSV.") # Suppress per-file skip message
            continue
            
        image_path = os.path.join(folder_path, filename)
        # print(f"Processing {filename} for Blue Veil features...") # Suppress per-file message
        
        current_features = {'filename': filename} # Initialize features for this file

        try:
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                print(f"Error reading {filename}, skipping...")
                continue # Skip this file
                
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
            
            h, w = img_resized.shape[:2]

            # Step 1: Segment the lesion (similar to extract_feature_C)
            center_y, center_x = h // 2, w // 2
            y_coords, x_coords = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            
            initial_mask_radius = min(h, w) // 2.8 # Adjusted radius
            circular_mask = dist_from_center <= initial_mask_radius
            
            pixels_for_kmeans = img_resized.reshape(-1, 3)
            kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(pixels_for_kmeans)
            labels = kmeans.labels_.reshape(h, w)
            
            center_label = labels[center_y, center_x]
            kmeans_mask = (labels == center_label)
            
            final_lesion_mask = np.logical_and(circular_mask, kmeans_mask)
            lesion_pixels_rgb = img_resized[final_lesion_mask]
            
            if len(lesion_pixels_rgb) == 0:
                print(f"No lesion detected in {filename} after segmentation, skipping BV feature extraction.")
                # Populate with default values for this file as it's skipped for BV calculation
                current_features['bv_present'] = 0
                current_features['bv_pixel_count'] = 0
                current_features['bv_area_ratio'] = 0.0
                current_features['bv_mean_R'] = 0.0
                current_features['bv_mean_G'] = 0.0
                current_features['bv_mean_B'] = 0.0
                current_features['bv_std_R'] = 0.0
                current_features['bv_std_G'] = 0.0
                current_features['bv_std_B'] = 0.0
                current_features['bv_mean_H'] = 0.0
                current_features['bv_mean_S'] = 0.0
                current_features['bv_mean_V'] = 0.0
                results.append(current_features) # Append features (with defaults) for this file

                if visualize: # Create a minimal visualization if requested for no lesion case
                    plt.figure(figsize=(12, 4)) # Adjusted for 3 plots
                    plt.subplot(1, 3, 1); plt.imshow(img_resized); plt.title("Resized Image"); plt.axis('off')
                    plt.subplot(1, 3, 2); plt.imshow(final_lesion_mask, cmap='gray'); plt.title("Lesion Mask (Empty)"); plt.axis('off')
                    lesion_display_img = np.zeros_like(img_resized) # Black image
                    plt.subplot(1, 3, 3); plt.imshow(lesion_display_img); plt.title("Extracted Lesion (Empty)"); plt.axis('off')
                    plt.suptitle(f"{filename} - No Lesion Detected")
                    vis_dir = os.path.join(folder_path, 'visualizations_bv')
                    os.makedirs(vis_dir, exist_ok=True)
                    plt.savefig(os.path.join(vis_dir, f"vis_bv_{filename}_no_lesion.png"))
                    plt.close()
                continue # Skip to next file for feature extraction

            # Step 2: Blue Veil Detection
            lesion_pixels_rgb_normalized = lesion_pixels_rgb / 255.0
            lesion_pixels_hsv = color.rgb2hsv(lesion_pixels_rgb_normalized)

            # HSV thresholds for blue-veil (tune as needed)
            # H: [0,1] from skimage.color.rgb2hsv (0-360 degrees)
            # Blue-ish range: ~180-270 degrees (0.5 - 0.75 in [0,1] scale)
            H_MIN_BV = 0.52  # Approx 187 deg (cyan-blue)
            H_MAX_BV = 0.75  # Approx 270 deg (blue-magenta)
            S_MIN_BV = 0.10  # Min saturation (allow desaturated/whitish/grayish blues)
            S_MAX_BV = 0.75  # Max saturation (avoid extremely vibrant pure blues if not veil-like)
            V_MIN_BV = 0.30  # Min value/brightness (avoid very dark pixels)
            V_MAX_BV = 0.95  # Max value/brightness (avoid pure white)

            bv_pixels_mask_1d = (lesion_pixels_hsv[:, 0] >= H_MIN_BV) & \
                                (lesion_pixels_hsv[:, 0] <= H_MAX_BV) & \
                                (lesion_pixels_hsv[:, 1] >= S_MIN_BV) & \
                                (lesion_pixels_hsv[:, 1] <= S_MAX_BV) & \
                                (lesion_pixels_hsv[:, 2] >= V_MIN_BV) & \
                                (lesion_pixels_hsv[:, 2] <= V_MAX_BV)
            
            detected_bv_rgb_pixels = lesion_pixels_rgb[bv_pixels_mask_1d]
            
            if len(detected_bv_rgb_pixels) > 0:
                current_features['bv_present'] = 1
                current_features['bv_pixel_count'] = len(detected_bv_rgb_pixels)
                current_features['bv_area_ratio'] = len(detected_bv_rgb_pixels) / float(len(lesion_pixels_rgb))
                
                mean_rgb_bv = np.mean(detected_bv_rgb_pixels, axis=0)
                std_rgb_bv = np.std(detected_bv_rgb_pixels, axis=0)

                if normalize_colors:
                    current_features['bv_mean_R'] = mean_rgb_bv[0] / 255.0
                    current_features['bv_mean_G'] = mean_rgb_bv[1] / 255.0
                    current_features['bv_mean_B'] = mean_rgb_bv[2] / 255.0
                    current_features['bv_std_R'] = std_rgb_bv[0] / 255.0
                    current_features['bv_std_G'] = std_rgb_bv[1] / 255.0
                    current_features['bv_std_B'] = std_rgb_bv[2] / 255.0
                else:
                    current_features['bv_mean_R'] = mean_rgb_bv[0]
                    current_features['bv_mean_G'] = mean_rgb_bv[1]
                    current_features['bv_mean_B'] = mean_rgb_bv[2]
                    current_features['bv_std_R'] = std_rgb_bv[0]
                    current_features['bv_std_G'] = std_rgb_bv[1]
                    current_features['bv_std_B'] = std_rgb_bv[2]

                detected_bv_hsv_pixels_subset = lesion_pixels_hsv[bv_pixels_mask_1d]
                mean_hsv_bv = np.mean(detected_bv_hsv_pixels_subset, axis=0)
                current_features['bv_mean_H'] = mean_hsv_bv[0]
                current_features['bv_mean_S'] = mean_hsv_bv[1]
                current_features['bv_mean_V'] = mean_hsv_bv[2]
            else: # No blue veil detected in the lesion
                current_features['bv_present'] = 0
                current_features['bv_pixel_count'] = 0
                current_features['bv_area_ratio'] = 0.0
                current_features['bv_mean_R'] = 0.0
                current_features['bv_mean_G'] = 0.0
                current_features['bv_mean_B'] = 0.0
                current_features['bv_std_R'] = 0.0
                current_features['bv_std_G'] = 0.0
                current_features['bv_std_B'] = 0.0
                current_features['bv_mean_H'] = 0.0
                current_features['bv_mean_S'] = 0.0
                current_features['bv_mean_V'] = 0.0
            
            results.append(current_features) # Add features for this successfully processed file

            if visualize:
                plt.figure(figsize=(16, 4)) 
                
                plt.subplot(1, 4, 1); plt.imshow(img_resized); plt.title("Resized Image"); plt.axis('off')
                plt.subplot(1, 4, 2); plt.imshow(final_lesion_mask, cmap='gray'); plt.title("Lesion Mask"); plt.axis('off')
                
                lesion_display_img = np.zeros_like(img_resized)
                lesion_display_img[final_lesion_mask] = img_resized[final_lesion_mask]
                plt.subplot(1, 4, 3); plt.imshow(lesion_display_img); plt.title("Extracted Lesion"); plt.axis('off')

                # For visualization of BV mask on the full image context (within lesion)
                hsv_img_resized_full = color.rgb2hsv(img_resized / 255.0)
                bv_candidate_mask_on_full_img = (hsv_img_resized_full[:, :, 0] >= H_MIN_BV) & \
                                                (hsv_img_resized_full[:, :, 0] <= H_MAX_BV) & \
                                                (hsv_img_resized_full[:, :, 1] >= S_MIN_BV) & \
                                                (hsv_img_resized_full[:, :, 1] <= S_MAX_BV) & \
                                                (hsv_img_resized_full[:, :, 2] >= V_MIN_BV) & \
                                                (hsv_img_resized_full[:, :, 2] <= V_MAX_BV)
                actual_bv_mask_2d = np.logical_and(final_lesion_mask, bv_candidate_mask_on_full_img)
                
                plt.subplot(1, 4, 4)
                plt.imshow(actual_bv_mask_2d, cmap='gray')
                plt.title(f"Blue Veil Mask (Present: {current_features['bv_present']})")
                plt.axis('off')
                
                plt.suptitle(f"Blue Veil Detection: {filename}", fontsize=10) # Add filename to suptitle for clarity
                plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
                
                vis_dir = os.path.join(folder_path, 'visualizations_bv')
                os.makedirs(vis_dir, exist_ok=True)
                plt.savefig(os.path.join(vis_dir, f"vis_bv_{filename}.png"))
                plt.close()

        except Exception as e:
            print(f"Error processing {filename} for Blue Veil features: {str(e)}. File skipped.")
            # Ensure skipped files still have a row in the results DataFrame,
            # even if features are defaulted to 0 or NaN.
            # To achieve this consistently, you might want to initialize current_features
            # with all default values at the start of the loop, then fill it if successful.
            # For simplicity here, if an error occurs *before* features are added,
            # we populate defaults and append, similar to the "no lesion" case.
            # Check if current_features was already appended (e.g. by 'no lesion' path)
            if current_features not in results:
                current_features['bv_present'] = 0
                current_features['bv_pixel_count'] = 0
                current_features['bv_area_ratio'] = 0.0
                current_features['bv_mean_R'] = 0.0
                current_features['bv_mean_G'] = 0.0
                current_features['bv_mean_B'] = 0.0
                current_features['bv_std_R'] = 0.0
                current_features['bv_std_G'] = 0.0
                current_features['bv_std_B'] = 0.0
                current_features['bv_mean_H'] = 0.0
                current_features['bv_mean_S'] = 0.0
                current_features['bv_mean_V'] = 0.0
                results.append(current_features)


    # Convert list of dicts to DataFrame
    new_df = pd.DataFrame(results)
    
    # Combine with existing data if available (following original C function's logic)
    final_df_to_return = pd.DataFrame() 
    if not new_df.empty:
        if existing_df is not None:
            # Since existing_df files were skipped, new_df contains only new files.
            # A simple concatenation is appropriate here.
            final_df_to_return = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            final_df_to_return = new_df
    elif existing_df is not None: # new_df is empty, but existing_df was loaded
        final_df_to_return = existing_df
    # If both new_df and existing_df are empty/None, final_df_to_return remains an empty DataFrame.
    
    return final_df_to_return

if __name__ == "__main__":
    # Example Usage (similar to your extract_feature_C __main__ block)
    # Replace with your actual image folder path
    # Ensure this path points to a directory with .jpg, .png, etc. images
    image_folder = r"C:\path\to\your\skin_lesion_images" 
    
    # Path for the CSV that this standalone script will manage for Blue Veil features
    output_csv_for_standalone_run = r"C:\path\to\your\output_folder\blue_veil_features_standalone.csv"

    # Create dummy image folder and images for testing if it doesn't exist
    # This is just for making the __main__ runnable for demonstration
    if not os.path.exists(image_folder) or not os.listdir(image_folder):
        print(f"Test image folder '{image_folder}' not found or empty. Creating dummy data for demonstration.")
        os.makedirs(image_folder, exist_ok=True)
        # Create a few dummy images
        for i in range(3):
            dummy_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            # Make one potentially "blue-ish"
            if i == 1:
                 dummy_img[:,:,0] = 50 # Low Red
                 dummy_img[:,:,1] = 70 # Low Green
                 dummy_img[:,:,2] = 180 # High Blue
            cv2.imwrite(os.path.join(image_folder, f"dummy_lesion_{i+1}.png"), dummy_img)
        print(f"Created dummy images in '{image_folder}'. Please replace with your actual image path for real use.")

    # Ensure the output directory for the CSV exists
    if output_csv_for_standalone_run:
        os.makedirs(os.path.dirname(output_csv_for_standalone_run), exist_ok=True)

    print(f"\nStarting Blue Veil feature extraction from: {image_folder}")
    df_bv_features = extract_feature_BV(
        folder_path=image_folder,
        output_csv=output_csv_for_standalone_run, # Pass CSV path for standalone management
        normalize_colors=True,
        visualize=True # Set to True to generate and save visualizations
    )

    if not df_bv_features.empty:
        # If output_csv_for_standalone_run was used, df_bv_features already includes its old data.
        # So, we just save df_bv_features to that path.
        df_bv_features.to_csv(output_csv_for_standalone_run, index=False)
        print(f"\nSaved/Updated extracted Blue Veil features to: {output_csv_for_standalone_run}")
        print("First 5 rows of the Blue Veil features DataFrame:")
        print(df_bv_features.head())
    else:
        print("\nNo Blue Veil features were extracted or loaded.")

    # To integrate with main_baseline.py:
    # In main_baseline.py, you would typically call:
    # df_bv = extract_feature_BV(image_folder, output_csv=None, normalize_colors=True, visualize=False)
    # And then merge df_bv with your main features DataFrame.
    # The `output_csv` parameter within `extract_feature_BV` is more for its standalone execution
    # or if you maintain separate CSVs per feature type before a final merge.