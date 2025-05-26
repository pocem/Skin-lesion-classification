import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage import color # For RGB to Gray
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans

def extract_feature_H(folder_path, output_csv=None, visualize=False, 
                      distances=[1, 3, 5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    Function to extract Haralick texture features from skin lesion images in a folder.
    
    Parameters:
    folder_path (str): Path to the folder containing skin lesion images.
    output_csv (str): Path to output CSV file for Haralick features. If the file exists, 
                      features for new images will be added to it.
    visualize (bool): Whether to visualize the segmentation and the grayscale lesion patch.
    distances (list): List of pixel distances for GLCM computation.
    angles (list): List of angles (in radians) for GLCM computation.
    
    Returns:
    pd.DataFrame: DataFrame containing Haralick texture features.
    """
    results = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    haralick_props = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
    feature_names = [f'h_{prop}_{stat}' for prop in haralick_props for stat in ['mean', 'std']]

    existing_df = None
    if output_csv and os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            print(f"Loaded existing Haralick features from {output_csv}")
        except Exception as e:
            print(f"Error loading existing CSV {output_csv}: {str(e)}")
            existing_df = None

    image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_extensions]

    for filename in image_files:
        if existing_df is not None and filename in existing_df['filename'].values:
            print(f"Skipping {filename} - already processed for Haralick features in existing CSV.")
            continue
            
        image_path = os.path.join(folder_path, filename)
        print(f"Processing {filename} for Haralick features...")
        
        current_features = {'filename': filename}
        # Initialize Haralick features to 0.0 in case of errors or non-computable scenarios
        for name in feature_names:
            current_features[name] = 0.0

        try:
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                print(f"Error reading {filename}, skipping...")
                continue
                
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
            h, w = img_resized.shape[:2]

            # Step 1: Segment the lesion (same as extract_feature_C and _BV)
            center_y, center_x = h // 2, w // 2
            y_coords, x_coords = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            initial_mask_radius = min(h, w) // 2.8
            circular_mask = dist_from_center <= initial_mask_radius
            
            pixels_for_kmeans = img_resized.reshape(-1, 3)
            kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(pixels_for_kmeans)
            labels = kmeans.labels_.reshape(h, w)
            
            center_label = labels[center_y, center_x]
            kmeans_mask = (labels == center_label)
            
            final_lesion_mask = np.logical_and(circular_mask, kmeans_mask)
            
            lesion_pixel_count = np.sum(final_lesion_mask)
            if lesion_pixel_count < 25: # Minimum 5x5 area roughly, or too few pixels for GLCM
                print(f"Lesion in {filename} is too small or not detected ({lesion_pixel_count} pixels), skipping Haralick.")
                results.append(current_features) # Append features (all zeros)
                # Optional: Minimal visualization for skipped small lesion
                if visualize:
                    plt.figure(figsize=(12,4))
                    plt.subplot(1,3,1); plt.imshow(img_resized); plt.title("Resized Original")
                    plt.subplot(1,3,2); plt.imshow(final_lesion_mask, cmap='gray'); plt.title("Lesion Mask (Too Small)")
                    lesion_rgb_display = np.zeros_like(img_resized)
                    if lesion_pixel_count > 0: lesion_rgb_display[final_lesion_mask] = img_resized[final_lesion_mask]
                    plt.subplot(1,3,3); plt.imshow(lesion_rgb_display); plt.title("Extracted Lesion (Too Small)")
                    plt.suptitle(f"{filename} - Lesion Too Small for Haralick", fontsize=10)
                    vis_dir = os.path.join(folder_path, 'visualizations_h')
                    os.makedirs(vis_dir, exist_ok=True)
                    plt.savefig(os.path.join(vis_dir, f"vis_h_{filename}_small_lesion.png"))
                    plt.close()
                continue

            # Step 2: Prepare grayscale lesion patch for Haralick features
            img_gray_resized = color.rgb2gray(img_resized) # Converts to float [0,1]
            img_gray_resized_uint8 = (img_gray_resized * 255).astype(np.uint8) # Convert to uint8 [0,255] for GLCM

            rows, cols = np.where(final_lesion_mask)
            min_r, max_r = np.min(rows), np.max(rows)
            min_c, max_c = np.min(cols), np.max(cols)
            
            # Crop the grayscale image and the mask to the bounding box of the lesion
            lesion_patch_gray = img_gray_resized_uint8[min_r:max_r+1, min_c:max_c+1]
            mask_patch = final_lesion_mask[min_r:max_r+1, min_c:max_c+1]

            # Ensure the effective region within the patch is not uniform
            unique_values_in_lesion_patch = np.unique(lesion_patch_gray[mask_patch])
            if len(unique_values_in_lesion_patch) < 2:
                print(f"Lesion patch in {filename} is uniform, Haralick features set to 0.")
                # Features are already 0.0 by default
            else:
                # Compute GLCM
                # The mask in graycomatrix should mark valid pixels for pair counting
                glcm = graycomatrix(lesion_patch_gray, 
                                    distances=distances, 
                                    angles=angles, 
                                    levels=256,
                                    symmetric=True, 
                                    normed=True,
                                    mask=mask_patch) # Use the mask_patch here
                
                # Compute Haralick properties
                for prop_name in haralick_props:
                    prop_values = graycoprops(glcm, prop_name) # Array of shape (len(distances), len(angles))
                    current_features[f'h_{prop_name}_mean'] = np.mean(prop_values)
                    current_features[f'h_{prop_name}_std'] = np.std(prop_values)

            results.append(current_features)

            if visualize:
                plt.figure(figsize=(16, 4))
                
                plt.subplot(1, 4, 1); plt.imshow(img_resized); plt.title("Resized Original"); plt.axis('off')
                plt.subplot(1, 4, 2); plt.imshow(final_lesion_mask, cmap='gray'); plt.title("Lesion Mask"); plt.axis('off')
                
                lesion_rgb_display = np.zeros_like(img_resized)
                lesion_rgb_display[final_lesion_mask] = img_resized[final_lesion_mask]
                plt.subplot(1, 4, 3); plt.imshow(lesion_rgb_display); plt.title("Extracted Lesion (RGB)"); plt.axis('off')
                
                # Display the actual grayscale patch used for GLCM
                # Create a display patch that shows lesion_patch_gray where mask_patch is true
                display_gray_patch = np.zeros_like(lesion_patch_gray, dtype=float)
                if lesion_patch_gray.size > 0 and mask_patch.size > 0 : # Check if patch is not empty
                    display_gray_patch[mask_patch] = lesion_patch_gray[mask_patch] / 255.0 # Normalize for display
                plt.subplot(1, 4, 4)
                plt.imshow(display_gray_patch, cmap='gray', vmin=0, vmax=1)
                plt.title("Grayscale Lesion Patch (for GLCM)"); plt.axis('off')
                
                plt.suptitle(f"Haralick Feature Extraction: {filename}", fontsize=10)
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                
                vis_dir = os.path.join(folder_path, 'visualizations_h')
                os.makedirs(vis_dir, exist_ok=True)
                plt.savefig(os.path.join(vis_dir, f"vis_h_{filename}.png"))
                plt.close()

        except Exception as e:
            print(f"Error processing {filename} for Haralick features: {str(e)}. Features set to 0.")
            # Append the current_features (which are initialized to 0.0)
            # Ensure filename is always present
            current_features_error = {'filename': filename}
            for name in feature_names:
                current_features_error[name] = 0.0
            results.append(current_features_error)


    new_df = pd.DataFrame(results)
    
    final_df_to_return = pd.DataFrame()
    if not new_df.empty:
        if existing_df is not None:
            final_df_to_return = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=['filename'], keep='last')
        else:
            final_df_to_return = new_df
    elif existing_df is not None:
        final_df_to_return = existing_df
    
    return final_df_to_return

if __name__ == "__main__":
    # Example Usage
    image_folder = r"C:\path\to\your\skin_lesion_images"  # Replace with your actual image folder path
    output_csv_for_standalone_run = r"C:\path\to\your\output_folder\haralick_features_standalone.csv"

    # Create dummy image folder and images for testing if it doesn't exist
    if not os.path.exists(image_folder) or not os.listdir(image_folder):
        print(f"Test image folder '{image_folder}' not found or empty. Creating dummy data for demonstration.")
        os.makedirs(image_folder, exist_ok=True)
        for i in range(3):
            # Create a base image
            dummy_img = np.random.randint(50, 200, (150, 150, 3), dtype=np.uint8)
            # Create a somewhat textured "lesion" in the center for one image
            if i == 1:
                center_y, center_x = 75, 75
                radius = 30
                for r_idx in range(center_y - radius, center_y + radius):
                    for c_idx in range(center_x - radius, center_x + radius):
                        if (r_idx - center_y)**2 + (c_idx - center_x)**2 < radius**2:
                             dummy_img[r_idx, c_idx, :] = np.clip(dummy_img[r_idx, c_idx, :] + np.random.randint(-30, 30, 3), 0, 255)
            # Create a more uniform "lesion" for another
            elif i == 2:
                center_y, center_x = 75, 75
                radius = 25
                for r_idx in range(center_y - radius, center_y + radius):
                    for c_idx in range(center_x - radius, center_x + radius):
                        if (r_idx - center_y)**2 + (c_idx - center_x)**2 < radius**2:
                             dummy_img[r_idx, c_idx, :] = [100, 110, 120] # Uniform color
            cv2.imwrite(os.path.join(image_folder, f"dummy_lesion_h_{i+1}.png"), dummy_img)
        print(f"Created dummy images in '{image_folder}'. Please replace with your actual image path for real use.")

    if output_csv_for_standalone_run:
         os.makedirs(os.path.dirname(output_csv_for_standalone_run), exist_ok=True)

    print(f"\nStarting Haralick feature extraction from: {image_folder}")
    df_h_features = extract_feature_H(
        folder_path=image_folder,
        output_csv=output_csv_for_standalone_run,
        visualize=True # Set to True for visualizations
    )

    if not df_h_features.empty:
        # Save the potentially updated DataFrame
        df_h_features.to_csv(output_csv_for_standalone_run, index=False)
        print(f"\nSaved/Updated extracted Haralick features to: {output_csv_for_standalone_run}")
        print("First 5 rows of the Haralick features DataFrame:")
        print(df_h_features.head())
    else:
        print("\nNo Haralick features were extracted or loaded.")

    # For main_baseline.py integration:
    # df_h = extract_feature_H(image_folder, output_csv=None, visualize=False)
    # Then merge df_h with other features.