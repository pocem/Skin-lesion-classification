

import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  # Make sure this import is present
from tqdm import tqdm # Import tqdm


def extract_feature_contrast(folder_path, output_csv=None, visualize=False):
    """
    Extract contrast-related features from skin lesion images in a folder.

    Contrast here is computed as the standard deviation of grayscale pixel intensities
    within a lesion mask estimated by a simple circular + KMeans segmentation (similar to BV).

    Parameters:
    - folder_path (str): Path to folder with images.
    - output_csv (str or None): If provided and file exists, load and append new data.
    - visualize (bool): Show/save visualizations of segmentation and contrast mask.

    Returns:
    - pd.DataFrame with contrast features.
    """
    results = []
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']

    existing_df = None
    if output_csv and os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            print(f"Loaded existing contrast features from {output_csv}")
        except Exception as e:
            print(f"Error loading {output_csv}: {e}")

    files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_exts]

    for filename in tqdm(files, desc="Extracting Contrast Features"): # Wrap loop with tqdm
        if existing_df is not None and filename in existing_df['filename'].values:
            # print(f"Skipping {filename} (already processed).") # Suppress per-file skip message
            continue

        filepath = os.path.join(folder_path, filename)
        # print(f"Processing {filename} for contrast features...") # Suppress per-file message

        try:
            img_bgr = cv2.imread(filepath)
            if img_bgr is None:
                print(f"Could not read {filename}, skipping.")
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)

            h, w = img_resized.shape[:2]

            # Simple circular mask around center as initial guess for lesion area
            center_y, center_x = h // 2, w // 2
            y_coords, x_coords = np.ogrid[:h, :w]
            dist = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
            radius = min(h, w) // 3
            circ_mask = dist <= radius

            # KMeans to segment lesion vs. background
            pixels = img_resized.reshape(-1, 3)
            kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(pixels)
            labels = kmeans.labels_.reshape(h, w)

            center_label = labels[center_y, center_x]
            lesion_mask = np.logical_and(circ_mask, labels == center_label)

            lesion_pixels = img_resized[lesion_mask]

            if lesion_pixels.size == 0:
                print(f"No lesion pixels found in {filename}, skipping.")
                continue

            # Convert lesion pixels to grayscale for contrast calculation
            lesion_gray = cv2.cvtColor(lesion_pixels.reshape(-1,1,3).astype(np.uint8), cv2.COLOR_RGB2GRAY).flatten()

            # Contrast = standard deviation of lesion grayscale intensities
            contrast_std = np.std(lesion_gray)
            contrast_mean = np.mean(lesion_gray)

            features = {
                'filename': filename,
                'contrast_mean_gray': contrast_mean,
                'contrast_std_gray': contrast_std,
                'lesion_pixel_count': lesion_gray.size
            }

            results.append(features)

            if visualize:
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(img_resized)
                plt.title("Resized Image")
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.imshow(lesion_mask, cmap='gray')
                plt.title("Lesion Mask")
                plt.axis('off')

                lesion_display = np.zeros_like(img_resized)
                lesion_display[lesion_mask] = img_resized[lesion_mask]
                plt.subplot(1, 3, 3)
                plt.imshow(lesion_display)
                plt.title(f"Extracted Lesion\nContrast std: {contrast_std:.2f}")
                plt.axis('off')

                vis_dir = os.path.join(folder_path, 'visualizations_contrast')
                os.makedirs(vis_dir, exist_ok=True)
                plt.savefig(os.path.join(vis_dir, f"vis_contrast_{filename}.png"))
                plt.close()

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    df_new = pd.DataFrame(results)

    if existing_df is not None:
        df_final = pd.concat([existing_df, df_new], ignore_index=True)
    else:
        df_final = df_new

    return df_final

if __name__ == "__main__":
    folder = r"C:\path\to\your\skin_lesion_images"
    output_csv_path = r"C:\path\to\your\output\contrast_features.csv"

    # Example call:
    df_contrast = extract_feature_contrast(folder, output_csv=output_csv_path, visualize=True)
    if not df_contrast.empty:
        df_contrast.to_csv(output_csv_path, index=False)
        print(f"Saved contrast features to {output_csv_path}")
        print(df_contrast.head())
    else:
        print("No contrast features extracted.")