import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage import color
from sklearn.cluster import KMeans
from tqdm import tqdm

def extract_blue_veil_features(folder_path: str, output_csv: str = None, visualize: bool = False) -> pd.DataFrame:
    """
    Extracts a single, powerful "Blue Veil Area Ratio" feature from skin lesion images.

    This function segments the lesion, identifies pixels within the lesion that match
    the color signature of a blue-whitish veil, and calculates the ratio of this
    area to the total lesion area.

    Args:
        folder_path: Path to the folder containing lesion images.
        output_csv: Optional path to save the resulting features to a CSV file.
        visualize: If True, saves visualizations of the process.

    Returns:
        A pandas DataFrame with 'filename' and the 'bv_area_ratio' feature.
    """
    results = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_extensions]

    for filename in tqdm(image_files, desc="Extracting Blue Veil Ratio"):
        image_path = os.path.join(folder_path, filename)
        bv_area_ratio = 0.0  # Default value if errors occur

        try:
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                print(f"Warning: Error reading {filename}, skipping.")
                base_filename = os.path.splitext(filename)[0]
                results.append({'filename': base_filename, 'bv_area_ratio': bv_area_ratio})
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
            h, w, _ = img_resized.shape

            # --- Step 1: Segment the Lesion ---
            lab_img = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
            pixels = lab_img.reshape(-1, 3)
            kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(pixels)
            labels = kmeans.labels_.reshape(h, w)

            # Heuristic: The lesion is the smaller of the two clusters
            lesion_label = 1 if np.sum(labels == 1) < np.sum(labels == 0) else 0
            final_lesion_mask = (labels == lesion_label)
            
            lesion_pixel_count = np.sum(final_lesion_mask)
            if lesion_pixel_count < 100: # Skip if lesion is too small
                results.append({'filename': filename, 'bv_area_ratio': bv_area_ratio})
                continue

            # --- Step 2: Identify Blue Veil Pixels within the Lesion ---
            lesion_pixels_rgb = img_resized[final_lesion_mask]
            lesion_pixels_hsv = color.rgb2hsv(lesion_pixels_rgb / 255.0)

            # HSV thresholds for blue-whitish veil
            H_MIN, H_MAX = 0.52, 0.75  # Hues from cyan-blue to blue-magenta
            S_MIN, S_MAX = 0.10, 0.75  # Allow desaturated (whitish/grayish) blues
            V_MIN, V_MAX = 0.30, 0.95  # Avoid pure black and pure white

            bv_mask_1d = (lesion_pixels_hsv[:, 0] >= H_MIN) & (lesion_pixels_hsv[:, 0] <= H_MAX) & \
                         (lesion_pixels_hsv[:, 1] >= S_MIN) & (lesion_pixels_hsv[:, 1] <= S_MAX) & \
                         (lesion_pixels_hsv[:, 2] >= V_MIN) & (lesion_pixels_hsv[:, 2] <= V_MAX)

            bv_pixel_count = np.sum(bv_mask_1d)
            
            # --- Step 3: Calculate the Final Feature ---
            bv_area_ratio = bv_pixel_count / lesion_pixel_count

            results.append({'filename': filename, 'bv_area_ratio': bv_area_ratio})
            
            # --- Visualization ---
            if visualize:
                bv_mask_2d = np.zeros_like(final_lesion_mask)
                # Create a 2D mask of where the blue veil is inside the lesion
                lesion_indices = np.where(final_lesion_mask)
                bv_indices_in_lesion = (lesion_indices[0][bv_mask_1d], lesion_indices[1][bv_mask_1d])
                bv_mask_2d[bv_indices_in_lesion] = True
                
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1); plt.imshow(img_resized); plt.title("Original Image")
                plt.subplot(1, 3, 2); plt.imshow(final_lesion_mask, cmap='gray'); plt.title("Lesion Mask")
                plt.subplot(1, 3, 3); plt.imshow(bv_mask_2d, cmap='gray'); plt.title(f"Blue Veil Mask\nRatio: {bv_area_ratio:.2f}")
                
                vis_dir = os.path.join(folder_path, 'visualizations_bv')
                os.makedirs(vis_dir, exist_ok=True)
                plt.savefig(os.path.join(vis_dir, f"vis_bv_{filename}.png"))
                plt.close()

        except Exception as e:
            print(f"CRITICAL Error processing {filename}: {e}")
            results.append({'filename': filename, 'bv_area_ratio': 0.0})

    df = pd.DataFrame(results)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\nSaved Blue Veil features to {output_csv}")

    return df

if __name__ == '__main__':
    folder_path = r"C:\Users\misog\portfolio\Machine learning skin lesion project\matched_data\images"
    output_csv_path = os.path.join(folder_path, "color_features.csv") 

    df_bv = extract_blue_veil_features(
        folder_path=folder_path,
        output_csv=output_csv_path,
        visualize=False
    )

    if not df_bv.empty:
        print("\nSuccessfully extracted refined Blue Veil features:")
        print(df_bv.head())
    else:
        print("\nNo Blue Veil features were extracted.")