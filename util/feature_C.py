import cv2
import numpy as np
import os
import pandas as pd
from skimage import color
from sklearn.cluster import KMeans
from tqdm import tqdm

# set of color features
MINIMAL_COLOR_FEATURES = [
    'c_mean_red', 'c_mean_green',
    'c_std_red', 'c_std_green', 'c_std_blue',
    'c_mean_hue', 'c_mean_saturation',
    'c_std_hue', 'c_std_saturation',
    'c_red_asymmetry', 'c_green_asymmetry', 'c_blue_asymmetry',
]

def _get_default_features(filename: str) -> dict:
    """Returns a dictionary of default feature values for failed cases."""
    base_filename = os.path.splitext(filename)[0]
    features = {'filename': base_filename}
    for f_name in MINIMAL_COLOR_FEATURES:
        features[f_name] = 0.0
    return features

def extract_feature_C(folder_path, output_csv=None, normalize_colors=True, visualize=False):
    """
    Extracts a minimal, non-redundant, and one-hot encoded set of color features.
    """
    results = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_extensions]

    for filename in tqdm(image_files, desc="Extracting Refined Color Features (C)"):
        image_path = os.path.join(folder_path, filename)

        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Error reading {filename}, skipping.")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))

            # --- Segmentation ---
            lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            pixels = lab_img.reshape(-1, 3)
            kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(pixels)
            labels = kmeans.labels_.reshape(img.shape[:2])

            if np.sum(labels == 0) > np.sum(labels == 1):
                lesion_label = 1
            else:
                lesion_label = 0
            
            final_mask = (labels == lesion_label)
            lesion_pixels = img[final_mask]

            if lesion_pixels.shape[0] < 50:
                print(f"Warning: Very few lesion pixels found in {filename}, skipping.")
                results.append(_get_default_features(filename))
                continue

            # --- Feature Calculation ---
            features = {'filename': filename}
            
            if normalize_colors:
                lesion_pixels_norm = lesion_pixels / 255.0
            else:
                lesion_pixels_norm = lesion_pixels

            # --- RGB Features ---
            features['c_mean_red'] = np.mean(lesion_pixels_norm[:, 0])
            features['c_mean_green'] = np.mean(lesion_pixels_norm[:, 1])
            features['c_std_red'] = np.std(lesion_pixels_norm[:, 0])
            features['c_std_green'] = np.std(lesion_pixels_norm[:, 1])
            features['c_std_blue'] = np.std(lesion_pixels_norm[:, 2])

            # --- HSV Features ---
            hsv_pixels = color.rgb2hsv(lesion_pixels_norm)
            features['c_mean_hue'] = np.mean(hsv_pixels[:, 0])
            features['c_mean_saturation'] = np.mean(hsv_pixels[:, 1])
            features['c_std_hue'] = np.std(hsv_pixels[:, 0])
            features['c_std_saturation'] = np.std(hsv_pixels[:, 1])

            # --- Color Asymmetry ---
            w, _ = img.shape
            left_mask, right_mask = np.zeros_like(final_mask), np.zeros_like(final_mask)
            left_mask[:, :w//2] = final_mask[:, :w//2]
            right_mask[:, w//2:] = final_mask[:, w//2:]
            
            left_pixels = img[left_mask]
            right_pixels = img[right_mask]
            
            if left_pixels.size > 0 and right_pixels.size > 0:
                if normalize_colors:
                    left_pixels_norm = left_pixels / 255.0
                    right_pixels_norm = right_pixels / 255.0
                features['c_red_asymmetry'] = abs(np.mean(left_pixels_norm[:, 0]) - np.mean(right_pixels_norm[:, 0]))
                features['c_green_asymmetry'] = abs(np.mean(left_pixels_norm[:, 1]) - np.mean(right_pixels_norm[:, 1]))
                features['c_blue_asymmetry'] = abs(np.mean(left_pixels_norm[:, 2]) - np.mean(right_pixels_norm[:, 2]))
            else:
                features['c_red_asymmetry'], features['c_green_asymmetry'], features['c_blue_asymmetry'] = 0.0, 0.0, 0.0
                   
            results.append(features)

        except Exception as e:
            print(f"CRITICAL Error processing {filename}: {e}")
            results.append(_get_default_features(filename))

    df = pd.DataFrame(results)

    # Reorder columns for consistency
    if not df.empty:
        final_columns = ['filename'] + MINIMAL_COLOR_FEATURES
        df = df[final_columns]

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\nSaved refined color features to {output_csv}")

    return df

if __name__ == "__main__":
    # folder of the original images 
    folder_path = r''
    output_csv_path = os.path.join(folder_path, "color_features.csv") 

    df_features = extract_feature_C(
        folder_path=folder_path,
        output_csv=output_csv_path,
        normalize_colors=True,
        visualize=False
    )

    if not df_features.empty:
        print("\nSuccessfully extracted refined color features with One-Hot Encoding. First 5 rows:")
        print(df_features.head())
        print("\nFeatures calculated:")
        print(df_features.columns.tolist())
    else:
        print("\nNo color features were extracted.")