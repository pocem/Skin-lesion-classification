import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage import segmentation, color, io, filters, measure, transform, morphology
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.ndimage import distance_transform_edt

folder_path= "your path"

def extract_asymmetry_features(folder_path, output_csv=None, visualize=False):
    """
    Function to extract asymmetry features from skin lesion images in a folder
    
    Parameters:
    folder_path (str): Path to the folder containing skin lesion images
    output_csv (str): Path to output CSV file. If the file exists, features will be added to it
    visualize (bool): Whether to visualize the asymmetry calculations
    
    Returns:
    pd.DataFrame: DataFrame containing asymmetry features for all images
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
    
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is an image
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in valid_extensions:
            continue
            
        # Skip if the image is already in the existing dataframe
        if existing_df is not None and filename in existing_df['filename'].values:
            print(f"Skipping {filename} - already processed")
            continue
            
        image_path = os.path.join(folder_path, filename)
        print(f"Processing {filename}...")
        
        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error reading {filename}, skipping...")
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize for consistency
            img = cv2.resize(img, (256, 256))
            
            # Get binary mask using existing segmentation approach
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            segments = segmentation.slic(img, n_segments=100, compactness=10, sigma=1)
            h, w = img.shape[:2]
            center_y, center_x = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            mask = dist_from_center <= min(h, w) // 3
            pixels = img.reshape(-1, 3)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(pixels)
            labels = kmeans.labels_.reshape(h, w)
            center_label = labels[center_y, center_x]
            refined_mask = (labels == center_label)
            final_mask = np.logical_and(mask, refined_mask)
            
            # Convert to binary for asymmetry calculations
            binary_mask = final_mask.astype(np.uint8)
            
            # Calculate all asymmetry features
            features = {'filename': filename}
            
            # 1. Basic mirror asymmetry
            basic_score = compute_basic_asymmetry(binary_mask)
            features['a_basic'] = basic_score
            
            # 2. PCA-aligned rotational asymmetry
            pca_score = compute_pca_asymmetry(binary_mask)
            features['a_pca'] = pca_score
            
            # 3. Boundary-weighted asymmetry
            boundary_score = compute_boundary_asymmetry(binary_mask)
            features['a_boundary'] = boundary_score
            
            # 4. Combined weighted score
            combined_score = 0.4*basic_score + 0.3*pca_score + 0.3*boundary_score
            features['a_combined'] = min(combined_score, 1.0)
            
            # Visualization
            # if visualize:
            #     visualize_asymmetry(img, binary_mask, features, filename)
            
            results.append(features)
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    # Convert to DataFrame
    new_df = pd.DataFrame(results)
    
    # Combine with existing data if available
    if existing_df is not None and not new_df.empty:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    elif not new_df.empty:
        combined_df = new_df
    else:
        combined_df = existing_df if existing_df is not None else pd.DataFrame()
    
    # Save to CSV if an output path is specified
    if output_csv and not combined_df.empty:
        combined_df.to_csv(output_csv, index=False)
        print(f"Features saved to {output_csv}")
    
    return combined_df

def compute_basic_asymmetry(mask):
    """Compute basic vertical/horizontal mirror asymmetry"""
    labeled = measure.label(mask)
    regions = measure.regionprops(labeled)
    
    if not regions:
        return 1.0
    
    lesion = max(regions, key=lambda r: r.area)
    minr, minc, maxr, maxc = lesion.bbox
    cropped = mask[minr:maxr, minc:maxc]
    
    vert_flip = np.fliplr(cropped)
    horiz_flip = np.flipud(cropped)
    
    vert_diff = np.sum(np.abs(cropped.astype(int) - vert_flip.astype(int)))
    horiz_diff = np.sum(np.abs(cropped.astype(int) - horiz_flip.astype(int)))
    
    total_area = lesion.area
    if total_area == 0:
        return 1.0
    
    return (vert_diff + horiz_diff) / (2 * total_area)

def compute_pca_asymmetry(mask):
    """Compute rotation-invariant asymmetry using PCA alignment"""
    try:
        y, x = np.where(mask)
        if len(x) < 2:
            return 1.0
            
        pca = PCA(n_components=2)
        coords = np.column_stack([x, y])
        pca.fit(coords - coords.mean(axis=0))
        angle = np.arctan2(pca.components_[0][1], pca.components_[0][0]) * 180 / np.pi
        
        rotated = transform.rotate(mask.astype(float), angle, resize=True, order=0)
        return np.sum(np.abs(rotated - np.fliplr(rotated))) / (2 * np.sum(mask))
    except:
        return compute_basic_asymmetry(mask)  # Fallback

def compute_boundary_asymmetry(mask):
    """Compute boundary-weighted asymmetry"""
    boundary = mask - morphology.binary_erosion(mask)
    weights = distance_transform_edt(~boundary)
    vert_flip = np.fliplr(mask)
    vert_diff = np.sum(weights * np.abs(mask.astype(int) - vert_flip.astype(int)))
    return vert_diff / max(np.sum(weights), 1)  # Avoid division by zero

if __name__ == "__main__":
    folder_path = "your path"
    output_csv_path = folder_path + "/asymmetry_features.csv"  # Or use os.path.join as you prefer

    # Run the feature extraction and save results to CSV
    df = extract_asymmetry_features(folder_path, output_csv=output_csv_path, visualize=False)

    # Optional: print the first few lines of the dataframe to check
    print(df.head())


extract_asymmetry_features(folder_path, visualize=False)





# for the report:

# def visualize_asymmetry(img, mask, features, filename):
#     """Visualize asymmetry calculations"""
#     plt.figure(figsize=(15, 10))
    
#     # Original image with mask
#     plt.subplot(2, 3, 1)
#     plt.imshow(img)
#     plt.title("Original Image")
    
#     plt.subplot(2, 3, 2)
#     plt.imshow(mask, cmap='gray')
#     plt.title("Lesion Mask")
    
#     # Basic asymmetry visualization
#     labeled = measure.label(mask)
#     regions = measure.regionprops(labeled)
#     lesion = max(regions, key=lambda r: r.area)
#     minr, minc, maxr, maxc = lesion.bbox
#     cropped = mask[minr:maxr, minc:maxc]
    
#     plt.subplot(2, 3, 3)
#     plt.imshow(cropped, cmap='gray')
#     plt.contour(np.fliplr(cropped), colors='red', linewidths=1)
#     plt.title(f"Basic Asymmetry\nScore: {features['a_basic']:.3f}")
    
#     # PCA asymmetry visualization
#     try:
#         y, x = np.where(cropped)
#         pca = PCA(n_components=2)
#         pca.fit(np.column_stack([x, y]))
#         angle = np.arctan2(pca.components_[0][1], pca.components_[0][0]) * 180 / np.pi
#         rotated = transform.rotate(cropped.astype(float), angle, resize=True, order=0)
        
#         plt.subplot(2, 3, 4)
#         plt.imshow(rotated, cmap='gray')
#         plt.contour(np.fliplr(rotated), colors='red', linewidths=1)
#         plt.title(f"PCA-Aligned Asymmetry\nScore: {features['a_pca']:.3f}")
#     except:
#         pass
    
#     # Boundary-weighted visualization
#     boundary = cropped - morphology.binary_erosion(cropped)
#     weights = distance_transform_edt(~boundary)
    
#     plt.subplot(2, 3, 5)
#     plt.imshow(weights, cmap='viridis')
#     plt.title("Boundary Weights")
    
#     plt.subplot(2, 3, 6)
#     plt.imshow(cropped, cmap='gray')
#     plt.contour(np.fliplr(cropped), colors='red', linewidths=1)
#     plt.title(f"Combined Score: {features['a_combined']:.3f}")
    
#     plt.tight_layout()
    
#     # Save visualization
#     vis_dir = os.path.join(os.path.dirname(folder_path), 'asymmetry_visualizations')
#     os.makedirs(vis_dir, exist_ok=True)
#     plt.savefig(os.path.join(vis_dir, f"asymmetry_{filename}"))
#     plt.close()
