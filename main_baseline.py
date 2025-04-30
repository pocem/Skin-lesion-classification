import sys
import os
from os.path import join, exists
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from util.img_util import readImageFile, saveImageFile
# Import feature extraction modules
from util.feature_A import extract_feature_A
from util.feature_B import extract_feature_B
from util.feature_C import extract_feature_C

def create_feature_dataset(original_img_dir, mask_img_dir, output_csv_path, labels_csv=None):
    """
    Process all images and create a dataset CSV with extracted features
    
    Args:
        original_img_dir: Directory with original wound images
        mask_img_dir: Directory with masked/segmented wound images
        output_csv_path: Path to save the feature CSV
        labels_csv: Path to CSV with image labels (if available)
    """
    print("Starting feature extraction process...")
    
    # Initialize the dataframe to store features
    data = {'filename': []}
    
    # If we have labels, load them
    if labels_csv and exists(labels_csv):
        print(f"Loading labels from {labels_csv}")
        labels_df = pd.read_csv(labels_csv)
        label_dict = dict(zip(labels_df['filename'], labels_df['label']))
        data['label'] = []
    
    # Get all image files from original directory
    img_files = [f for f in os.listdir(original_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(img_files)} images to process")
    
    # Process each image
    for i, img_file in enumerate(img_files):
        if i % 10 == 0:
            print(f"Processing image {i+1}/{len(img_files)}: {img_file}")
            
        # Add filename to data
        data['filename'].append(img_file)
        
        # Add label if available
        if labels_csv and exists(labels_csv):
            data['label'].append(label_dict.get(img_file, -1))  # -1 for unknown
        
        # Read original image
        orig_img_path = join(original_img_dir, img_file)
        orig_img = readImageFile(orig_img_path)
        
        # Read corresponding mask image if available
        mask_img_path = join(mask_img_dir, img_file)
        mask_img = readImageFile(mask_img_path) if exists(mask_img_path) else None
        
        # Extract features from original image
        feat_a = extract_feature_A(orig_img)
        feat_b = extract_feature_B(orig_img)
        feat_c = extract_feature_C(orig_img, mask_img)  # Assuming feature C might use the mask
        
        # Add features to data dictionary (handle first image case)
        if i == 0:
            # Initialize feature columns based on return types
            if isinstance(feat_a, (int, float)):
                data['feat_A'] = [feat_a]
            elif isinstance(feat_a, (list, np.ndarray)):
                for j, val in enumerate(feat_a):
                    data[f'feat_A_{j}'] = [val]
                    
            if isinstance(feat_b, (int, float)):
                data['feat_B'] = [feat_b]
            elif isinstance(feat_b, (list, np.ndarray)):
                for j, val in enumerate(feat_b):
                    data[f'feat_B_{j}'] = [val]
                    
            if isinstance(feat_c, (int, float)):
                data['feat_C'] = [feat_c]
            elif isinstance(feat_c, (list, np.ndarray)):
                for j, val in enumerate(feat_c):
                    data[f'feat_C_{j}'] = [val]
        else:
            # Add values for subsequent images
            if isinstance(feat_a, (int, float)):
                data['feat_A'].append(feat_a)
            elif isinstance(feat_a, (list, np.ndarray)):
                for j, val in enumerate(feat_a):
                    data[f'feat_A_{j}'].append(val)
                    
            if isinstance(feat_b, (int, float)):
                data['feat_B'].append(feat_b)
            elif isinstance(feat_b, (list, np.ndarray)):
                for j, val in enumerate(feat_b):
                    data[f'feat_B_{j}'].append(val)
                    
            if isinstance(feat_c, (int, float)):
                data['feat_C'].append(feat_c)
            elif isinstance(feat_c, (list, np.ndarray)):
                for j, val in enumerate(feat_c):
                    data[f'feat_C_{j}'].append(val)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)
    print(f"Feature dataset created and saved to {output_csv_path}")
    print(f"Dataset shape: {df.shape}, with features: {[col for col in df.columns if col.startswith('feat_')]}")
    return df

def main(original_img_dir, mask_img_dir, labels_csv_path, output_csv_path, result_path, recreate_features=False):
    """
    Main function for baseline approach
    
    Args:
        original_img_dir: Directory with original wound images
        mask_img_dir: Directory with masked/segmented wound images
        labels_csv_path: Path to CSV with labels (filename,label)
        output_csv_path: Path to save the extracted features CSV
        result_path: Path to save the classification results
        recreate_features: Whether to recreate features or use existing CSV
    """
    print("\n--- BASELINE APPROACH ---\n")
    
    # Create or load feature dataset
    if recreate_features or not exists(output_csv_path):
        print(f"Creating new feature dataset at {output_csv_path}")
        data_df = create_feature_dataset(original_img_dir, mask_img_dir, output_csv_path, labels_csv=labels_csv_path)
    else:
        print(f"Loading existing feature dataset from {output_csv_path}")
        data_df = pd.read_csv(output_csv_path)
    
    # Select baseline features
    baseline_feats = [col for col in data_df.columns if col.startswith("feat_")]
    print(f"Using {len(baseline_feats)} features: {baseline_feats}")
    
    x_all = data_df[baseline_feats]
    y_all = data_df["label"]
    
    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3, random_state=42)
    print(f"Training set size: {x_train.shape[0]}, Test set size: {x_test.shape[0]}")
    
    # Train classifier
    print("Training logistic regression classifier...")
    clf = LogisticRegression(max_iter=1000, verbose=1)
    clf.fit(x_train, y_train)
    
    # Test classifier
    print("Testing classifier...")
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Save results
    print(f"Saving results to {result_path}")
    result_df = data_df.loc[x_test.index, ["filename"]].copy()
    result_df['true_label'] = y_test.values
    result_df['predicted_label'] = y_pred
    result_df.to_csv(result_path, index=False)
    print("Results saved successfully!")

if __name__ == "__main__":
    # Configure paths - adjust these to your specific PC folders
    original_img_dir = "./data/"  # Original wound images
    mask_img_dir = "./data_masks/"  # Masked/segmented images
    labels_csv_path = "./dataset.csv"  # CSV with labels
    
    # Output files
    output_csv_path = "./dataset_baseline_features.csv"  # Where to save extracted features
    result_path = "./result/result_baseline.csv"  # Where to save results
    
    # Make sure result directory exists
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    
    # Run main function (set recreate_features=True to force regeneration of features)
    main(original_img_dir, mask_img_dir, labels_csv_path, output_csv_path, result_path, recreate_features=True)
