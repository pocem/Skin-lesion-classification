import sys
import os
from os.path import join, exists
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Import custom modules with error handling
try:
    from util.img_util import readImageFile
    from util.feature_A import extract_asymmetry_features
    from util.feature_B import extract_border_features
    from util.feature_C import extract_feature_C
    from models_evaluation import train_and_select_model
except ImportError as e:
    print(f"Warning: Could not import custom modules: {e}")
    print("Make sure the following modules exist:")
    print("- util/img_util.py with readImageFile function")
    print("- util/feature_A.py with extract_asymmetry_features function")
    print("- util/feature_B.py with extract_border_features function")
    print("- util/feature_C.py with extract_feature_C function")
    print("- models_evaluation.py with train_and_select_model function")
    sys.exit(1)

def create_feature_dataset(original_img_dir, mask_img_dir, output_csv_path, labels_csv=None):
    print("Starting feature extraction process...")
    
    # Validate input directories
    if not exists(original_img_dir):
        raise FileNotFoundError(f"Original image directory not found: {original_img_dir}")
    
    if not exists(mask_img_dir):
        print(f"Warning: Mask directory not found: {mask_img_dir}")
        print("Continuing without mask images...")

    if labels_csv and exists(labels_csv):
        print(f"Loading labels from {labels_csv}")
        try:
            labels_df = pd.read_csv(labels_csv)
            if 'filename' not in labels_df.columns or 'label' not in labels_df.columns:
                raise ValueError("Labels CSV must contain 'filename' and 'label' columns")
            label_dict = dict(zip(labels_df['filename'], labels_df['label']))
        except Exception as e:
            print(f"Error loading labels: {e}")
            label_dict = {}
    else:
        print("No labels file provided or file doesn't exist")
        label_dict = {}

    img_files = [f for f in os.listdir(original_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(img_files)} images to process")
    
    if len(img_files) == 0:
        raise ValueError(f"No image files found in {original_img_dir}")

    all_features = []

    for i, img_file in enumerate(img_files):
        if i % 10 == 0:
            print(f"Processing image {i+1}/{len(img_files)}: {img_file}")

        orig_img_path = join(original_img_dir, img_file)
        mask_img_path = join(mask_img_dir, img_file) if exists(mask_img_dir) else None

        try:
            orig_img = readImageFile(orig_img_path)
            mask_img = readImageFile(mask_img_path) if (mask_img_path and exists(mask_img_path)) else None

            features = {"filename": img_file}

            # Add label if available
            if label_dict:
                features["label"] = label_dict.get(img_file, -1)

            # Update features from each module
            features.update(extract_asymmetry_features(orig_img))
            features.update(extract_border_features(orig_img))
            features.update(extract_feature_C(orig_img, mask_img))

            all_features.append(features)
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue

    if len(all_features) == 0:
        raise ValueError("No features were extracted from any images")

    df = pd.DataFrame(all_features)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    df.to_csv(output_csv_path, index=False)
    print(f"Feature dataset saved to {output_csv_path}")
    
    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    print(f"Dataset shape: {df.shape}, with {len(feature_cols)} features")
    return df

def main(original_img_dir, mask_img_dir, labels_csv_path, output_csv_path, result_path, recreate_features=False):
    print("\n--- BASELINE APPROACH ---\n")
    
    # Validate paths
    if not original_img_dir or not output_csv_path or not result_path:
        raise ValueError("All path parameters must be provided and non-empty")

    if recreate_features or not exists(output_csv_path):
        print(f"Creating new feature dataset at {output_csv_path}")
        data_df = create_feature_dataset(original_img_dir, mask_img_dir, output_csv_path, labels_csv=labels_csv_path)
    else:
        print(f"Loading existing feature dataset from {output_csv_path}")
        data_df = pd.read_csv(output_csv_path)

    baseline_feats = [col for col in data_df.columns if col.startswith("feat_")]
    print(f"Using {len(baseline_feats)} features: {baseline_feats}")
    
    if len(baseline_feats) == 0:
        raise ValueError("No features found in dataset. Make sure feature extraction is working correctly.")

    # Check if labels exist
    if 'label' not in data_df.columns:
        raise ValueError("No 'label' column found in dataset. Cannot train model without labels.")

    x_all = data_df[baseline_feats]
    y_all = data_df["label"]
    
    # Remove rows with missing labels
    valid_mask = y_all != -1
    x_all = x_all[valid_mask]
    y_all = y_all[valid_mask]
    
    if len(x_all) == 0:
        raise ValueError("No valid labels found in dataset")

    print(f"Using {len(x_all)} samples with valid labels")

    x_train, x_temp, y_train, y_temp = train_test_split(x_all, y_all, test_size=0.3, random_state=42, stratify=y_all)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    best_model, best_model_name = train_and_select_model(x_train, y_train, x_val, y_val)

    print("\n--- TEST PHASE ---")
    y_test_pred = best_model.predict(x_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"Best Model: {best_model_name}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Save results
    results = {
        'model_name': best_model_name,
        'test_accuracy': test_acc,
        'confusion_matrix': cm.tolist()
    }
    
    # Create result directory if it doesn't exist
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    
    # Save results to CSV (simplified version)
    results_df = pd.DataFrame([{
        'model_name': best_model_name,
        'test_accuracy': test_acc
    }])
    results_df.to_csv(result_path, index=False)
    print(f"Results saved to {result_path}")

if __name__ == "__main__":
    # Configure paths - adjust these to your specific folders
    original_img_dir = r"C:\Users\Erik\OneDrive - ITU\Escritorio\2 semester\Semester project\Introduction to final project\matched_pairs\images"
    mask_img_dir = r"C:\Users\Erik\OneDrive - ITU\Escritorio\2 semester\Semester project\Introduction to final project\matched_pairs\masks"
    labels_csv_path = "./dataset.csv"
    output_csv_path = "./dataset_baseline_features.csv"
    result_path = "./result/result_baseline.csv"
    
    try:
        main(original_img_dir, mask_img_dir, labels_csv_path, output_csv_path, result_path, recreate_features=True)
    except Exception as e:
        print(f"Error running main: {e}")
        sys.exit(1)
