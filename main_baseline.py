import sys
import os
from os.path import join, exists
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from util.img_util import readImageFile
from util.feature_A import extract_asymmetry_features
from util.feature_B import extract_border_features
from util.feature_C import extract_feature_C
from models_evaluation import train_and_select_model

def create_feature_dataset(original_img_dir, mask_img_dir, output_csv_path, labels_csv=None):
    print("Starting feature extraction process...")

    if labels_csv and exists(labels_csv):
        print(f"Loading labels from {labels_csv}")
        labels_df = pd.read_csv(labels_csv)
        label_dict = dict(zip(labels_df['filename'], labels_df['label']))
    else:
        label_dict = {}

    img_files = [f for f in os.listdir(original_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(img_files)} images to process")

    all_features = []

    for i, img_file in enumerate(img_files):
        if i % 10 == 0:
            print(f"Processing image {i+1}/{len(img_files)}: {img_file}")

        orig_img_path = join(original_img_dir, img_file)
        mask_img_path = join(mask_img_dir, img_file)

        orig_img = readImageFile(orig_img_path)
        mask_img = readImageFile(mask_img_path) if exists(mask_img_path) else None

        features = {"filename": img_file}

        # Add label if available
        if label_dict:
            features["label"] = label_dict.get(img_file, -1)

        # Update features from each module
        features.update(extract_asymmetry_features(orig_img))
        features.update(extract_border_features(orig_img))
        features.update(extract_feature_C(orig_img, mask_img))

        all_features.append(features)

    df = pd.DataFrame(all_features)
    df.to_csv(output_csv_path, index=False)
    print(f"Feature dataset saved to {output_csv_path}")
    print(f"Dataset shape: {df.shape}, with features: {[col for col in df.columns if col.startswith('feat_')]}")
    return df

def main(original_img_dir, mask_img_dir, labels_csv_path, output_csv_path, result_path, recreate_features=False):
    print("\n--- BASELINE APPROACH ---\n")

    if recreate_features or not exists(output_csv_path):
        print(f"Creating new feature dataset at {output_csv_path}")
        data_df = create_feature_dataset(original_img_dir, mask_img_dir, output_csv_path, labels_csv=labels_csv_path)
    else:
        print(f"Loading existing feature dataset from {output_csv_path}")
        data_df = pd.read_csv(output_csv_path)

    baseline_feats = [col for col in data_df.columns if col.startswith("feat_")]
    print(f"Using {len(baseline_feats)} features: {baseline_feats}")

    x_all = data_df[baseline_feats]
    y_all = data_df["label"]

    x_train, x_temp, y_train, y_temp = train_test_split(x_all, y_all, test_size=0.3, random_state=42, stratify=y_all)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    best_model, best_model_name = train_and_select_model(x_train, y_train, x_val, y_val)

    print("\n--- TEST PHASE ---")
    y_test_pred = best_model.predict(x_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

if __name__ == "__main__":
<<<<<<< HEAD
    original_img_dir = "./data/"
    mask_img_dir = "./data_masks/"
    labels_csv_path = "./dataset.csv"
    output_csv_path = "./dataset_baseline_features.csv"
    result_path = "./result/result_baseline.csv"

=======
    # Configure paths - adjust these to your specific PC folders
    original_img_dir = ""  # Original wound images
    mask_img_dir = ""  # Masked/segmented images
    labels_csv_path = ""  # CSV with labels

    # Output files
    output_csv_path = ""  # Where to save extracted features
    result_path = ""  # Where to save results
    
    # Make sure result directory exists
>>>>>>> ef5d8b731413a74d3d5524c8e152fb79e024d5ea
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    main(original_img_dir, mask_img_dir, labels_csv_path, output_csv_path, result_path, recreate_features=True)