import sys
import os
from os.path import join, exists
import numpy as np
# import time # Not strictly used, can be removed
# from tqdm import tqdm # tqdm is used in create_feature_dataset
# from collections import defaultdict # Not strictly used, can be removed
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold # Added StratifiedKFold
from sklearn.ensemble import RandomForestClassifier # Added RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import shutil
from tqdm import tqdm # Ensure tqdm is imported if used in create_feature_dataset

# Import custom modules
try:
    from util.feature_A import extract_asymmetry_features
    from util.feature_B import extract_border_features_from_folder, calculate_border_score
    from util.feature_C import extract_feature_C
    from util.contrast_feature import extract_feature_contrast
    from util.blue_veil import extract_feature_BV
    from util.hair_removal_feature import remove_and_save_hairs
    # from models_evaluation import train_and_select_model # Commented out
except ImportError as e:
    print(f"Error: Could not import custom feature/model modules: {e}")
    print("Please ensure all feature modules (feature_A.py, feature_B.py, feature_C.py, contrast_feature.py, blue_veil.py, hair_removal_feature.py) are in the 'util' directory.")
    # print("Ensure models_evaluation.py is in the same directory or Python path if using train_and_select_model.")
    sys.exit(1)

def create_feature_dataset(original_img_dir, mask_img_dir, output_csv_path, labels_csv=None, recreate_features=False):
    print("Starting EXTENDED feature extraction process (with Contrast, BV, Hair Removal)...")

    if not exists(original_img_dir):
        raise FileNotFoundError(f"Original image directory not found: {original_img_dir}")

    base_output_dir = os.path.dirname(output_csv_path)
    # Ensure a unique name for hair removed images for this extended pipeline
    hair_removed_img_dir_path = os.path.join(base_output_dir, "hair_removed_images_extended_pipeline")

    if recreate_features:
        print(f"Recreate features is True, removing existing hair-removed images directory: {hair_removed_img_dir_path}")
        if exists(hair_removed_img_dir_path):
            shutil.rmtree(hair_removed_img_dir_path)
        # else: # No need for else if exist_ok=True is used for makedirs
        #     print(f"Hair-removed images directory '{hair_removed_img_dir_path}' does not exist, creating new.")
    os.makedirs(hair_removed_img_dir_path, exist_ok=True)

    print(f"\nHair removal processing: Original images from '{original_img_dir}'")
    print(f"Processed images will be stored in: '{hair_removed_img_dir_path}'")

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    try:
        original_image_files = [f for f in os.listdir(original_img_dir) if f.lower().endswith(valid_extensions)]
        print(f"Found {len(original_image_files)} images in original directory for hair processing.")
    except Exception as e:
        print(f"Error listing files in original_img_dir '{original_img_dir}': {e}")
        return pd.DataFrame()

    hair_params = {
        "blackhat_kernel_size": (15, 15), "threshold_value": 18,
        "dilation_kernel_size": (3, 3), "dilation_iterations": 2,
        "inpaint_radius": 5, "min_hair_contours_to_process": 3, # From your original extended script
        "min_contour_area": 15 # From your original extended script
    }

    processed_files_in_loop = 0
    inpainted_count = 0
    copied_original_count = 0
    hair_removal_error_count = 0
    hair_ratios_data = [] # Renamed for clarity

    for filename in tqdm(original_image_files, desc="Performing Hair Removal"):
        original_image_path = os.path.join(original_img_dir, filename)
        target_path_in_hair_removed_dir = os.path.join(hair_removed_img_dir_path, filename)

        current_hair_ratio_val = 0.0 # Store the 0-1 ratio

        # This logic is for skipping already processed images if recreate_features is False.
        # If recreate_features is True (as it is in your main call), this block is skipped.
        if not recreate_features and os.path.exists(target_path_in_hair_removed_dir):
            # If we skip, we don't know the hair ratio unless we re-calculate or store it.
            # For simplicity with recreate_features=True, this branch is less critical.
            # If recreate_features=False becomes common, this needs a way to get the stored ratio.
            print(f"Skipping hair removal for {filename}, already exists and recreate_features=False. Hair ratio will be 0.")
            hair_ratios_data.append({'filename': filename, 'hair_ratio': current_hair_ratio_val})
            processed_files_in_loop +=1
            continue

        try:
            hair_ratio_from_func, saved_img_path, msg = remove_and_save_hairs(
                image_path=original_image_path,
                output_dir=hair_removed_img_dir_path,
                **hair_params
            )
            current_hair_ratio_val = hair_ratio_from_func # This is the 0-1 ratio

            if "hairs removed" in msg and saved_img_path and os.path.exists(saved_img_path):
                inpainted_count += 1
            # This condition handles cases where no significant hairs were found and the original was copied by remove_and_save_hairs
            elif "No significant hairs found" in msg and saved_img_path and os.path.exists(saved_img_path):
                 copied_original_count += 1 # remove_and_save_hairs already copied it
            # Fallback: if remove_and_save_hairs didn't create the file for some reason (e.g., error before saving, or unexpected message)
            elif not os.path.exists(target_path_in_hair_removed_dir):
                print(f"\nWarning: Hair removal for {filename} - message: '{msg}'. Output file '{target_path_in_hair_removed_dir}' not found. Copying original.")
                shutil.copy2(original_image_path, target_path_in_hair_removed_dir)
                copied_original_count += 1
            else: # File exists, likely copied or inpainted successfully.
                pass # Counted by inpainted_count or copied_original_count if message matched

            processed_files_in_loop +=1

        except FileNotFoundError:
             print(f"\nError: Original image {filename} not found at {original_image_path} during hair removal loop.")
             hair_removal_error_count +=1
        except Exception as e_hair:
            print(f"\nError during hair removal for {filename}: {e_hair}. Attempting to copy original.")
            hair_removal_error_count += 1
            try:
                if not os.path.exists(target_path_in_hair_removed_dir):
                    shutil.copy2(original_image_path, target_path_in_hair_removed_dir)
                copied_original_count += 1 # Count as copied even if it was an error recovery
                # processed_files_in_loop += 1 #  Avoid double counting if error happens after initial processing steps
            except Exception as e_copy:
                print(f"\nFailed to copy original {filename} after hair removal error: {e_copy}")

        hair_ratios_data.append({'filename': filename, 'hair_ratio': current_hair_ratio_val})

    print(f"\nHair removal loop summary: Files processed/checked in loop: {processed_files_in_loop}, Actually inpainted: {inpainted_count}, Originals copied (by script or func): {copied_original_count}, Errors during processing: {hair_removal_error_count}")

    df_hair_ratios = pd.DataFrame(hair_ratios_data) # Renamed
    print(f"Hair ratio features extracted. Shape: {df_hair_ratios.shape}")
    if not df_hair_ratios.empty:
        print(df_hair_ratios.head())


    feature_processing_dir = hair_removed_img_dir_path # Features are extracted from hair-removed images
    try:
        num_images_for_features = len(os.listdir(feature_processing_dir))
        if num_images_for_features == 0:
            print(f"CRITICAL: No images found in '{feature_processing_dir}' for feature extraction. Exiting.")
            return pd.DataFrame()
        print(f"Total images in '{feature_processing_dir}' for subsequent feature extraction: {num_images_for_features}")
    except FileNotFoundError:
        print(f"CRITICAL: Feature processing directory '{feature_processing_dir}' not found. This usually means hair removal failed for all images or the path is incorrect.")
        return pd.DataFrame()


    dfs = {}
    feature_extractors = {
        "A": extract_asymmetry_features,
        "B_raw": extract_border_features_from_folder,
        "C": extract_feature_C,
        "Contrast": extract_feature_contrast,
        "BV": extract_feature_BV
    }

    dfs["Hair_Ratio"] = df_hair_ratios # Using the DataFrame with hair ratios

    for name, func in feature_extractors.items():
        print(f"\nExtracting {name} features from: {feature_processing_dir}")
        try:
            if name == "B_raw":
                df_temp_raw = func(folder_path=feature_processing_dir, output_csv=None, visualize=False)
                if not df_temp_raw.empty:
                    dfs["B"] = calculate_border_score(df_temp_raw)
                    # Ensure 'filename' exists before attempting to drop other columns if B is calculated
                    if 'filename' not in dfs["B"].columns and not dfs["B"].empty:
                         print(f"CRITICAL WARNING: DataFrame from calculate_border_score (for B) is missing 'filename'. Shape: {dfs['B'].shape}")
                    cols_to_drop_from_B = ['sobel_mean_safe', 'avg_contour_perimeter_safe', 'laplacian_mean_safe', 'avg_contour_area_safe']
                    dfs["B"] = dfs["B"].drop(columns=[col for col in cols_to_drop_from_B if col in dfs["B"].columns], errors='ignore')
                    print(f"Border features (B) processed. Shape: {dfs['B'].shape}")
                else:
                    print(f"Warning: {name} feature extraction returned an empty DataFrame.")
                    dfs["B"] = pd.DataFrame(columns=['filename']) # Ensure it has a filename column for merge logic
            else:
                # Make sure all feature extractors return a DataFrame with 'filename'
                temp_df = func(folder_path=feature_processing_dir, output_csv=None, visualize=False)
                if not temp_df.empty and 'filename' not in temp_df.columns:
                    print(f"CRITICAL WARNING: {name} feature extraction returned DataFrame missing 'filename' column. Shape: {temp_df.shape}")
                    dfs[name] = pd.DataFrame(columns=['filename'])
                elif temp_df.empty:
                    print(f"Warning: {name} feature extraction returned an empty DataFrame.")
                    dfs[name] = pd.DataFrame(columns=['filename'])
                else:
                    dfs[name] = temp_df
                    print(f"{name} features extracted. Shape: {dfs[name].shape}")
        except Exception as e:
            print(f"Error during {name} feature extraction: {e}")
            dfs[name] = pd.DataFrame(columns=['filename']) # Ensure filename column for merge

    metadata_df = None
    if labels_csv and exists(labels_csv):
        print(f"\nLoading metadata from {labels_csv}")
        try:
            raw_metadata_df = pd.read_csv(labels_csv)
            if 'img_id' in raw_metadata_df.columns:
                raw_metadata_df = raw_metadata_df.rename(columns={'img_id': 'filename'})
            label_column_name = 'diagnostic' # As in your original script
            if 'filename' not in raw_metadata_df.columns or label_column_name not in raw_metadata_df.columns:
                print(f"ERROR: Metadata CSV must contain 'filename' (or 'img_id') and '{label_column_name}'. Metadata will not be used.")
                metadata_df = None
            else:
                if label_column_name != 'real_label': # Standardize to 'real_label'
                    raw_metadata_df.rename(columns={label_column_name: 'real_label'}, inplace=True)
                cancer_diagnoses = ["BCC", "SCC", "MEL"]
                raw_metadata_df['binary_target'] = raw_metadata_df['real_label'].apply(lambda x: 1 if x in cancer_diagnoses else 0)
                cols_to_keep_from_metadata = ['filename', 'real_label', 'binary_target']
                # Ensure all kept columns actually exist after potential renames
                actual_cols_to_keep = [col for col in cols_to_keep_from_metadata if col in raw_metadata_df.columns]
                metadata_df = raw_metadata_df[actual_cols_to_keep].copy()
                print(f"Metadata selected. Shape: {metadata_df.shape[0]} entries. Columns: {metadata_df.columns.tolist()}")
                if 'filename' not in metadata_df.columns: # Final check
                    print("ERROR: 'filename' column missing in metadata_df after selection. Metadata will not be used for merging.")
                    metadata_df = None
        except Exception as e:
            print(f"Error loading or processing metadata: {e}")
            metadata_df = None
    else:
        print("\nNo metadata file provided or found. Proceeding without metadata.")

    print("\nMerging feature DataFrames...")
    dataframes_to_merge = []
    if metadata_df is not None and not metadata_df.empty and 'filename' in metadata_df.columns:
        dataframes_to_merge.append(metadata_df)
        print(f"metadata_df added for merge. Shape: {metadata_df.shape}")
    else:
        print("metadata_df is None, empty, or missing 'filename'. Not added to merge.")


    for name, df in dfs.items():
        if name == "B_raw": continue # We use dfs["B"] which is processed
        if df is not None and not df.empty and 'filename' in df.columns:
            # Deduplicate filenames within each feature DataFrame before merge, if necessary
            if df['filename'].duplicated().any():
                print(f"Warning: Duplicate filenames found in df_{name}. Keeping first occurrence.")
                df = df.drop_duplicates(subset=['filename'], keep='first')
                dfs[name] = df # Update the dictionary entry
            dataframes_to_merge.append(df)
            print(f"df_{name} added for merge. Shape: {df.shape}")
        else:
             print(f"df_{name} is None, empty, or missing 'filename'. Not added to merge.")

    if not dataframes_to_merge:
        print("No DataFrames to merge (empty list). Exiting feature creation.")
        return pd.DataFrame()
    if len(dataframes_to_merge) < 2 and metadata_df is None:
        print("Not enough DataFrames for a meaningful merge (need at least one feature set, or metadata + features).")
        if dataframes_to_merge: return dataframes_to_merge[0] # Return the single DF if it exists
        return pd.DataFrame()


    final_df = dataframes_to_merge[0]
    print(f"Base DataFrame for merge: Columns: {final_df.columns.tolist()}, Shape: {final_df.shape}")
    if final_df['filename'].duplicated().any(): # Check base for duplicates
        print(f"Warning: Duplicate filenames found in base DataFrame for merge. Keeping first.")
        final_df = final_df.drop_duplicates(subset=['filename'], keep='first')


    for i, df_to_merge in enumerate(dataframes_to_merge[1:]):
        df_name_for_log = "Unknown_DF"
        for key_name, val_df in dfs.items(): # Find name of df_to_merge
            if val_df is df_to_merge:
                df_name_for_log = key_name
                break

        print(f"Merging with DataFrame for feature '{df_name_for_log}' (shape {df_to_merge.shape})")
        if 'filename' not in df_to_merge.columns:
            print(f"CRITICAL WARNING: DataFrame for '{df_name_for_log}' is missing 'filename' column. Skipping merge.")
            continue
        if df_to_merge['filename'].duplicated().any(): # Should have been handled, but double check
            print(f"Warning: Duplicate filenames found in df_to_merge '{df_name_for_log}' just before merge. Keeping first.")
            df_to_merge = df_to_merge.drop_duplicates(subset=['filename'], keep='first')


        final_df = pd.merge(final_df, df_to_merge, on='filename', how='inner')
        if final_df.empty:
            print(f"CRITICAL WARNING: DataFrame empty after merging with {df_name_for_log}. This often means no common filenames or issues with filename consistency (e.g. '.jpg').");
            break
        else:
            print(f"Shape after merging with {df_name_for_log}: {final_df.shape}")

    if final_df.empty:
        print("Resulting merged DataFrame is empty.")
    else:
        print(f"Merged DataFrame final shape: {final_df.shape}. Columns: {final_df.columns.tolist()}")
        if final_df['filename'].duplicated().any():
             print(f"WARNING: Duplicates found in 'filename' column of final_df AFTER all merges. Dropping duplicates, keeping first.")
             final_df = final_df.drop_duplicates(subset=['filename'], keep='first').reset_index(drop=True)
             print(f"Shape after final duplicate drop: {final_df.shape}")


    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    final_df.to_csv(output_csv_path, index=False)
    print(f"\nMerged feature dataset saved to {output_csv_path}")

    if not final_df.empty:
        expected_label_cols = ['real_label', 'binary_target']
        missing = [col for col in expected_label_cols if col not in final_df.columns]
        if missing: print(f"WARNING: Final DataFrame MISSING label columns: {missing}")
        else: print(f"SUCCESS: 'real_label' and 'binary_target' columns present in final DataFrame.")
    return final_df


def main(original_img_dir, mask_img_dir, labels_csv_path, output_csv_path, result_path, recreate_features=False):
    print("\n--- FEATURE DATASET CREATION (EXTENDED FEATURES - Contrast, BV, Hair Removal) ---\n")

    if not original_img_dir or not output_csv_path:
        raise ValueError("original_img_dir and output_csv_path must be provided.")

    data_df = None
    # For CV, always recreate features ensures consistency if parameters change
    # However, for very large datasets, you might want to control this.
    # Your previous main call had recreate_features=True implicitly, so keeping that behavior.
    if recreate_features or not exists(output_csv_path):
        print(f"Creating new EXTENDED feature dataset at {output_csv_path} (recreate_features={recreate_features})")
        data_df = create_feature_dataset(original_img_dir, mask_img_dir, output_csv_path,
                                         labels_csv=labels_csv_path, recreate_features=recreate_features)
    else:
        print(f"Loading existing EXTENDED feature dataset from {output_csv_path}")
        try:
            data_df = pd.read_csv(output_csv_path)
        except Exception as e:
            print(f"Error loading existing dataset: {e}. Will attempt to recreate.")
            data_df = create_feature_dataset(original_img_dir, mask_img_dir, output_csv_path,
                                             labels_csv=labels_csv_path, recreate_features=True) # Force recreate on load error


    if data_df is None or data_df.empty:
        print("Failed to create or load the EXTENDED feature dataset. Exiting.")
        return

    print("\n--- Merged EXTENDED Dataset Information ---")
    data_df.info(verbose=True, show_counts=True)
    print("\nFirst 5 rows of the merged EXTENDED dataset:")
    print(data_df.head())

    if 'filename' not in data_df.columns:
        print("CRITICAL ERROR: 'filename' column is missing in the final DataFrame!")
        return
    if data_df['filename'].isnull().any():
        print(f"Warning: {data_df['filename'].isnull().sum()} 'filename' entries are NaN. This usually indicates an issue in merging.")
    if data_df['filename'].duplicated().any():
        print(f"Warning: {data_df['filename'].duplicated().sum()} duplicate filenames found. Consolidating by keeping the first occurrence.")
        data_df = data_df.drop_duplicates(subset=['filename'], keep='first').reset_index(drop=True)

    # --- DATA PREPARATION FOR MODELING ---
    print("\n--- DATA PREPARATION FOR MODELING (EXTENDED FEATURES) ---")

    if 'binary_target' not in data_df.columns:
        source_label_col = next((col for col in ['real_label', 'diagnostic'] if col in data_df.columns), None)
        if source_label_col:
            print(f"Creating 'binary_target' from '{source_label_col}'.")
            cancer_diagnoses_map = ["BCC", "SCC", "MEL"]
            data_df['binary_target'] = data_df[source_label_col].apply(lambda x: 1 if x in cancer_diagnoses_map else 0)
            if source_label_col == 'diagnostic' and 'real_label' not in data_df.columns: # Ensure 'real_label' exists
                data_df.rename(columns={'diagnostic': 'real_label'}, inplace=True)
        else:
            print("CRITICAL ERROR: Target label column ('binary_target', 'real_label', or 'diagnostic') not found. Cannot proceed."); return

    data_df.dropna(subset=['binary_target'], inplace=True)
    if data_df.empty:
        print("CRITICAL ERROR: Dataset empty after dropping NaNs in 'binary_target'."); return

    data_df['label'] = data_df['binary_target'].astype(int)
    class_names_for_report = ['non-cancer', 'cancer']
    print(f"\nLabel distribution:\n{data_df['label'].value_counts(normalize=True)}")

    if 'real_label' not in data_df.columns: # Fallback for real_label
        data_df['real_label'] = data_df['label'].map({0: 'derived_non-cancer', 1: 'derived_cancer'})

    if 'c_dominant_channel' in data_df.columns: # One-hot encode if color features were added
        print("\nOne-hot encoding 'c_dominant_channel'...")
        try:
            data_df = pd.get_dummies(data_df, columns=['c_dominant_channel'], prefix='c_dom_channel', dummy_na=False)
        except Exception as e_ohe:
            print(f"Error during one-hot encoding 'c_dominant_channel': {e_ohe}")

    potential_non_feature_cols = ['filename', 'real_label', 'binary_target', 'label', 'diagnostic',
                                  'patient_id', 'lesion_id', 'smoke', 'drink', 'background_father',
                                  'background_mother', 'age', 'pesticide', 'gender',
                                  'skin_cancer_history', 'cancer_history', 'has_piped_water',
                                  'has_sewage_system', 'fitspatrick', 'region', 'diameter_1',
                                  'diameter_2', 'itch', 'grew', 'hurt', 'changed', 'bleed',
                                  'elevation', 'biopsed']
    feature_columns = [col for col in data_df.columns if col not in potential_non_feature_cols]

    if not feature_columns:
        print("CRITICAL ERROR: No feature columns identified for extended dataset."); return
    print(f"\nUsing {len(feature_columns)} EXTENDED feature columns. First 10: {feature_columns[:10]}...")

    x_all = data_df[feature_columns].copy()
    y_all = data_df["label"].copy()
    current_filenames = data_df['filename'].copy()

    print("\nConverting EXTENDED features to numeric, handling NaNs/Infs...")
    for feat in feature_columns:
        x_all[feat] = pd.to_numeric(x_all[feat], errors='coerce')

    all_nan_cols = x_all.columns[x_all.isnull().all()].tolist()
    if all_nan_cols:
        print(f"Warning: Columns dropped due to all NaN: {all_nan_cols}")
        x_all = x_all.drop(columns=all_nan_cols)
        feature_columns = [col for col in feature_columns if col not in all_nan_cols] # Update feature_columns list
        if not feature_columns:
            print("CRITICAL ERROR: All EXTENDED feature columns dropped after NaN conversion."); return

    x_all.replace([np.inf, -np.inf], np.nan, inplace=True)
    imputer = SimpleImputer(strategy='mean')
    if x_all.empty:
        print("CRITICAL ERROR: x_all (features) empty before imputation for EXTENDED dataset."); return
    x_all_imputed = imputer.fit_transform(x_all)
    x_all = pd.DataFrame(x_all_imputed, columns=x_all.columns, index=x_all.index)

    if len(x_all) == 0 or y_all.nunique() < 2:
        print(f"Skipping model training for EXTENDED dataset: Samples: {len(x_all)}, Unique Labels: {y_all.nunique()}"); return

    # --- K-FOLD CROSS-VALIDATION with RANDOM FOREST ONLY (EXTENDED FEATURES) ---
    N_SPLITS = 5
    min_class_count = y_all.value_counts().min()
    if N_SPLITS > min_class_count:
        print(f"Warning: N_SPLITS ({N_SPLITS}) for EXTENDED set is greater than the number of samples in the smallest class ({min_class_count}).")
        print(f"Reducing N_SPLITS to {min_class_count} to allow stratified splitting.")
        N_SPLITS = min_class_count
        if N_SPLITS < 2:
             print("Smallest class has less than 2 samples. Cannot perform K-Fold CV for EXTENDED set. Exiting model training.")
             return

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_results_list = []
    all_test_predictions_df = pd.DataFrame()

    print(f"\n--- {N_SPLITS}-FOLD CROSS-VALIDATION (Random Forest Only on EXTENDED Features) ---")

    for fold_num, (dev_indices, test_indices) in enumerate(skf.split(x_all, y_all)):
        print(f"\n--- FOLD {fold_num + 1}/{N_SPLITS} (EXTENDED FEATURES) ---")

        x_dev_fold = x_all.iloc[dev_indices]
        y_dev_fold = y_all.iloc[dev_indices]
        x_test_fold = x_all.iloc[test_indices]
        y_test_fold = y_all.iloc[test_indices]
        filenames_test_fold = current_filenames.iloc[test_indices]

        try:
            x_train_inner, x_val_inner, y_train_inner, y_val_inner = train_test_split(
                x_dev_fold, y_dev_fold, test_size=0.25, random_state=42, stratify=y_dev_fold
            )
        except ValueError as e_split_inner:
            print(f"Error during inner data splitting for EXTENDED fold {fold_num + 1}: {e_split_inner}.")
            print(f"Class distribution in y_dev_fold: \n{y_dev_fold.value_counts()}")
            continue

        print(f"EXTENDED Fold {fold_num + 1}: Train_inner size: {len(x_train_inner)}, Val_inner size: {len(x_val_inner)}, Test_fold size: {len(x_test_fold)}")

        if x_train_inner.empty or x_val_inner.empty or y_train_inner.nunique() < 2:
            print(f"EXTENDED Fold {fold_num + 1}: Training_inner or validation_inner set is empty or has insufficient classes. Skipping this fold.")
            continue

        try:
            model_fold = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            model_fold.fit(x_train_inner, y_train_inner)
            model_name_fold = "RandomForestClassifier_Extended"

            y_val_inner_pred = model_fold.predict(x_val_inner)
            val_acc_inner_fold = accuracy_score(y_val_inner, y_val_inner_pred)
            print(f"EXTENDED Fold {fold_num + 1} - Inner Validation Accuracy (RF): {val_acc_inner_fold:.4f}")

            print(f"\nEXTENDED Fold {fold_num + 1} - Test Phase on test_fold data...")
            y_test_pred_fold = model_fold.predict(x_test_fold)
            y_test_pred_proba_fold = model_fold.predict_proba(x_test_fold) # RF has predict_proba

            test_acc_fold = accuracy_score(y_test_fold, y_test_pred_fold)
            cm_labels_binary_fold = [0, 1]
            cls_report_dict_fold = classification_report(
                y_test_fold, y_test_pred_fold, labels=cm_labels_binary_fold,
                target_names=class_names_for_report, output_dict=True, zero_division=0
            )
            cls_report_str_fold = classification_report(
                y_test_fold, y_test_pred_fold, labels=cm_labels_binary_fold,
                target_names=class_names_for_report, zero_division=0
            )

            print(f"EXTENDED Fold {fold_num + 1} - Model: {model_name_fold}")
            print(f"EXTENDED Fold {fold_num + 1} - Test Accuracy on test_fold: {test_acc_fold:.4f}")
            print(f"EXTENDED Fold {fold_num + 1} - Confusion Matrix (test_fold):\n{confusion_matrix(y_test_fold, y_test_pred_fold, labels=cm_labels_binary_fold)}")
            print(f"EXTENDED Fold {fold_num + 1} - Classification Report (test_fold):\n{cls_report_str_fold}")

            fold_summary = {
                'fold': fold_num + 1,
                'model_name': model_name_fold,
                'validation_accuracy_inner': val_acc_inner_fold,
                'test_accuracy_fold': test_acc_fold,
                'num_training_samples_inner': len(x_train_inner),
                'num_validation_samples_inner': len(x_val_inner),
                'num_test_samples_fold': len(x_test_fold),
                'num_features_used': len(feature_columns)
            }
            for class_label_report_fold in class_names_for_report:
                if class_label_report_fold in cls_report_dict_fold:
                    for metric in ['precision', 'recall', 'f1-score', 'support']:
                        fold_summary[f'{class_label_report_fold}_{metric}_test_fold'] = cls_report_dict_fold[class_label_report_fold][metric]
            for avg_type_fold in ['macro avg', 'weighted avg']:
                if avg_type_fold in cls_report_dict_fold:
                    for metric in ['precision', 'recall', 'f1-score']:
                        fold_summary[f'{avg_type_fold.replace(" ", "_")}_{metric}_test_fold'] = cls_report_dict_fold[avg_type_fold][metric]
            fold_results_list.append(fold_summary)

            current_fold_predictions_df = pd.DataFrame({
                'fold': fold_num + 1,
                'filename': filenames_test_fold.reset_index(drop=True).values,
                'true_label_encoded': y_test_fold.reset_index(drop=True).values,
                'predicted_label_encoded': y_test_pred_fold,
                'true_label_text': y_test_fold.reset_index(drop=True).map({0: 'non-cancer', 1: 'cancer'}).values,
                'predicted_label_text': pd.Series(y_test_pred_fold).map({0: 'non-cancer', 1: 'cancer'}).values
            })
            if y_test_pred_proba_fold is not None and y_test_pred_proba_fold.shape[0] == len(filenames_test_fold):
                if y_test_pred_proba_fold.shape[1] >= len(class_names_for_report):
                    current_fold_predictions_df[f'proba_{class_names_for_report[0]}'] = y_test_pred_proba_fold[:, 0]
                    current_fold_predictions_df[f'proba_{class_names_for_report[1]}'] = y_test_pred_proba_fold[:, 1]
            else: # Fallback if proba array is problematic
                current_fold_predictions_df[f'proba_{class_names_for_report[0]}'] = np.nan
                current_fold_predictions_df[f'proba_{class_names_for_report[1]}'] = np.nan
            all_test_predictions_df = pd.concat([all_test_predictions_df, current_fold_predictions_df], ignore_index=True)

        except Exception as e_model_fold:
            print(f"Error during model training/evaluation for EXTENDED fold {fold_num + 1}: {e_model_fold}")
            import traceback; traceback.print_exc()

    # --- AGGREGATE RESULTS FROM K-FOLD CV (EXTENDED FEATURES) ---
    if not fold_results_list:
        print("No folds were successfully processed for EXTENDED features. Cannot generate CV summary. Exiting.")
        return

    print("\n\n--- K-FOLD CROSS-VALIDATION SUMMARY (Random Forest Only on EXTENDED Features) ---")
    cv_summary_df = pd.DataFrame(fold_results_list)

    avg_metrics_summary = {
        'model_type_fixed': "RandomForestClassifier_Extended",
        'num_folds_processed': len(cv_summary_df),
        'num_features_used': cv_summary_df['num_features_used'].iloc[0] if not cv_summary_df.empty else 0
    }

    metrics_to_process = ['test_accuracy_fold', 'validation_accuracy_inner']
    for cl_label in class_names_for_report:
        for metric in ['precision', 'recall', 'f1-score', 'support']:
            metrics_to_process.append(f'{cl_label}_{metric}_test_fold')
    for avg_type in ['macro avg', 'weighted avg']:
        for metric in ['precision', 'recall', 'f1-score']:
            metrics_to_process.append(f'{avg_type.replace(" ", "_")}_{metric}_test_fold')

    for metric_col in metrics_to_process:
        if metric_col in cv_summary_df.columns:
            base_metric_name = metric_col.replace('_test_fold', '').replace('_inner', '_inner_val')
            if 'support' in metric_col:
                avg_metrics_summary[f'total_{base_metric_name}'] = cv_summary_df[metric_col].sum()
                avg_metrics_summary[f'mean_{base_metric_name}'] = cv_summary_df[metric_col].mean()
                avg_metrics_summary[f'std_{base_metric_name}'] = cv_summary_df[metric_col].std()
                print(f"Total {base_metric_name}: {avg_metrics_summary[f'total_{base_metric_name}']}")
                print(f"Mean {base_metric_name} per fold: {avg_metrics_summary[f'mean_{base_metric_name}']:.4f} +/- {avg_metrics_summary[f'std_{base_metric_name}']:.4f}")
            else:
                avg_metrics_summary[f'mean_{base_metric_name}'] = cv_summary_df[metric_col].mean()
                avg_metrics_summary[f'std_{base_metric_name}'] = cv_summary_df[metric_col].std()
                print(f"Mean {base_metric_name}: {avg_metrics_summary[f'mean_{base_metric_name}']:.4f} +/- {avg_metrics_summary[f'std_{base_metric_name}']:.4f}")

    print(f"\nOverall Average Inner Validation Accuracy (EXTENDED) across {N_SPLITS} folds: {avg_metrics_summary.get('mean_validation_accuracy_inner_val', np.nan):.4f} +/- {avg_metrics_summary.get('std_validation_accuracy_inner_val', np.nan):.4f}")
    print(f"Overall Average Test Accuracy (EXTENDED) across {N_SPLITS} folds: {avg_metrics_summary.get('mean_test_accuracy', np.nan):.4f} +/- {avg_metrics_summary.get('std_test_accuracy', np.nan):.4f}")

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    fold_details_csv_path = os.path.join(os.path.dirname(result_path), f"{os.path.splitext(os.path.basename(result_path))[0]}_CV_fold_details.csv")
    cv_summary_df.to_csv(fold_details_csv_path, index=False)
    print(f"Detailed K-Fold CV results per fold (EXTENDED) saved to {fold_details_csv_path}")

    all_predictions_csv_path = os.path.join(os.path.dirname(result_path), f"{os.path.splitext(os.path.basename(result_path))[0]}_CV_all_predictions.csv")
    all_test_predictions_df.to_csv(all_predictions_csv_path, index=False)
    print(f"All test predictions from K-Fold CV (EXTENDED) saved to {all_predictions_csv_path}")

    aggregated_summary_df = pd.DataFrame([avg_metrics_summary])
    aggregated_summary_df.to_csv(result_path, index=False)
    print(f"Aggregated K-Fold CV summary report (EXTENDED) saved to {result_path}")


if __name__ == "__main__":
    original_img_dir = r"C:\Users\misog\SCHOOL\2nd semester\Projects in Data Science\matched_pairs\images"
    mask_img_dir = r"C:\Users\misog\SCHOOL\2nd semester\Projects in Data Science\masks" # mask_img_dir is not used by create_feature_dataset in this extended script
    labels_csv_path = r"C:\Users\misog\SCHOOL\2nd semester\Projects in Data Science\final project\2025-FYP-Final\data\filtered_metadata_img_id_first.csv"

    # Updated output directory and filenames for clarity
    output_feature_csv_dir = "./result_extended_CV_RF_only"
    os.makedirs(output_feature_csv_dir, exist_ok=True)

    merged_csv_filename = "dataset_extended_CV_RF_only.csv"
    output_csv_path = os.path.join(output_feature_csv_dir, merged_csv_filename)

    model_result_filename = "model_evaluation_extended_CV_RF_only_summary.csv"
    result_path = os.path.join(output_feature_csv_dir, model_result_filename)

    try:
        # recreate_features=True is important for consistent CV runs if feature extraction parameters change
        main(original_img_dir, mask_img_dir, labels_csv_path, output_csv_path, result_path, recreate_features=True)
    except Exception as e:
        print(f"Error running main script for extended features with CV and RF: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)