# In main_extended.py
import sys
import os
from os.path import join, exists
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import shutil # For copying files

# ... (rest of your imports remain the same) ...
try:
    from util.feature_A import extract_asymmetry_features
    from util.feature_B import extract_border_features_from_folder, calculate_border_score
    from util.feature_C import extract_feature_C
<<<<<<< HEAD
    from util.haralick_extended import extract_haralick_features
=======
    # Corrected import based on the actual function name in haralick_extended.py
    from util.haralick_extended import extract_feature_H as extract_haralick_features 
>>>>>>> ae55e045a4ecce7b83d752845eedecebc55cd85e
    from util.blue_veil import extract_feature_BV
    from util.hair_removal_feature import remove_and_save_hairs 
    from models_evaluation import train_and_select_model
except ImportError as e:
    print(f"Error: Could not import custom feature/model modules: {e}")
    # ... (rest of error message) ...
    sys.exit(1)


def create_feature_dataset(original_img_dir, mask_img_dir, output_csv_path, labels_csv=None, recreate_features=False):
    print("Starting EXTENDED feature extraction process...")

    if not exists(original_img_dir):
        raise FileNotFoundError(f"Original image directory not found: {original_img_dir}")

<<<<<<< HEAD
    # --- HAIR REMOVAL STEP ---
    hair_removed_img_dir_path = os.path.join(os.path.dirname(output_csv_path), "hair_removed_images")
=======
    base_output_dir = os.path.dirname(output_csv_path) 
    hair_removed_img_dir_path = os.path.join(base_output_dir, "hair_removed_images_extended") 
>>>>>>> ae55e045a4ecce7b83d752845eedecebc55cd85e
    
    perform_hair_removal_processing = True # Flag to control if we enter the hair removal loop

    if recreate_features and exists(hair_removed_img_dir_path):
        print(f"Recreate features is True, removing existing hair-removed images directory: {hair_removed_img_dir_path}")
        shutil.rmtree(hair_removed_img_dir_path)
    
<<<<<<< HEAD
    os.makedirs(hair_removed_img_dir_path, exist_ok=True) # Ensure directory exists

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    original_image_files = [f for f in os.listdir(original_img_dir) if f.lower().endswith(valid_extensions)]
    num_original_images = len(original_image_files)

    if not recreate_features and exists(hair_removed_img_dir_path):
        hair_removed_files = [f for f in os.listdir(hair_removed_img_dir_path) if f.lower().endswith(valid_extensions)]
        num_hair_removed_images = len(hair_removed_files)
        
        # Heuristic: if counts are very close, assume it's processed.
        # A more robust check would be to verify all original filenames exist in the hair_removed_dir.
        if num_hair_removed_images >= num_original_images * 0.98: # Allowing for a tiny discrepancy
            print(f"Hair-removed image directory '{hair_removed_img_dir_path}' exists and seems complete ({num_hair_removed_images} files vs {num_original_images} originals). Skipping hair removal processing.")
            perform_hair_removal_processing = False
        else:
            print(f"Hair-removed image directory '{hair_removed_img_dir_path}' exists but seems incomplete ({num_hair_removed_images} files vs {num_original_images} originals) or recreate_features is False but dir might be partially populated. Will process missing/new files.")
            # The loop below will still run, but individual file checks will skip already processed ones.
    
    if perform_hair_removal_processing:
        print(f"\nHair removal processing: Original images from '{original_img_dir}'")
        print(f"Processed images (hair-removed or original) will be in: '{hair_removed_img_dir_path}'")

        hair_params = {
            "blackhat_kernel_size": (15, 15), "threshold_value": 18,
            "dilation_kernel_size": (3, 3), "dilation_iterations": 2,
            "inpaint_radius": 5, "min_hair_contours_to_process": 3,
            "min_contour_area": 15
        }
        processed_hair_removal_count = 0
        copied_original_count = 0
        hair_removal_error_count = 0
        skipped_existing_count = 0

        for filename in original_image_files:
            original_image_path = os.path.join(original_img_dir, filename)
            target_path_in_hair_removed_dir = os.path.join(hair_removed_img_dir_path, filename)

            # If not recreating features AND the target file already exists, skip this specific file.
            # This check is inside the loop to handle cases where the folder exists but is incomplete.
            if not recreate_features and os.path.exists(target_path_in_hair_removed_dir):
                # print(f"Skipping hair removal for {filename}, already exists in target and not recreating features.")
                skipped_existing_count += 1
                continue
            
            try:
                hair_count, saved_img_path, msg = remove_and_save_hairs(
                    image_path=original_image_path,
                    output_dir=hair_removed_img_dir_path,
                    **hair_params
                )
                
                if "No significant hairs found" in msg or "original image skipped" in msg:
                    if not os.path.exists(target_path_in_hair_removed_dir):
                        shutil.copy2(original_image_path, target_path_in_hair_removed_dir)
                    copied_original_count += 1
                elif os.path.exists(target_path_in_hair_removed_dir):
                    processed_hair_removal_count += 1
                else:
                    shutil.copy2(original_image_path, target_path_in_hair_removed_dir)
                    copied_original_count += 1
            
            except FileNotFoundError:
                print(f"Error during hair removal for {filename}: Original image not found at {original_image_path}.")
                hair_removal_error_count += 1
            except Exception as e_hair:
                print(f"Error during hair removal for {filename}: {e_hair}. Attempting to copy original.")
                hair_removal_error_count += 1
                try:
                    if not os.path.exists(target_path_in_hair_removed_dir):
                        shutil.copy2(original_image_path, target_path_in_hair_removed_dir)
                    copied_original_count += 1
                except Exception as e_copy:
                    print(f"Failed to copy original {filename} after hair removal error: {e_copy}")
        
        print(f"Hair removal stage summary: Processed/inpainted this run: {processed_hair_removal_count}, Originals copied this run: {copied_original_count}, Skipped (already existed): {skipped_existing_count}, Errors this run: {hair_removal_error_count}")
    
    else: # This else corresponds to perform_hair_removal_processing = False
        print(f"Using existing hair-removed images from: '{hair_removed_img_dir_path}'")

    print(f"Total images available in hair-removed directory for feature extraction: {len(os.listdir(hair_removed_img_dir_path))}")
=======
    print(f"\nHair removal process: Original images from '{original_img_dir}'")
    print(f"Processed images will be in: '{hair_removed_img_dir_path}'")

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    hair_params = { 
        "blackhat_kernel_size": (15, 15), "threshold_value": 18,
        "dilation_kernel_size": (3, 3), "dilation_iterations": 2,
        "inpaint_radius": 5, "min_hair_contours_to_process": 3,
        "min_contour_area": 15
    }
    processed_images_count = 0
    inpainted_count = 0
    copied_original_count = 0
    hair_removal_error_count = 0

    original_image_files = [f for f in os.listdir(original_img_dir) if f.lower().endswith(valid_extensions)]
    print(f"Found {len(original_image_files)} images in original directory for hair processing.")

    for filename in original_image_files:
        original_image_path = os.path.join(original_img_dir, filename)
        target_path_in_hair_removed_dir = os.path.join(hair_removed_img_dir_path, filename)

        # If not recreating features and the processed file exists, skip.
        if not recreate_features and os.path.exists(target_path_in_hair_removed_dir):
            processed_images_count += 1
            # We don't know if it was inpainted or copied in a previous run without more checks,
            # but it exists, so we count it as "processed" for this stage.
            continue
            
        try:
            hair_count, saved_image_path, msg = remove_and_save_hairs(
                image_path=original_image_path,
                output_dir=hair_removed_img_dir_path, # function saves here if hairs are processed
                **hair_params
            )
            # saved_image_path will be the path to the inpainted image OR the original image name in output_dir
            # if the function decided to save/copy. The current remove_and_save_hairs only saves if inpainted.

            if "hairs removed" in msg and saved_image_path and os.path.exists(saved_image_path):
                # Image was inpainted and saved by the function
                # print(f"Hair removal successful for {filename}: {msg}")
                inpainted_count += 1
            elif "No significant hairs found" in msg or "original image skipped" in msg:
                # Hairs not significant, or function decided to skip inpainting.
                # We need to copy the original.
                # print(f"Hair removal criteria not met for {filename} ({msg}). Copying original.")
                shutil.copy2(original_image_path, target_path_in_hair_removed_dir)
                copied_original_count += 1
            else: # Some other message, or saved_image_path is None (shouldn't happen with current remove_and_save_hairs)
                print(f"Warning: Hair removal for {filename} - unexpected state or message: '{msg}'. Copying original as fallback.")
                shutil.copy2(original_image_path, target_path_in_hair_removed_dir)
                copied_original_count += 1
            
            processed_images_count += 1

        except Exception as e_hair:
            print(f"Error during hair removal for {filename}: {e_hair}. Attempting to copy original.")
            hair_removal_error_count += 1
            try:
                if not os.path.exists(target_path_in_hair_removed_dir): # Avoid re-copying if already there
                    shutil.copy2(original_image_path, target_path_in_hair_removed_dir)
                copied_original_count += 1
                processed_images_count += 1 # Count as processed for feature extraction
            except Exception as e_copy:
                print(f"Failed to copy original {filename} after hair removal error: {e_copy}")
    
    print(f"Hair removal stage summary: Total images considered: {len(original_image_files)}, Successfully processed (available in target dir): {processed_images_count}, Actually inpainted: {inpainted_count}, Originals copied: {copied_original_count}, Errors: {hair_removal_error_count}")
    print(f"Total images in '{hair_removed_img_dir_path}' for feature extraction: {len(os.listdir(hair_removed_img_dir_path))}")
>>>>>>> ae55e045a4ecce7b83d752845eedecebc55cd85e

    feature_processing_dir = hair_removed_img_dir_path

    # --- Extract Asymmetry Features (Feature A) ---
    # ... (rest of the feature extraction, metadata loading, merging, and saving remains the same)
    print(f"\nExtracting Asymmetry features from: {feature_processing_dir}")
    try:
        df_A = extract_asymmetry_features(folder_path=feature_processing_dir, output_csv=None, visualize=False)
        if df_A.empty: print("Warning: Asymmetry feature extraction (feature_A) returned an empty DataFrame.")
<<<<<<< HEAD
        else: print(f"Asymmetry features extracted: {df_A.shape[0]} images, {df_A.shape[1]-1 if 'filename' in df_A.columns else df_A.shape[1]} features.")
    except Exception as e:
        print(f"Error during Asymmetry feature extraction: {e}"); df_A = pd.DataFrame(columns=['filename'])
=======
        else: print(f"Asymmetry features extracted: {df_A.shape[0]} images, {df_A.shape[1]-1} features.")
    except Exception as e: print(f"Error during Asymmetry feature extraction: {e}"); df_A = pd.DataFrame(columns=['filename'])
>>>>>>> ae55e045a4ecce7b83d752845eedecebc55cd85e

    # --- Extract Border Features (Feature B) ---
    print(f"\nExtracting Border features from: {feature_processing_dir}")
    try:
        df_B_raw = extract_border_features_from_folder(folder_path=feature_processing_dir, output_csv=None, visualize=False)
        if df_B_raw.empty: print("Warning: Border feature extraction (feature_B raw) returned an empty DataFrame."); df_B = pd.DataFrame(columns=['filename'])
        else:
            print(f"Raw Border features extracted: {df_B_raw.shape[0]} images.")
            df_B = calculate_border_score(df_B_raw)
            print(f"Border scores calculated. Total border features df: {df_B.shape[0]} images.")
            cols_to_drop_from_B = ['sobel_mean_safe', 'avg_contour_perimeter_safe', 'laplacian_mean_safe', 'avg_contour_area_safe']
            df_B = df_B.drop(columns=[col for col in cols_to_drop_from_B if col in df_B.columns], errors='ignore')
    except Exception as e: print(f"Error during Border feature extraction: {e}"); df_B = pd.DataFrame(columns=['filename'])

    # --- Extract Color Features (Feature C) ---
    print(f"\nExtracting Color features from: {feature_processing_dir}")
    try:
        df_C = extract_feature_C(folder_path=feature_processing_dir, output_csv=None, normalize_colors=True, visualize=False)
        if df_C.empty: print("Warning: Color feature extraction (feature_C) returned an empty DataFrame.")
        else: print(f"Color features extracted: {df_C.shape[0]} images.")
    except Exception as e: print(f"Error during Color feature extraction: {e}"); df_C = pd.DataFrame(columns=['filename'])

    # --- Extract Haralick Features (Feature H) ---
    print(f"\nExtracting Haralick features from: {feature_processing_dir}")
    try:
<<<<<<< HEAD
        df_H = extract_haralick_features(folder_path=feature_processing_dir, output_csv=None, visualize=False)
=======
        df_H = extract_haralick_features(folder_path=feature_processing_dir, output_csv=None, visualize=False) 
>>>>>>> ae55e045a4ecce7b83d752845eedecebc55cd85e
        if df_H.empty: print("Warning: Haralick feature extraction (feature_H) returned an empty DataFrame.")
        else: print(f"Haralick features extracted: {df_H.shape[0]} images.")
    except Exception as e: print(f"Error during Haralick feature extraction: {e}"); df_H = pd.DataFrame(columns=['filename'])
        
    # --- Extract Blue Veil Features (Feature BV) ---
    print(f"\nExtracting Blue Veil features from: {feature_processing_dir}")
    try:
        df_BV = extract_feature_BV(folder_path=feature_processing_dir, output_csv=None, normalize_colors=True, visualize=False)
        if df_BV.empty: print("Warning: Blue Veil feature extraction (feature_BV) returned an empty DataFrame.")
        else: print(f"Blue Veil features extracted: {df_BV.shape[0]} images.")
    except Exception as e: print(f"Error during Blue Veil feature extraction: {e}"); df_BV = pd.DataFrame(columns=['filename'])

    # --- Load Labels / Metadata ---
    metadata_df = None
    if labels_csv and exists(labels_csv):
        print(f"\nLoading metadata from {labels_csv}")
        try:
            raw_metadata_df = pd.read_csv(labels_csv)
            if 'img_id' in raw_metadata_df.columns: raw_metadata_df = raw_metadata_df.rename(columns={'img_id': 'filename'})
            label_column_name = 'diagnostic'
            if 'filename' not in raw_metadata_df.columns or label_column_name not in raw_metadata_df.columns:
<<<<<<< HEAD
                print(f"ERROR: Metadata CSV must contain 'filename' (or 'img_id') and '{label_column_name}'."); metadata_df = None 
=======
                print(f"ERROR: Metadata CSV must contain 'filename' (or 'img_id') and '{label_column_name}'.")
>>>>>>> ae55e045a4ecce7b83d752845eedecebc55cd85e
            else:
                if label_column_name != 'real_label': raw_metadata_df.rename(columns={label_column_name: 'real_label'}, inplace=True)
                cancer_diagnoses = ["BCC", "SCC", "MEL"]
                raw_metadata_df['binary_target'] = raw_metadata_df['real_label'].apply(lambda x: 1 if x in cancer_diagnoses else 0)
                cols_to_keep_from_metadata = ['filename', 'real_label', 'binary_target']
                metadata_df = raw_metadata_df[cols_to_keep_from_metadata].copy()
                print(f"Metadata (filename, real_label, binary_target) selected. Shape: {metadata_df.shape[0]} entries.")
<<<<<<< HEAD
        except Exception as e: print(f"Error loading metadata: {e}"); metadata_df = None 
    else: print("\nNo metadata file provided or found. Proceeding without metadata.")
=======
        except Exception as e:
            print(f"Error loading metadata: {e}")
    else:
        print("\nNo metadata file provided or found. Proceeding without metadata.")
>>>>>>> ae55e045a4ecce7b83d752845eedecebc55cd85e

    # --- Merge DataFrames ---
    print("\nMerging feature DataFrames...")
    dataframes_to_merge = []
    if metadata_df is not None and not metadata_df.empty: dataframes_to_merge.append(metadata_df); print(f"metadata_df added. Shape: {metadata_df.shape}")
    
    feature_df_map = {"A": df_A, "B": df_B, "C": df_C, "H": df_H, "BV": df_BV}
    for name, df in feature_df_map.items():
<<<<<<< HEAD
        if not df.empty and 'filename' in df.columns: dataframes_to_merge.append(df); print(f"df_{name} added. Shape: {df.shape}")
        elif not df.empty: print(f"Skipping df_{name} in merge due to missing 'filename' column.")
        else: print(f"df_{name} is empty, not added.")

    if not dataframes_to_merge: print("No DataFrames with 'filename' column to merge. Exiting feature creation."); return pd.DataFrame()
    
=======
        if not df.empty and 'filename' in df.columns:
            dataframes_to_merge.append(df)
            print(f"df_{name} added. Shape: {df.shape}. Filename sample: {df['filename'].iloc[0] if not df.empty else 'N/A'}")
        else:
             print(f"df_{name} is empty or missing 'filename', not added.")

    if not dataframes_to_merge or \
       (metadata_df is None and not any(not df.empty for df in [df_A, df_B, df_C, df_H, df_BV])) or \
       (metadata_df is not None and len(dataframes_to_merge) < 2) : 
        print("Not enough DataFrames to merge meaningfully. Exiting feature creation.")
        return pd.DataFrame()
            
>>>>>>> ae55e045a4ecce7b83d752845eedecebc55cd85e
    final_df = dataframes_to_merge[0]
    print(f"Base DataFrame for merge (likely metadata if present): Columns: {final_df.columns.tolist()[:5]}..., Shape: {final_df.shape}")

    for i, df_to_merge in enumerate(dataframes_to_merge[1:]):
<<<<<<< HEAD
        # Logic for getting df_being_merged_name for logging (from previous response)
        df_being_merged_name = f"DataFrame_at_index_{i+1}" # Simplified for brevity here, can re-add detailed naming
        if df_being_merged_name == "Unknown_DF" and len(list(feature_df_map.keys())) > i:
             df_being_merged_name = f"df_{list(feature_df_map.keys())[i if metadata_df is None or metadata_df.empty else i-1 if i > 0 else 0]}"


        print(f"Attempting to merge with {df_being_merged_name} (shape {df_to_merge.shape})")
        common_filenames_count = len(set(final_df['filename']).intersection(set(df_to_merge['filename'])))
        if common_filenames_count == 0: print(f"CRITICAL WARNING: No common 'filename' values found for merge with {df_being_merged_name}.")
        else: print(f"Found {common_filenames_count} common 'filename' values for merge with {df_being_merged_name}.")

        final_df = pd.merge(final_df, df_to_merge, on='filename', how='inner')
        if final_df.empty: print(f"CRITICAL WARNING: DataFrame became empty after merging with {df_being_merged_name}."); break 
        else: print(f"Shape after merging with {df_being_merged_name}: {final_df.shape}")
            
    if final_df.empty: print("Resulting merged DataFrame is empty.")
    else: print(f"Merged DataFrame final shape: {final_df.shape}. Columns: {final_df.columns.tolist()}")
=======
        df_name_for_log = [k for k,v in feature_df_map.items() if v is df_to_merge]
        df_name_for_log = df_name_for_log[0] if df_name_for_log else f"DF_{i+1}"

        print(f"Merging with DataFrame {df_name_for_log}: Columns: {df_to_merge.columns.tolist()[:5]}..., Shape: {df_to_merge.shape}")
        
        common_filenames = set(final_df['filename']).intersection(set(df_to_merge['filename']))
        if not common_filenames:
            print(f"CRITICAL WARNING: No common filenames between current merged DataFrame and {df_name_for_log}. Merge will result in empty. Skipping this merge.")
            continue 

        final_df = pd.merge(final_df, df_to_merge, on='filename', how='inner')
        print(f"Shape after merging with {df_name_for_log}: {final_df.shape}")
        if final_df.empty:
            print(f"CRITICAL WARNING: DataFrame became empty after merging with {df_name_for_log}.")
            break 
            
    if final_df.empty:
        print("Resulting merged DataFrame is empty.")
    else:
        print(f"Merged DataFrame final shape: {final_df.shape}. Columns: {final_df.columns.tolist()}")
>>>>>>> ae55e045a4ecce7b83d752845eedecebc55cd85e

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    final_df.to_csv(output_csv_path, index=False)
    print(f"\nMerged feature dataset saved to {output_csv_path}")
    
    if not final_df.empty:
        expected_label_cols = ['real_label', 'binary_target']
        missing_label_cols_in_final = [col for col in expected_label_cols if col not in final_df.columns]
        if missing_label_cols_in_final:
            print(f"WARNING: Final merged DataFrame MISSING: {missing_label_cols_in_final}")
        else:
            print(f"SUCCESS: 'real_label' and 'binary_target' columns are present in the final DataFrame.")
    return final_df

<<<<<<< HEAD
# The main() function and if __name__ == "__main__": block remain the same.
# ...
=======

>>>>>>> ae55e045a4ecce7b83d752845eedecebc55cd85e
def main(original_img_dir, mask_img_dir, labels_csv_path, output_csv_path, result_path, recreate_features=False):
    print("\n--- FEATURE DATASET CREATION (EXTENDED FEATURES) ---\n")
    
    if not original_img_dir or not output_csv_path:
        raise ValueError("original_img_dir and output_csv_path must be provided.")

    data_df = None
    # The logic for loading/creating the main feature CSV remains the same.
    # The create_feature_dataset function now handles the hair removal caching.
    if recreate_features or not exists(output_csv_path):
        print(f"Creating new feature dataset at {output_csv_path}")
        # Pass recreate_features to create_feature_dataset so it knows whether to rebuild hair_removed_images
        data_df = create_feature_dataset(original_img_dir, mask_img_dir, output_csv_path, 
                                         labels_csv=labels_csv_path, recreate_features=recreate_features)
    else:
        print(f"Loading existing feature dataset from {output_csv_path}")
        try:
            data_df = pd.read_csv(output_csv_path)
            # Even if CSV exists, we might want to ensure hair-removed images are present if they were deleted
            # or if this is a new run on a different machine.
            # For simplicity here, we assume if the CSV exists, the pre-processing was done.
            # A more robust approach might involve checking the hair_removed_img_dir_path consistency here too.
        except Exception as e:
            print(f"Error loading existing dataset: {e}. Will attempt to recreate.")
            data_df = create_feature_dataset(original_img_dir, mask_img_dir, output_csv_path, 
<<<<<<< HEAD
                                             labels_csv=labels_csv_path, recreate_features=True) 
=======
                                             labels_csv=labels_csv_path, recreate_features=True)
>>>>>>> ae55e045a4ecce7b83d752845eedecebc55cd85e

    if data_df is None or data_df.empty:
        print("Failed to create or load the feature dataset. Exiting.")
        return
    
    print("\n--- Merged Dataset Information ---")
    data_df.info(verbose=True, show_counts=True) 
    print("\nFirst 5 rows of the merged dataset:")
    print(data_df.head())
    
    if 'filename' not in data_df.columns:
        print("CRITICAL ERROR: 'filename' column is missing in the final DataFrame!")
        return
    if data_df['filename'].isnull().any():
<<<<<<< HEAD
        print(f"Warning: {data_df['filename'].isnull().sum()} 'filename' entries are NaN.")
=======
        print(f"Warning: {data_df['filename'].isnull().sum()} 'filename' entries are NaN. This usually indicates an issue in merging.")
>>>>>>> ae55e045a4ecce7b83d752845eedecebc55cd85e
    if data_df['filename'].duplicated().any():
        print(f"Warning: {data_df['filename'].duplicated().sum()} duplicate filenames found. Consolidating by keeping the first occurrence.")
        data_df = data_df.drop_duplicates(subset=['filename'], keep='first').reset_index(drop=True)

    print("\n--- MODEL TRAINING AND EVALUATION ---")
    
    if 'binary_target' not in data_df.columns:
        source_label_col = next((col for col in ['real_label', 'diagnostic'] if col in data_df.columns), None)
        if source_label_col:
            print(f"Creating 'binary_target' from '{source_label_col}'.")
            cancer_diagnoses_map = ["BCC", "SCC", "MEL"]
            data_df['binary_target'] = data_df[source_label_col].apply(lambda x: 1 if x in cancer_diagnoses_map else 0)
            if source_label_col == 'diagnostic' and 'real_label' not in data_df.columns:
                data_df.rename(columns={'diagnostic': 'real_label'}, inplace=True)
        else:
            print("CRITICAL ERROR: Target label column ('binary_target', 'real_label', or 'diagnostic') not found."); return
    
    data_df.dropna(subset=['binary_target'], inplace=True) 
    if data_df.empty: print("CRITICAL ERROR: Dataset empty after dropping rows with missing 'binary_target'."); return

    data_df['label'] = data_df['binary_target'].astype(int)
    class_names_for_report = ['non-cancer', 'cancer'] 
    print(f"\nLabel distribution:\n{data_df['label'].value_counts(normalize=True)}")

<<<<<<< HEAD
    if 'c_dominant_channel' in data_df.columns:
        print("\nOne-hot encoding 'c_dominant_channel'...")
        try: data_df = pd.get_dummies(data_df, columns=['c_dominant_channel'], prefix='c_dom_channel', dummy_na=False); print("'c_dominant_channel' one-hot encoded.")
=======
    if 'real_label' not in data_df.columns and 'diagnostic' in data_df.columns:
        data_df['real_label'] = data_df['diagnostic']
    elif 'real_label' not in data_df.columns:
        data_df['real_label'] = data_df['label'].map({0: 'derived_non-cancer', 1: 'derived_cancer'})

    if 'c_dominant_channel' in data_df.columns: # From feature C
        print("\nOne-hot encoding 'c_dominant_channel'...")
        try:
            data_df = pd.get_dummies(data_df, columns=['c_dominant_channel'], prefix='c_dom_channel', dummy_na=False)
>>>>>>> ae55e045a4ecce7b83d752845eedecebc55cd85e
        except Exception as e_ohe: print(f"Error during one-hot encoding 'c_dominant_channel': {e_ohe}")
    
    potential_non_feature_cols = ['filename', 'real_label', 'binary_target', 'label', 'diagnostic', 
                                  'patient_id', 'lesion_id', 'smoke', 'drink', 
                                  'background_father', 'background_mother', 'age', 'pesticide', 
                                  'gender', 'skin_cancer_history', 'cancer_history', 'has_piped_water', 
                                  'has_sewage_system', 'fitspatrick', 'region', 'diameter_1', 
                                  'diameter_2', 'itch', 'grew', 'hurt', 'changed', 'bleed', 
                                  'elevation', 'biopsed'] 
    feature_columns = [col for col in data_df.columns if col not in potential_non_feature_cols]
    
<<<<<<< HEAD
    if not feature_columns: print("CRITICAL ERROR: No feature columns identified. Cannot train model."); return
    print(f"\nUsing {len(feature_columns)} feature columns for training: {feature_columns[:10]}...")
=======
    if not feature_columns:
        print("CRITICAL ERROR: No feature columns identified. Cannot train model.")
        return
    print(f"\nUsing {len(feature_columns)} feature columns for training. First 10: {feature_columns[:10]}...")
>>>>>>> ae55e045a4ecce7b83d752845eedecebc55cd85e

    x_all = data_df[feature_columns].copy()
    y_all = data_df["label"].copy() 
    current_filenames = data_df['filename'].copy()

    print("\nConverting features to numeric and handling NaNs/Infs...")
    for feat in feature_columns: x_all[feat] = pd.to_numeric(x_all[feat], errors='coerce')

    all_nan_cols = x_all.columns[x_all.isnull().all()].tolist()
    if all_nan_cols:
        print(f"Warning: Columns became all NaN and will be dropped: {all_nan_cols}")
        x_all = x_all.drop(columns=all_nan_cols)
        feature_columns = [col for col in feature_columns if col not in all_nan_cols]
        if not feature_columns: print("CRITICAL ERROR: All feature columns dropped."); return
    
    x_all.replace([np.inf, -np.inf], np.nan, inplace=True)
    imputer = SimpleImputer(strategy='mean') 
<<<<<<< HEAD
    if x_all.empty: print("CRITICAL ERROR: x_all DataFrame empty before imputation."); return
=======
    
    if x_all.empty: print("CRITICAL ERROR: x_all DataFrame empty before imputation."); return

>>>>>>> ae55e045a4ecce7b83d752845eedecebc55cd85e
    x_all_imputed = imputer.fit_transform(x_all)
    x_all = pd.DataFrame(x_all_imputed, columns=x_all.columns, index=x_all.index)

    if len(x_all) == 0 or y_all.nunique() < 2:
<<<<<<< HEAD
        print(f"Skipping model training: Insufficient samples/classes. Samples: {len(x_all)}, Unique Labels: {y_all.nunique()}"); return
=======
        print(f"Skipping model training: Samples: {len(x_all)}, Unique Labels: {y_all.nunique()}")
        return
>>>>>>> ae55e045a4ecce7b83d752845eedecebc55cd85e

    print(f"\nSplitting data (Total samples: {len(x_all)})...")
    try:
        x_train, x_temp, y_train, y_temp, filenames_train, filenames_temp = train_test_split(
            x_all.reset_index(drop=True), y_all.reset_index(drop=True), current_filenames.reset_index(drop=True), 
            test_size=0.4, random_state=42, stratify=y_all.reset_index(drop=True)
        )
        x_val, x_test, y_val, y_test, filenames_val, filenames_test = train_test_split(
            x_temp.reset_index(drop=True), y_temp.reset_index(drop=True), filenames_temp.reset_index(drop=True), 
            test_size=0.5, random_state=42, stratify=y_temp.reset_index(drop=True)
        )
<<<<<<< HEAD
    except ValueError as e_split: print(f"Error during data splitting: {e_split}. Labels: \n{y_all.value_counts()}"); return
=======
    except ValueError as e_split:
        print(f"Error during data splitting: {e_split}. Class distribution: \n{y_all.value_counts()}")
        return
>>>>>>> ae55e045a4ecce7b83d752845eedecebc55cd85e

    print(f"Training set: {len(x_train)}, Validation set: {len(x_val)}, Test set: {len(x_test)}")

    try:
<<<<<<< HEAD
        if x_train.empty or x_val.empty or y_train.nunique() < 2:
            print("Training/validation empty or insufficient classes in training target. Skipping model training."); return
=======
        if x_train.empty or x_val.empty or y_train.nunique() < 2 :
            print("Training/validation set empty or insufficient classes. Skipping model training.")
            return
>>>>>>> ae55e045a4ecce7b83d752845eedecebc55cd85e

        best_model, best_model_name, best_val_acc = train_and_select_model(x_train, y_train, x_val, y_val)
        if best_model is None: print("No model selected. Exiting."); return

        print("\n--- TEST PHASE ---")
        y_test_pred = best_model.predict(x_test)
        y_test_pred_proba = None
        if hasattr(best_model, "predict_proba"):
            y_test_pred_proba = best_model.predict_proba(x_test)
        else:
            print(f"Warning: Model {best_model_name} does not have predict_proba.")
            y_test_pred_proba = np.zeros((len(y_test_pred), len(class_names_for_report)))
            for i, pred_label in enumerate(y_test_pred): y_test_pred_proba[i, pred_label] = 1.0

        test_acc = accuracy_score(y_test, y_test_pred)
        cm_display = confusion_matrix(y_test, y_test_pred, labels=[0,1])
        cls_report_dict = classification_report(y_test, y_test_pred, labels=[0,1], target_names=class_names_for_report, output_dict=True, zero_division=0)
        cls_report_str = classification_report(y_test, y_test_pred, labels=[0,1], target_names=class_names_for_report, zero_division=0)
        
        print(f"\nBest Model: {best_model_name}, Test Accuracy: {test_acc:.4f}")
        print(f"Confusion Matrix:\n{cm_display}\nClassification Report:\n{cls_report_str}")

        test_results_df = pd.DataFrame({
<<<<<<< HEAD
            'filename': filenames_test.values, 'true_label_encoded': y_test.values, 'predicted_label_encoded': y_test_pred,
            'true_label_text': y_test.map({0: 'non-cancer', 1: 'cancer'}).values,
=======
            'filename': filenames_test.reset_index(drop=True).values,
            'true_label_encoded': y_test.reset_index(drop=True).values,
            'predicted_label_encoded': y_test_pred,
            'true_label_text': y_test.reset_index(drop=True).map({0: 'non-cancer', 1: 'cancer'}).values,
>>>>>>> ae55e045a4ecce7b83d752845eedecebc55cd85e
            'predicted_label_text': pd.Series(y_test_pred).map({0: 'non-cancer', 1: 'cancer'}).values
        })
        if y_test_pred_proba is not None and y_test_pred_proba.shape[0] == len(y_test_pred) and y_test_pred_proba.shape[1] >= len(class_names_for_report):
            test_results_df[f'proba_{class_names_for_report[0]}'] = y_test_pred_proba[:, 0]
            test_results_df[f'proba_{class_names_for_report[1]}'] = y_test_pred_proba[:, 1]
        
        test_details_csv_path = os.path.join(os.path.dirname(result_path), f"{os.path.splitext(os.path.basename(result_path))[0]}_predictions_details.csv") 
        test_results_df.to_csv(test_details_csv_path, index=False)
        print(f"Detailed test predictions saved to {test_details_csv_path}")

        summary_report_data = {
            'model_name': best_model_name, 'validation_accuracy': best_val_acc, 'test_accuracy': test_acc,
            'num_training_samples': len(x_train), 'num_validation_samples': len(x_val), 'num_test_samples': len(x_test),
            'num_features_used': len(feature_columns),
        }
        for class_label_report in class_names_for_report: 
            if class_label_report in cls_report_dict:
                for metric in ['precision', 'recall', 'f1-score', 'support']:
                    summary_report_data[f'{class_label_report}_{metric}_test'] = cls_report_dict[class_label_report][metric]
        for avg_type in ['macro avg', 'weighted avg']:
            if avg_type in cls_report_dict:
                 for metric in ['precision', 'recall', 'f1-score']:
                    summary_report_data[f'{avg_type.replace(" ", "_")}_{metric}_test'] = cls_report_dict[avg_type][metric]

        results_summary_df = pd.DataFrame([summary_report_data])
<<<<<<< HEAD
        extended_result_path = os.path.join(os.path.dirname(result_path), f"{os.path.splitext(os.path.basename(result_path))[0]}_extended.csv") # Ensure unique name
        results_summary_df.to_csv(extended_result_path, index=False)
        print(f"Summary model evaluation results saved to {extended_result_path}")
=======
        results_summary_df.to_csv(result_path, index=False) 
        print(f"Summary model evaluation results saved to {result_path}")
>>>>>>> ae55e045a4ecce7b83d752845eedecebc55cd85e

    except Exception as e_model:
        print(f"Error during model training/evaluation: {e_model}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    original_img_dir = r"C:\Users\Erik\OneDrive - ITU\Escritorio\2 semester\Semester project\Introduction to final project\matched_pairs\images"
    mask_img_dir = r"C:\Users\Erik\OneDrive - ITU\Escritorio\2 semester\Semester project\Introduction to final project\matched_pairs\masks" 
    labels_csv_path = r"C:\Users\Erik\OneDrive - ITU\Escritorio\2 semester\Semester project\Introduction to final project\2025-FYP-Final\data\filtered_metadata_img_id_first.csv"
    
    # Changed output directory for extended features to keep them separate
    output_feature_csv_dir = "./result_extended" 
    os.makedirs(output_feature_csv_dir, exist_ok=True)
    
<<<<<<< HEAD
    merged_csv_filename = "dataset_extended_ABC_H_BV_features.csv" 
    output_csv_path = os.path.join(output_feature_csv_dir, merged_csv_filename)
    
    model_result_filename = "model_evaluation_extended_features_summary.csv" 
    result_path = os.path.join(output_feature_csv_dir, model_result_filename)
    
    try:
        # Set recreate_features=False to try and use existing hair-removed images and existing feature CSV.
        # Set to True to force regeneration of everything.
        main(original_img_dir, mask_img_dir, labels_csv_path, output_csv_path, result_path, recreate_features=False) 
=======
    # New filename for the dataset with extended features
    merged_csv_filename = "dataset_extended_features_ABC_H_BV.csv" 
    output_csv_path = os.path.join(output_feature_csv_dir, merged_csv_filename)
    
    # New model result filename for extended features
    model_result_filename = "model_evaluation_extended_summary.csv" 
    result_path = os.path.join(output_feature_csv_dir, model_result_filename)
    
    try:
        main(original_img_dir, mask_img_dir, labels_csv_path, output_csv_path, result_path, recreate_features=True) 
>>>>>>> ae55e045a4ecce7b83d752845eedecebc55cd85e
    except Exception as e:
        print(f"Error running main script: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)