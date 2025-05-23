# In main_baseline.py
import sys
import os
from os.path import join, exists
import numpy as np
import pandas as pd
# from sklearn.linear_model import LogisticRegression # Not needed for just CSV creation
# from sklearn.metrics import accuracy_score, confusion_matrix # Not needed for just CSV creation
# from sklearn.model_selection import train_test_split # Not needed for just CSV creation

# Import custom modules
try:
    # Assuming your feature files are in a 'util' subdirectory
    # If they are in the same directory as main_baseline.py, change the import path
    # e.g., from feature_A import extract_asymmetry_features
    from util.feature_A import extract_asymmetry_features
    from util.feature_B import extract_border_features_from_folder, calculate_border_score
    from util.feature_C import extract_feature_C
    # train_and_select_model might not be needed if only creating the dataset
    # from models_evaluation import train_and_select_model
except ImportError as e:
    print(f"Error: Could not import custom feature modules: {e}")
    print("Please ensure feature_A.py, feature_B.py, feature_C.py are in the 'util' directory (or adjust import paths).")
    print("The required functions are: extract_asymmetry_features, extract_border_features_from_folder, calculate_border_score, extract_feature_C.")
    sys.exit(1)
# In main_baseline.py

def create_feature_dataset(original_img_dir, mask_img_dir, output_csv_path, labels_csv=None):
    print("Starting feature extraction process...")

    if not exists(original_img_dir):
        raise FileNotFoundError(f"Original image directory not found: {original_img_dir}")

    # --- Extract Asymmetry Features (Feature A) ---
    print(f"\nExtracting Asymmetry features from: {original_img_dir}")
    # Pass output_csv=None to prevent feature_A from saving its own CSV
    # It will return the DataFrame directly.
    try:
        df_A = extract_asymmetry_features(folder_path=original_img_dir, output_csv=None, visualize=False)
        if df_A.empty:
            print("Warning: Asymmetry feature extraction (feature_A) returned an empty DataFrame.")
        else:
            print(f"Asymmetry features extracted: {df_A.shape[0]} images, {df_A.shape[1]-1} features (excluding filename).")
            # Ensure filename column is present
            if 'filename' not in df_A.columns and not df_A.empty:
                 print("CRITICAL WARNING: df_A is missing 'filename' column!")
                 # Potentially create one if image order is guaranteed, but safer to fix feature_A.py
    except Exception as e:
        print(f"Error during Asymmetry feature extraction: {e}")
        df_A = pd.DataFrame(columns=['filename']) # Empty df with filename to allow merge

    # --- Extract Border Features (Feature B) ---
    print(f"\nExtracting Border features from: {original_img_dir}")
    try:
        # extract_border_features_from_folder returns a DataFrame with raw border features
        df_B_raw = extract_border_features_from_folder(folder_path=original_img_dir, output_csv=None, visualize=False)
        if df_B_raw.empty:
            print("Warning: Border feature extraction (feature_B raw) returned an empty DataFrame.")
            df_B = pd.DataFrame(columns=['filename']) # Empty df with filename for merge
        else:
            print(f"Raw Border features extracted: {df_B_raw.shape[0]} images, {df_B_raw.shape[1]-1} features.")
            if 'filename' not in df_B_raw.columns and not df_B_raw.empty:
                print("CRITICAL WARNING: df_B_raw is missing 'filename' column!")

            # Calculate border score using the dedicated function
            df_B = calculate_border_score(df_B_raw)
            print(f"Border scores calculated. Total border features df: {df_B.shape[0]} images, {df_B.shape[1]-1} features.")
            
            # Optional: Drop intermediate columns from calculate_border_score if they are not needed in the final CSV
            cols_to_drop_from_B = ['sobel_mean_safe', 'avg_contour_perimeter_safe', 'laplacian_mean_safe', 'avg_contour_area_safe']
            df_B = df_B.drop(columns=[col for col in cols_to_drop_from_B if col in df_B.columns], errors='ignore')

    except Exception as e:
        print(f"Error during Border feature extraction: {e}")
        df_B = pd.DataFrame(columns=['filename']) # Empty df with filename

    # --- Extract Color Features (Feature C) ---
    print(f"\nExtracting Color features from: {original_img_dir}")
    # feature_C's main function processes a folder.
    # Note: feature_C.py uses its own segmentation, mask_img_dir might not be directly used by it
    # unless you modify feature_C to accept pre-computed masks.
    try:
        df_C = extract_feature_C(folder_path=original_img_dir, output_csv=None, normalize_colors=True, visualize=False)
        if df_C.empty:
            print("Warning: Color feature extraction (feature_C) returned an empty DataFrame.")
        else:
            print(f"Color features extracted: {df_C.shape[0]} images, {df_C.shape[1]-1} features.")
            if 'filename' not in df_C.columns and not df_C.empty:
                 print("CRITICAL WARNING: df_C is missing 'filename' column!")
    except Exception as e:
        print(f"Error during Color feature extraction: {e}")
        df_C = pd.DataFrame(columns=['filename']) # Empty df with filename

    # --- Load Labels ---
    labels_df = None
    if labels_csv and exists(labels_csv):
        print(f"\nLoading labels from {labels_csv}")
        try:
            labels_df = pd.read_csv(labels_csv)
            if 'filename' not in labels_df.columns or 'label' not in labels_df.columns:
                print("Warning: Labels CSV must contain 'filename' and 'label' columns. Proceeding without labels from this file.")
                labels_df = None # Reset if format is wrong
            else:
                print(f"Labels loaded: {labels_df.shape[0]} entries.")
        except Exception as e:
            print(f"Error loading labels: {e}. Proceeding without labels.")
            labels_df = None
    else:
        print("\nNo labels file provided or file doesn't exist. Proceeding without labels.")

    # --- Merge DataFrames ---
    print("\nMerging feature DataFrames...")
    
    # Start with a list of DataFrames to merge. Ensure 'filename' is the key.
    dataframes_to_merge = []
    if labels_df is not None and not labels_df.empty and 'filename' in labels_df.columns:
        dataframes_to_merge.append(labels_df)
    
    if not df_A.empty and 'filename' in df_A.columns:
        dataframes_to_merge.append(df_A)
    elif not df_A.empty:
        print("Skipping df_A in merge due to missing 'filename' column.")
        
    if not df_B.empty and 'filename' in df_B.columns:
        dataframes_to_merge.append(df_B)
    elif not df_B.empty:
        print("Skipping df_B in merge due to missing 'filename' column.")

    if not df_C.empty and 'filename' in df_C.columns:
        dataframes_to_merge.append(df_C)
    elif not df_C.empty:
        print("Skipping df_C in merge due to missing 'filename' column.")

    if not dataframes_to_merge:
        print("No DataFrames to merge. Exiting feature creation.")
        return pd.DataFrame()

    # Perform the merge
    final_df = dataframes_to_merge[0]
    for df_to_merge in dataframes_to_merge[1:]:
        final_df = pd.merge(final_df, df_to_merge, on='filename', how='outer') # 'outer' merge to keep all filenames

    if final_df.empty:
        print("Resulting merged DataFrame is empty.")
    else:
        print(f"Merged DataFrame shape: {final_df.shape}")

    # Ensure 'label' column exists, fill with a default (e.g., -1 or np.nan) if it wasn't in labels_df or for files not in labels_df
    if 'label' not in final_df.columns:
        if labels_df is not None : # If we expected labels but they didn't merge in for all
            print("Adding 'label' column with default NaN for missing labels.")
            final_df['label'] = np.nan # Or -1, depending on how you want to handle missing labels
        # If labels_df was None, no label column is added unless you explicitly want one

    # Save the merged DataFrame
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    final_df.to_csv(output_csv_path, index=False)
    print(f"\nMerged feature dataset saved to {output_csv_path}")
    
    if not final_df.empty:
        feature_cols = [col for col in final_df.columns if col not in ['filename', 'label']]
        print(f"Dataset contains {len(feature_cols)} feature columns.")
        # print(f"Example feature columns: {feature_cols[:5]}") # Print first 5 feature names
    
    return final_df

# In main_baseline.py

def main(original_img_dir, mask_img_dir, labels_csv_path, output_csv_path, result_path, recreate_features=False):
    print("\n--- FEATURE DATASET CREATION ---\n")
    
    if not original_img_dir or not output_csv_path: # result_path not strictly needed for dataset creation
        raise ValueError("original_img_dir and output_csv_path must be provided.")

    data_df = None
    if recreate_features or not exists(output_csv_path):
        print(f"Creating new feature dataset at {output_csv_path}")
        data_df = create_feature_dataset(original_img_dir, mask_img_dir, output_csv_path, labels_csv=labels_csv_path)
    else:
        print(f"Loading existing feature dataset from {output_csv_path}")
        try:
            data_df = pd.read_csv(output_csv_path)
        except Exception as e:
            print(f"Error loading existing dataset: {e}. Will attempt to recreate.")
            data_df = create_feature_dataset(original_img_dir, mask_img_dir, output_csv_path, labels_csv=labels_csv_path)

    if data_df is not None and not data_df.empty:
        print("\n--- Merged Dataset Information ---")
        print(data_df.info())
        print("\nFirst 5 rows of the merged dataset:")
        print(data_df.head())
        
        # Check for common issues:
        if 'filename' not in data_df.columns:
            print("CRITICAL ERROR: 'filename' column is missing in the final DataFrame!")
        else:
            if data_df['filename'].isnull().any():
                print("Warning: Some 'filename' entries are NaN after merge. This usually indicates an issue in how DataFrames were constructed or merged.")
            if data_df['filename'].duplicated().any():
                print("Warning: Duplicate filenames found in the merged DataFrame. Check for issues in source data or merging logic.")

        # Identify feature columns (all columns except 'filename' and 'label', if 'label' exists)
        feature_columns = [col for col in data_df.columns if col not in ['filename', 'label']]
        
        # Handle categorical 'c_dominant_channel' from feature_C.py
        # For simplicity, if you're just creating the CSV, you can leave it.
        # If you proceed to model training, it needs one-hot encoding or similar.
        if 'c_dominant_channel' in data_df.columns:
            print("\nNote: 'c_dominant_channel' is a categorical feature. It may need encoding for model training.")
            # Example: Convert 'c_dominant_channel' to numerical if it's only one type or one-hot encode
            # For now, just acknowledge its presence.

        # --- MODEL TRAINING AND EVALUATION (Commented out for CSV creation focus) ---
        # If you want to proceed with model training later, uncomment and adapt this section.
        # Ensure 'label' column exists and has valid data.
        # Ensure feature columns are numeric and handle NaNs/Infs.
        
        # print("\n--- MODEL TRAINING AND EVALUATION (SKIPPED FOR CSV CREATION) ---")
        # if 'label' not in data_df.columns:
        #     print("Skipping model training: 'label' column not found in the dataset.")
        #     return

        # # Prepare data for model
        # baseline_feats = [col for col in data_df.columns if col not in ['filename', 'label', 'c_dominant_channel']] # Exclude non-numeric/identifiers
        
        # if not baseline_feats:
        #     print("Skipping model training: No feature columns identified.")
        #     return

        # # Convert features to numeric, coercing errors, then fill NaNs
        # for feat in baseline_feats:
        #     data_df[feat] = pd.to_numeric(data_df[feat], errors='coerce')
        # data_df[baseline_feats] = data_df[baseline_feats].fillna(data_df[baseline_feats].mean()) # Simple mean imputation
        # data_df.replace([np.inf, -np.inf], np.nan, inplace=True) # Replace infs
        # data_df[baseline_feats] = data_df[baseline_feats].fillna(data_df[baseline_feats].mean()) # Impute again if infs became NaNs

        # x_all = data_df[baseline_feats]
        # y_all = data_df["label"]
        
        # # Remove rows with missing labels (if any)
        # valid_label_mask = y_all.notna() & (y_all != -1) # Assuming -1 might be another missing indicator
        # x_all = x_all[valid_label_mask]
        # y_all = y_all[valid_label_mask]
        
        # if len(x_all) == 0:
        #     print("Skipping model training: No samples with valid labels found.")
        #     return
        # if y_all.nunique() < 2:
        #     print("Skipping model training: Not enough unique classes in labels for stratified split or training.")
        #     return

        # print(f"Using {len(x_all)} samples with valid labels for model training.")

        # x_train, x_temp, y_train, y_temp = train_test_split(x_all, y_all, test_size=0.3, random_state=42, stratify=y_all)
        # x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

        # # Ensure models_evaluation.py and train_and_select_model are correctly set up
        # try:
        #     from models_evaluation import train_and_select_model # Re-import if necessary
        #     best_model, best_model_name = train_and_select_model(x_train, y_train, x_val, y_val)

        #     print("\n--- TEST PHASE ---")
        #     y_test_pred = best_model.predict(x_test)
        #     test_acc = accuracy_score(y_test, y_test_pred)
        #     cm = confusion_matrix(y_test, y_test_pred)
        #     print(f"Best Model: {best_model_name}")
        #     print(f"Test Accuracy: {test_acc:.4f}")
        #     print(f"Confusion Matrix:\n{cm}")
            
        #     # Save results
        #     results_df = pd.DataFrame([{'model_name': best_model_name, 'test_accuracy': test_acc}])
        #     os.makedirs(os.path.dirname(result_path), exist_ok=True)
        #     results_df.to_csv(result_path, index=False)
        #     print(f"Results saved to {result_path}")
        # except ImportError:
        #     print("Skipping model training: 'models_evaluation' module or 'train_and_select_model' function not found.")
        # except Exception as e_model:
        #     print(f"Error during model training/evaluation: {e_model}")

    else:
        print("Failed to create or load the feature dataset.")


if __name__ == "__main__":
    # Configure paths - ADJUST THESE TO YOUR SPECIFIC FOLDERS
    # Assuming 'util' folder is at the same level as main_baseline.py or in Python path
    original_img_dir = r"C:\Users\Erik\OneDrive - ITU\Escritorio\2 semester\Semester project\Introduction to final project\matched_pairs\images"
    mask_img_dir = r"C:\Users\Erik\OneDrive - ITU\Escritorio\2 semester\Semester project\Introduction to final project\matched_pairs\masks" # Used by feature_B if adapted, less by A and C
    labels_csv_path = "./dataset.csv"  # Your CSV with 'filename' and 'label' columns
    
    # Define output path for the final merged CSV
    # Place it in 'result' directory to keep things organized
    merged_csv_filename = "merged_ABC_features.csv"
    output_csv_path = os.path.join("./result", merged_csv_filename) # e.g., ./result/merged_ABC_features.csv
    
    # Result path for model evaluation (if you uncomment that part)
    model_result_filename = "result_baseline_model.csv"
    result_path = os.path.join("./result", model_result_filename) # e.g., ./result/result_baseline_model.csv

    # Ensure result directory exists
    os.makedirs("./result", exist_ok=True)
    
    try:
        # Set recreate_features=True to always regenerate the CSV
        # Set to False to load if exists, or create if not.
        main(original_img_dir, mask_img_dir, labels_csv_path, output_csv_path, result_path, recreate_features=True)
    except Exception as e:
        print(f"Error running main script: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Configure paths - adjust these to your specific folders
    original_img_dir = r"C:\Users\Erik\OneDrive - ITU\Escritorio\2 semester\Semester project\Introduction to final project\matched_pairs\images"
    mask_img_dir = r"C:\Users\Erik\OneDrive - ITU\Escritorio\2 semester\Semester project\Introduction to final project\matched_pairs\masks"
    labels_csv_path = r"C:\Users\Erik\OneDrive - ITU\Escritorio\2 semester\Semester project\Introduction to final project\2025-FYP-Final\data\filtered_metadata_img_id_first.csv"
    output_csv_path = "./dataset_baseline_features.csv"
    result_path = r"C:\Users\Erik\OneDrive - ITU\Escritorio\2 semester\Semester project\Introduction to final project\2025-FYP-Final\resultresult_baseline.csv"
    
    try:
        main(original_img_dir, mask_img_dir, labels_csv_path, output_csv_path, result_path, recreate_features=True)
    except Exception as e:
        print(f"Error running main: {e}")
        sys.exit(1)
