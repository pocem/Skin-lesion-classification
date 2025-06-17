# build_binary_dataset.py
import sys
import os
import pandas as pd
from tqdm import tqdm

# Import REFINED custom modules
try:
    from util.feature_A import extract_asymmetry_features
    from util.feature_B import extract_border_features_from_folder
    from util.feature_C import extract_feature_C
    from util.blue_veil import extract_blue_veil_features
except ImportError as e:
    print(f"Error: Could not import refined custom feature modules: {e}")
    print("Please ensure feature_A.py, feature_B.py, feature_C.py, and blue_veil.py are in the 'util' directory.")
    sys.exit(1)

def normalize_filename(filename: str) -> str:
    """
    A centralized function to guarantee a standard filename format.
    Removes extensions and common suffixes like '_mask'.
    """
    base = os.path.basename(str(filename))
    name_without_ext = os.path.splitext(base)[0]
    cleaned_name = name_without_ext.replace('_mask', '')
    return cleaned_name

def build_dataset(original_img_dir: str, mask_img_dir: str, labels_csv_path: str, final_output_path: str) -> bool:
    """
    Runs the full data pipeline: extracts all features, loads metadata,
    normalizes, merges, creates a binary label, and saves a final master dataset.
    
    Returns:
        True if successful, False otherwise.
    """
    print("--- Starting Full Data Build and ETL Pipeline for BINARY Classification ---")

    # --- 1. Feature Extraction from Raw Images ---
    dfs = {}
    feature_extractors = {
        "A_asymmetry": (extract_asymmetry_features, {'folder_path': mask_img_dir}),
        "B_border": (extract_border_features_from_folder, {'folder_path': original_img_dir}),
        "C_color": (extract_feature_C, {'folder_path': original_img_dir}),
        "BV_blue_veil": (extract_blue_veil_features, {'folder_path': original_img_dir})
    }

    for name, (func, params) in feature_extractors.items():
        print(f"\nExtracting {name} features...")
        try:
            temp_df = func(**params)
            if temp_df.empty:
                print(f"Warning: {name} extraction returned an empty DataFrame.")
            dfs[name] = temp_df
        except Exception as e:
            print(f"Error during {name} extraction: {e}")
            dfs[name] = pd.DataFrame()

    # --- 2. Metadata Loading ---
    print(f"\nLoading metadata from: {labels_csv_path}")
    try:
        metadata_df = pd.read_csv(labels_csv_path)
    except FileNotFoundError:
        print(f"FATAL ERROR: Metadata file not found at '{labels_csv_path}'.")
        return False

    # --- 3. Centralized Normalization ---
    print("\n--- Standardizing all filenames centrally ---")
    
    for name, df in dfs.items():
        if not df.empty and 'filename' in df.columns:
            df['filename'] = df['filename'].apply(normalize_filename)

    metadata_df.columns = metadata_df.columns.str.strip()
    if 'filename' in metadata_df.columns and 'diagnostic' in metadata_df.columns:
        metadata_df.rename(columns={'diagnostic': 'real_label'}, inplace=True)
        metadata_df['filename'] = metadata_df['filename'].apply(normalize_filename)
        metadata_df = metadata_df[['filename', 'real_label']].copy()
    else:
        print("FATAL ERROR: 'filename' or 'diagnostic' column missing in metadata CSV.")
        return False
        
    # --- 4. Robust Merging Logic ---
    print("\nMerging all DataFrames...")
    final_df = metadata_df
    for name, df in dfs.items():
        if not df.empty:
            final_df = pd.merge(final_df, df, on='filename', how='inner')
    
    if final_df.empty:
        print("CRITICAL: Merge resulted in an empty DataFrame. Halting.")
        return False

    # --- 5. Create Binary Label ---
    print("\nCreating binary labels (Malignant vs. Benign)...")
    malignant_classes = ['BCC', 'SCC', 'MEL', 'ACK']
    
    if 'real_label' not in final_df.columns:
        print("VALIDATION FAILED: 'real_label' column is missing before binary conversion.")
        return False
        
    final_df['label'] = final_df['real_label'].apply(
        lambda lesion_type: 1 if lesion_type in malignant_classes else 0
    )
    
    print("Binary label distribution:")
    print(final_df['label'].value_counts(normalize=True))
    
    # We can now drop the original multi-class text label
    final_df.drop(columns=['real_label'], inplace=True)
    
    # --- 6. Final Validation and Save ---
    print(f"\nProcess complete. Final dataset shape: {final_df.shape}")
    if 'label' not in final_df.columns:
        print("FINAL VALIDATION FAILED: Final binary 'label' column is missing.")
        return False
        
    print("VALIDATION SUCCESSFUL: Final binary dataset is ready.")
    output_dir = os.path.dirname(final_output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    final_df.to_csv(final_output_path, index=False)
    print(f"Successfully saved final binary dataset to: {final_output_path}")
    return True


if __name__ == "__main__":
    # --- Define Input Paths ---
    matched_data_dir = r"C:\Users\misog\portfolio\Machine learning skin lesion project\matched_data"
    original_img_dir = os.path.join(matched_data_dir, "images")
    mask_img_dir = os.path.join(matched_data_dir, "masks")
    labels_csv_path = r'C:\Users\misog\portfolio\Machine learning skin lesion project\Skin-lesion-classification\data\metadata_matched.csv'
    
    # --- Define Output Path ---
    # This is the single, clean file this script will create, ready for binary modeling
    FINAL_BINARY_DATASET_CSV = r'./final_binary_dataset.csv'

    # --- Pre-run Check ---
    if not os.path.exists(labels_csv_path):
        print(f"FATAL ERROR: The metadata file was not found at: {os.path.abspath(labels_csv_path)}")
        sys.exit(1)
        
    # --- Run the Build Pipeline ---
    success = build_dataset(original_img_dir, mask_img_dir, labels_csv_path, FINAL_BINARY_DATASET_CSV)
    
    if success:
        print("\nBinary data build process finished successfully.")
    else:
        print("\nBinary data build process failed. Please check the error messages above.")
        sys.exit(1)