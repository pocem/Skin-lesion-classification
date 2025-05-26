import cv2
import numpy as np
import os

def remove_and_save_hairs(
    image_path,
    output_dir,
    blackhat_kernel_size=(15, 15),
    threshold_value=18,
    dilation_kernel_size=(3, 3),
    dilation_iterations=2,
    
    inpaint_radius=5,
    min_hair_contours_to_process=3,
    min_contour_area=15,
):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(image_path)

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel_blackhat = cv2.getStructuringElement(cv2.MORPH_RECT, blackhat_kernel_size)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_blackhat)

    _, thresh = cv2.threshold(blackhat, threshold_value, 255, cv2.THRESH_BINARY)

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, dilation_kernel_size)
    dilated_mask = cv2.dilate(thresh, kernel_dilate, iterations=dilation_iterations)

    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
    hair_count = len(significant_contours)

    output_path = os.path.join(output_dir, filename)

    if hair_count < min_hair_contours_to_process:
        # Not enough significant hairs detected
        return 0, output_path, "No significant hairs found, original image skipped."

    inpainted_image = cv2.inpaint(img, dilated_mask, inpaint_radius, cv2.INPAINT_TELEA)
    cv2.imwrite(output_path, inpainted_image)

    return hair_count, output_path, f"{hair_count} hairs removed."


def process_folder(input_folder, output_folder="output_cleaned"):
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    os.makedirs(output_folder, exist_ok=True)

    print(f"Processing images from: {input_folder}")
    print(f"Saving results to: {output_folder}")
    print("-" * 40)

    params = {
        "blackhat_kernel_size": (15, 15),
        "threshold_value": 18,
        "dilation_kernel_size": (3, 3),
        "dilation_iterations": 2,
        "inpaint_radius": 5,
        "min_hair_contours_to_process": 3,
        "min_contour_area": 15
    }

    processed = 0
    skipped = 0
    errored = 0
    total = 0

    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.lower().endswith(supported_extensions):
                total += 1
                image_path = os.path.join(root, filename)

                try:
                    count, out_path, msg = remove_and_save_hairs(
                        image_path=image_path,
                        output_dir=output_folder,
                        **params
                    )
                    if "No significant hairs found" in msg:
                        print(f"[SKIPPED] {filename}: {msg}")
                        skipped += 1
                    else:
                        print(f"[OK] {filename}: {msg}")
                        processed += 1
                except Exception as e:
                    print(f"[ERROR] {filename}: {e}")
                    errored += 1

    print("-" * 40)
    print(f"Total image files found: {total}")
    print(f"Processed: {processed}")
    print(f"Skipped (no/few hairs): {skipped}")
    print(f"Errors: {errored}")


# --- Run it ---
if __name__ == "__main__":
    input_folder = r"C:\Users\laura\Documents\University\2nd semester\Projects in Data Science\Projects\Final Project\matched_pairs\images"
    output_folder = r"C:\Users\laura\Documents\University\2nd semester\Projects in Data Science\Projects\Final Project\images after hair removal"
    process_folder(input_folder, output_folder)
