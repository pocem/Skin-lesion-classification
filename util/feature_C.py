# util/feature_C.py
import cv2
import numpy as np

def extract_feature_C(image_path):
    """
    Extract edge density feature using Canny edge detection
    
    Args:
        image_path: path to the image file
        
    Returns:
        float: edge density (ratio of edge pixels to total pixels)
    """
    # Read image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Detect edges
    edges = cv2.Canny(img, 100, 200)
    
    # Calculate edge density
    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.shape[0] * edges.shape[1]
    density = edge_pixels / total_pixels
    
    return np.array([density])
