#blue veil

def detect_blue_white_veil(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define blue hue range
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Define white veil (low saturation, high value)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Combine masks
    combined_mask = cv2.bitwise_and(mask_blue, mask_white)
    result = cv2.bitwise_and(image, image, mask=combined_mask)

    # Compute percentage of lesion covered by BWV
    bwv_area = np.sum(combined_mask > 0)
    total_area = image.shape[0] * image.shape[1]
    bwv_ratio = bwv_area / total_area
    return bwv_ratio, result
