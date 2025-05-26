#Haralick 

import cv2
import mahotas
import numpy as np

def extract_haralick_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize for consistency
    gray = cv2.resize(gray, (256, 256))
    # Compute Haralick texture features
    features = mahotas.features.haralick(gray).mean(axis=0)
    return features
