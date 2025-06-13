import cv2
import numpy as np
from src.constants import SEG_PARAMS


class ImagePreprocessor:
    def __init__(self, params=None):
        
        self.params = SEG_PARAMS.copy()
        if params:
            self.params.update(params)
    
    def apply_bilateral_filter(self, image):
        return cv2.bilateralFilter(
            image,
            self.params['bilateral_d'],
            self.params['bilateral_sigma_color'],
            self.params['bilateral_sigma_space']
        )
    
    def convert_to_lab(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    
    def preprocess(self, image):
        original_bgr = image.copy()
        filtered = self.apply_bilateral_filter(original_bgr)
        lab = self.convert_to_lab(filtered)
        
        return {
            'processed': lab,
            'bgr': original_bgr,
            'scale_factor': 1.0
        } 