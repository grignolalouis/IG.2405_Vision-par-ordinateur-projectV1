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
    
    def preprocess(self, image):
        original_bgr = image.copy()
        filtered = self.apply_bilateral_filter(original_bgr)
        
        return {
            'bgr': filtered,
            'scale_factor': 1.0
        } 