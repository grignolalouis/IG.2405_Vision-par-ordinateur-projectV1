"""
Module de prétraitement des images
"""

import cv2
import numpy as np
from .constants import SEG_PARAMS


class ImagePreprocessor:
    """Classe pour prétraiter les images avant segmentation"""
    
    def __init__(self, params=None):
        """
        Initialise le préprocesseur
        
        Args:
            params (dict): Paramètres personnalisés (optionnel)
        """
        self.params = SEG_PARAMS.copy()
        if params:
            self.params.update(params)
    
    def apply_bilateral_filter(self, image):
        """
        Applique un filtre bilatéral pour réduire le bruit
        
        Args:
            image: Image BGR
            
        Returns:
            Image filtrée
        """
        return cv2.bilateralFilter(
            image,
            self.params['bilateral_d'],
            self.params['bilateral_sigma_color'],
            self.params['bilateral_sigma_space']
        )
    
    def convert_to_lab(self, image):
        """
        Convertit l'image en espace Lab
        
        Args:
            image: Image BGR
            
        Returns:
            Image Lab
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    
    def preprocess(self, image):
        """
        Pipeline complet de prétraitement SANS redimensionnement
        Travaille sur l'image originale pour préserver les coordonnées exactes
        
        Args:
            image: Image BGR originale
            
        Returns:
            dict: {
                'processed': image prétraitée Lab,
                'bgr': image BGR originale,
                'scale_factor': 1.0 (pas de redimensionnement)
            }
        """
        # Pas de redimensionnement - conserver l'image originale
        original_bgr = image.copy()
        
        # Filtrer pour réduire le bruit tout en préservant les contours
        filtered = self.apply_bilateral_filter(original_bgr)
        
        # Convertir en Lab pour une meilleure segmentation couleur
        lab = self.convert_to_lab(filtered)
        
        return {
            'processed': lab,
            'bgr': original_bgr,
            'scale_factor': 1.0  # Pas de redimensionnement
        } 