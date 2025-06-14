"""
Module d'initialisation du package de reconnaissance automatique des lignes de métro parisien.

Ce module expose les classes et constantes principales du système de détection et classification
des panneaux de métro. Il centralise les imports pour faciliter l'utilisation du package et
définit les métadonnées du projet.

Auteur: LGrignola
Version: 1.0.0
"""

from .constants import METRO_COLORS, SEG_PARAMS, CLASS_PARAMS, DEFAULT_PATHS
from .preprocessing import ImagePreprocessor
from .classification import LineClassifier
from .detector import MetroSignDetector
from .data_loader import DataLoader
from .yolo_segmentation import YOLOMetroSegmenter

__version__ = "1.0.0"
__author__ = "LGrignola"
__description__ = "Système de reconnaissance automatique des lignes de métro parisien"

__all__ = [
    'METRO_COLORS',
    'SEG_PARAMS',
    'CLASS_PARAMS', 
    'DEFAULT_PATHS',
    'ImagePreprocessor',
    'MetroSegmenter',
    'LineClassifier',
    'MetroSignDetector',
    'DataLoader'
] 