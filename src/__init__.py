from .constants import METRO_COLORS, SEG_PARAMS, CLASS_PARAMS, DEFAULT_PATHS
from .preprocessing import ImagePreprocessor
from .segmentation import MetroSegmenter
from .classification import LineClassifier
from .detector import MetroSignDetector
from .data_loader import DataLoader

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