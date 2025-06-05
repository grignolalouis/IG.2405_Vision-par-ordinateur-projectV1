#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Package principal pour la reconnaissance des lignes de métro parisien

Ce package contient tous les modules nécessaires pour :
1. Le prétraitement des images
2. La segmentation des zones d'intérêt
3. La classification des lignes de métro
4. La détection complète des panneaux
"""

# Importer les modules que nous avons créés
from .constants import METRO_COLORS, SEG_PARAMS, CLASS_PARAMS, DEFAULT_PATHS
from .preprocessing import ImagePreprocessor
from .segmentation import MetroSegmenter
from .classification import LineClassifier
from .detector import MetroSignDetector
from .data_loader import DataLoader

__version__ = "1.0.0"
__author__ = "Équipe IG.2405"
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