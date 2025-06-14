"""
Module d'interface utilisateur pour le système de détection de panneaux de métro parisien.

Ce module expose les composants principaux de l'interface graphique Tkinter qui permettent
l'interaction avec le système de détection et classification des panneaux de métro.

Auteur: LGrignola
Version: 1.0.0
"""

from .gui_main import MetroProjectMainGUI
from .image_renderer import ImageRenderer
from .metrics_formatter import MetricsFormatter
from .workflow_manager import WorkflowManager
from .data_processor import DataProcessor
from .dialog_manager import DialogManager

__all__ = [
    'MetroProjectMainGUI',
    'ImageRenderer', 
    'MetricsFormatter',
    'WorkflowManager',
    'DataProcessor',
    'DialogManager'
] 