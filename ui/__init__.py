"""
Package UI pour l'interface utilisateur du projet de détection de panneaux métro.

Ce package contient :
- gui_main.py : Interface utilisateur principale
- image_renderer.py : Rendu visuel des images avec bounding boxes
- metrics_formatter.py : Formatage des métriques d'affichage
- workflow_manager.py : Orchestration des flux de travail
- data_processor.py : Calculs et analyses de données
"""

from .gui_main import MetroProjectMainGUI
from .image_renderer import ImageRenderer
from .metrics_formatter import MetricsFormatter
from .workflow_manager import WorkflowManager
from .data_processor import DataProcessor

__all__ = [
    'MetroProjectMainGUI',
    'ImageRenderer', 
    'MetricsFormatter',
    'WorkflowManager',
    'DataProcessor'
] 