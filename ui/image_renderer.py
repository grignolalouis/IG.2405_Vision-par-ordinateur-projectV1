"""
Module de rendu visuel des images avec bounding boxes.
Sépare la logique de visualisation de l'interface utilisateur.
"""

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from src.constants import METRO_COLORS, GUI_PARAMS


class ImageRenderer:
    """Classe responsable du rendu visuel des images avec annotations."""
    
    def __init__(self):
        self.max_image_size = GUI_PARAMS['max_image_size']
        self.colors = GUI_PARAMS['colors']
        self.iou_threshold = GUI_PARAMS['iou_threshold']
    
    def render_image_with_boxes(self, image, ground_truth_boxes=None, prediction_boxes=None, 
                               display_mode="comparison"):
        """
        Rendre une image avec les bounding boxes selon le mode d'affichage.
        
        Args:
            image: Image OpenCV (BGR)
            ground_truth_boxes: Liste des boîtes de vérité terrain
            prediction_boxes: Liste des boîtes de prédiction
            display_mode: "gt", "pred", ou "comparison"
            
        Returns:
            PIL Image prête pour affichage Tkinter
        """
        if image is None:
            return None
            
        display_img = image.copy()
        
        # Dessiner selon le mode
        if display_mode == "gt" or display_mode == "comparison":
            if ground_truth_boxes:
                display_img = self._draw_ground_truth_boxes(display_img, ground_truth_boxes)
        
        if display_mode == "pred" or display_mode == "comparison":
            if prediction_boxes:
                display_img = self._draw_prediction_boxes(
                    display_img, prediction_boxes, ground_truth_boxes, display_mode
                )
        
        # Convertir et redimensionner pour affichage
        return self._prepare_for_display(display_img)
    
    def _draw_ground_truth_boxes(self, image, gt_boxes):
        """Dessiner les boîtes de vérité terrain."""
        for gt in gt_boxes:
            cv2.rectangle(image, 
                         (gt['xmin'], gt['ymin']), 
                         (gt['xmax'], gt['ymax']), 
                         self.colors['gt'], 2)
            cv2.putText(image, f"GT: {gt['line']}", 
                       (gt['xmin'], gt['ymin']-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['gt'], 2)
        return image
    
    def _draw_prediction_boxes(self, image, pred_boxes, gt_boxes=None, display_mode="pred"):
        """Dessiner les boîtes de prédiction avec classification TP/FP/WC."""
        for i, pred in enumerate(pred_boxes):
            color = self.colors['pred_fp']
            label_suffix = " (FP)" 
            thickness = 2
            
            if display_mode == "comparison" and gt_boxes:
                best_iou, matched_gt = self._find_best_matching_gt(pred, gt_boxes)
                
                if best_iou > self.iou_threshold: 
                    if pred['line'] == matched_gt['line']:
                        color = self.colors['pred_tp']
                        label_suffix = f" (TP, IoU={best_iou:.2f})"
                        thickness = 3
                    else:
                        color = self.colors['pred_wc']
                        label_suffix = f" (WC, IoU={best_iou:.2f})" 
                else:
                    label_suffix = f" (FP, IoU={best_iou:.2f})"
            else:
                color = self.colors['pred_tp']
                label_suffix = ""
                thickness = 3
            
            # Dessiner le rectangle
            cv2.rectangle(image, 
                         (pred['xmin'], pred['ymin']), 
                         (pred['xmax'], pred['ymax']), 
                         color, thickness)

            # Ajouter le texte
            text = f"P{i+1}: L{pred['line']} ({pred['confidence']:.2f}){label_suffix}"
            cv2.putText(image, text, 
                       (pred['xmin'], pred['ymax']+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image
    
    def _find_best_matching_gt(self, pred_box, gt_boxes):
        """Trouver la meilleure boîte GT correspondante."""
        best_iou = 0
        matched_gt = None
        
        for gt in gt_boxes:
            iou = self._calculate_iou(pred_box, gt)
            if iou > best_iou:
                best_iou = iou
                matched_gt = gt
        
        return best_iou, matched_gt
    
    def _calculate_iou(self, box1, box2):
        """Calculer l'IoU entre deux boîtes."""
        x1 = max(box1['xmin'], box2['xmin'])
        y1 = max(box1['ymin'], box2['ymin'])
        x2 = min(box1['xmax'], box2['xmax'])
        y2 = min(box1['ymax'], box2['ymax'])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1['xmax'] - box1['xmin']) * (box1['ymax'] - box1['ymin'])
        area2 = (box2['xmax'] - box2['xmin']) * (box2['ymax'] - box2['ymin'])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _prepare_for_display(self, image):
        """Préparer l'image pour l'affichage Tkinter."""
        # Convertir BGR vers RGB
        display_img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Redimensionner si nécessaire
        h, w = display_img_rgb.shape[:2]
        if max(h, w) > self.max_image_size:
            scale = self.max_image_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            display_img_rgb = cv2.resize(display_img_rgb, (new_w, new_h))
        
        # Convertir en PIL Image puis PhotoImage
        pil_img = Image.fromarray(display_img_rgb)
        return ImageTk.PhotoImage(pil_img)
    
    def get_line_color_info(self, line_num):
        """Obtenir les informations de couleur pour une ligne de métro."""
        return METRO_COLORS.get(line_num, {'name': 'Inconnue', 'color': '#808080'}) 