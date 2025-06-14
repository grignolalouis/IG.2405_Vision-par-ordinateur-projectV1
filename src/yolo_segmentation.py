"""
Module de segmentation YOLO pour la détection de panneaux de métro.

Ce module implémente la classe YOLOMetroSegmenter qui utilise le modèle YOLOv8n (You Only Look Once)
pour détecter et localiser les panneaux de métro dans les images. 

Auteur: LGrignola
"""

import torch
from ultralytics import YOLO
import numpy as np
import cv2
import os

class YOLOMetroSegmenter:
    def __init__(self, model_path=None, device=None, confidence_threshold=0.5):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        
        if model_path is None:
            trained_model_path = 'runs/detect/metro_detection/weights/best.pt'
            if os.path.exists(trained_model_path):
                model_path = trained_model_path
                print(f"Utilisation du modèle entraîné: {model_path}")
            else:
                model_path = 'models/yolov8n.pt'
                if os.path.exists(model_path):
                    print(f"Utilisation du modèle par défaut: {model_path}")
                else:
                    model_path = 'yolov8n.pt'
                    print(f"Modèle non trouvé dans models/, tentative depuis le répertoire racine: {model_path}")
        
        self.model_path = model_path
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle YOLO non trouvé: {model_path}")
        
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        print(f"Modèle YOLO initialisé sur {self.device}")

    def segment(self, image):
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("L'image doit être en couleur (3 canaux)")
            
        if image.shape[2] == 3:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image
            
        results = self.model(img_rgb, verbose=False)
        
        rois = []
        
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                confidence = float(boxes.conf[i].item())
                
                if confidence >= self.confidence_threshold:
                    h, w = image.shape[:2]
                    x1 = max(0, min(w-1, x1))
                    y1 = max(0, min(h-1, y1))
                    x2 = max(0, min(w-1, x2))
                    y2 = max(0, min(h-1, y2))
                    
                    if x2 > x1 and y2 > y1:
                        rois.append({
                            'bbox': (x1, y1, x2, y2),
                            'line_num_color': None,
                            'confidence': confidence
                        })
        
        return rois
    
    def get_model_info(self):
        return {
            'model_path': self.model_path,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'model_type': 'YOLOv8'
        }
    
    def set_confidence_threshold(self, threshold):
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        print(f"Seuil de confiance mis à jour: {self.confidence_threshold}")

def test_yolo_segmenter(image_path, model_path=None):
    if not os.path.exists(image_path):
        print(f"Image non trouvée: {image_path}")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Impossible de charger l'image: {image_path}")
        return
    
    segmenter = YOLOMetroSegmenter(model_path=model_path)
    
    rois = segmenter.segment(image)
    
    print(f"Détections trouvées: {len(rois)}")
    for i, roi in enumerate(rois):
        x1, y1, x2, y2 = roi['bbox']
        conf = roi['confidence']
        print(f"  ROI {i+1}: bbox=({x1}, {y1}, {x2}, {y2}), confidence={conf:.3f}")
    
    return rois 