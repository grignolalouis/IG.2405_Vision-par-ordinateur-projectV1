"""
Module de traitement et d'analyse des données.
Sépare la logique de calcul de l'interface utilisateur.
"""

import os
import numpy as np
import scipy.io as sio
import re
from src.constants import METRO_COLORS, GUI_PARAMS


class DataProcessor:
    """Classe responsable du traitement et de l'analyse des données."""
    
    def __init__(self):
        self.iou_threshold = GUI_PARAMS['iou_threshold']
    
    def calculate_performance_metrics(self, test_images, ground_truth, predictions):
        """
        Calculer les métriques de performance complètes.
        
        Args:
            test_images: Liste des chemins d'images de test
            ground_truth: Dictionnaire des vérités terrain
            predictions: Dictionnaire des prédictions
            
        Returns:
            Dictionnaire des métriques de performance
        """
        tp = fp = fn = 0
        correct_classifications = 0
        total_detections = 0
        line_stats = {line: {'tp': 0, 'fp': 0, 'fn': 0} for line in METRO_COLORS.keys()}
        
        for image_path in test_images:
            image_name_base = os.path.basename(image_path)
            gt_boxes = ground_truth.get(image_name_base, [])
            pred_boxes = predictions.get(image_name_base, [])
    
            matched_gt = set()
            matched_pred = set()
            
            for i, pred in enumerate(pred_boxes):
                best_iou = 0
                best_gt_idx = -1
                
                for j, gt in enumerate(gt_boxes):
                    if j in matched_gt:
                        continue
                    
                    iou = self.calculate_iou(pred, gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
                
                if best_iou > self.iou_threshold:
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(i)
                    tp += 1
                    
                    if pred['line'] == gt_boxes[best_gt_idx]['line']:
                        correct_classifications += 1
                        line_stats[pred['line']]['tp'] += 1
                    else:
                        line_stats[pred['line']]['fp'] += 1
                        line_stats[gt_boxes[best_gt_idx]['line']]['fn'] += 1
                else:
                    fp += 1
                    line_stats[pred['line']]['fp'] += 1
                
                total_detections += 1
            
            fn += len(gt_boxes) - len(matched_gt)
            for j, gt in enumerate(gt_boxes):
                if j not in matched_gt:
                    line_stats[gt['line']]['fn'] += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = correct_classifications / total_detections if total_detections > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'total_gt': sum(len(boxes) for boxes in ground_truth.values()),
            'total_pred': sum(len(boxes) for boxes in predictions.values()),
            'line_stats': line_stats
        }
    
    def calculate_iou(self, box1, box2):
        """
        Calculer l'IoU (Intersection over Union) entre deux boîtes.
        
        Args:
            box1, box2: Dictionnaires avec clés 'xmin', 'ymin', 'xmax', 'ymax'
            
        Returns:
            Valeur IoU entre 0 et 1
        """
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
    
    def load_ground_truth_file(self, gt_file_path, original_images):
        """
        Charger un fichier de vérités terrain au format MAT.
        
        Args:
            gt_file_path: Chemin vers le fichier .mat
            original_images: Liste des chemins d'images originales
            
        Returns:
            Dictionnaire des vérités terrain {nom_image: [annotations]}
        """
        try:
            data = sio.loadmat(gt_file_path)
            
            # Supposer le format standard: [image_id, y1, y2, x1, x2, line_num]
            if 'BD' in data:
                annotations = data['BD']
            else:
                # Essayer de trouver la première variable qui ressemble à des annotations
                for key, value in data.items():
                    if not key.startswith('__') and isinstance(value, np.ndarray) and value.ndim == 2:
                        annotations = value
                        break
                else:
                    raise ValueError("Format de fichier MAT non reconnu")
            
            ground_truth = {}
            
            # Traiter les annotations
            for ann in annotations:
                if len(ann) >= 6:
                    image_id = int(ann[0])
                    y1, y2, x1, x2 = map(int, ann[1:5])
                    line_num = int(ann[5])
                    
                    # Trouver le nom de l'image correspondant à l'ID
                    image_name = self._find_image_name_by_id(image_id, original_images)
                    if image_name:
                        if image_name not in ground_truth:
                            ground_truth[image_name] = []
                        
                        ground_truth[image_name].append({
                            'xmin': x1,
                            'ymin': y1,
                            'xmax': x2,
                            'ymax': y2,
                            'line': line_num
                        })
            
            return ground_truth
            
        except Exception as e:
            raise Exception(f"Erreur chargement vérités terrain: {str(e)}")
    
    def _find_image_name_by_id(self, image_id, original_images):
        """
        Trouver le nom d'image correspondant à un ID.
        
        Args:
            image_id: ID numérique de l'image
            original_images: Liste des chemins d'images
            
        Returns:
            Nom de l'image correspondante ou None
        """
        # Essayer différents formats de nommage
        patterns = [
            f"IM ({image_id}).JPG",
            f"IM({image_id}).JPG", 
            f"metro{image_id:03d}.jpg",
            f"image{image_id}.jpg",
            f"{image_id}.jpg",
            f"{image_id}.JPG"
        ]
        
        for original_path in original_images:
            image_name = os.path.basename(original_path)
            if image_name in patterns:
                return image_name
            
            # Essayer d'extraire l'ID du nom de fichier
            match = re.search(r'(\d+)', image_name)
            if match and int(match.group(1)) == image_id:
                return image_name
        
        return None
    
    def convert_coordinates_after_resize(self, predictions, resize_factor):
        """
        Convertir les coordonnées après redimensionnement vers l'espace original.
        
        Args:
            predictions: Dictionnaire des prédictions avec coordonnées redimensionnées
            resize_factor: Facteur de redimensionnement appliqué
            
        Returns:
            Dictionnaire des prédictions avec coordonnées converties
        """
        if resize_factor == 1.0:
            return predictions
        
        converted_predictions = {}
        
        for image_name, pred_list in predictions.items():
            converted_predictions[image_name] = []
            
            for pred in pred_list:
                converted_pred = pred.copy()
                
                # Reconvertir les coordonnées
                converted_pred['xmin'] = int(pred['xmin'] / resize_factor)
                converted_pred['ymin'] = int(pred['ymin'] / resize_factor)
                converted_pred['xmax'] = int(pred['xmax'] / resize_factor)
                converted_pred['ymax'] = int(pred['ymax'] / resize_factor)
                
                converted_predictions[image_name].append(converted_pred)
        
        return converted_predictions
    
    def get_current_image_info(self, current_image_path, ground_truth, predictions):
        """
        Obtenir les informations sur l'image courante pour l'affichage.
        
        Args:
            current_image_path: Chemin de l'image courante
            ground_truth: Dictionnaire des vérités terrain
            predictions: Dictionnaire des prédictions
            
        Returns:
            Dictionnaire avec les informations de l'image courante
        """
        if not current_image_path:
            return None
        
        image_name = os.path.basename(current_image_path)
        gt_boxes = ground_truth.get(image_name, [])
        pred_boxes = predictions.get(image_name, [])
        
        avg_confidence = None
        if pred_boxes:
            avg_confidence = sum(p['confidence'] for p in pred_boxes) / len(pred_boxes)
        
        return {
            'name': image_name,
            'gt_count': len(gt_boxes),
            'pred_count': len(pred_boxes),
            'avg_confidence': avg_confidence
        } 