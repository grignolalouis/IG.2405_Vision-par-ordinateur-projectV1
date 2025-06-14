"""
Module de traitement et export des données pour l'interface utilisateur.

Ce module implémente la classe DataProcessor qui gère le traitement des données de
performance, le chargement des vérités terrain et l'export des résultats vers
différents formats. Il fait le lien entre les données brutes et l'interface utilisateur.

Auteur: LGrignola
"""

import os
import numpy as np
import scipy.io as sio
import re
from src.constants import METRO_COLORS, GUI_PARAMS
from .evaluation import MetricsEvaluator

class DataProcessor:
    def __init__(self):
        self.evaluator = MetricsEvaluator()
    
    def calculate_performance_metrics(self, test_images, ground_truth, predictions):
        return self.evaluator.calculate_performance_metrics(test_images, ground_truth, predictions)
    
    def load_ground_truth_file(self, gt_file_path, original_images):
        try:
            data = sio.loadmat(gt_file_path)
            
            if 'BD' in data:
                annotations = data['BD']
            else:
                for key, value in data.items():
                    if not key.startswith('__') and isinstance(value, np.ndarray) and value.ndim == 2:
                        annotations = value
                        break
                else:
                    raise ValueError("Format de fichier MAT non reconnu")
            
            ground_truth = {}
            
            for ann in annotations:
                if len(ann) >= 6:
                    image_id = int(ann[0])
                    y1, y2, x1, x2 = map(int, ann[1:5])
                    line_num = int(ann[5])
                    
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
            
            match = re.search(r'(\d+)', image_name)
            if match and int(match.group(1)) == image_id:
                return image_name
        
        return None
    
    def convert_coordinates_after_resize(self, predictions, resize_factor):
        if resize_factor == 1.0:
            return predictions
        
        converted_predictions = {}
        
        for image_name, pred_list in predictions.items():
            converted_predictions[image_name] = []
            
            for pred in pred_list:
                converted_pred = pred.copy()
                
                converted_pred['xmin'] = int(pred['xmin'] / resize_factor)
                converted_pred['ymin'] = int(pred['ymin'] / resize_factor)
                converted_pred['xmax'] = int(pred['xmax'] / resize_factor)
                converted_pred['ymax'] = int(pred['ymax'] / resize_factor)
                
                converted_predictions[image_name].append(converted_pred)
        
        return converted_predictions
    
    def get_current_image_info(self, current_image_path, ground_truth, predictions):
        return self.evaluator.get_current_image_info(current_image_path, ground_truth, predictions)
    
    def extract_image_id_from_filename(self, filename):
        patterns = [
            r'IM \((\d+)\)\.JPG',
            r'IM\((\d+)\)\.JPG', 
            r'metro(\d+)\.jpg',
            r'image(\d+)\.jpg',
            r'(\d+)\.jpg',
            r'(\d+)\.JPG',
            r'(\d+)\.png',
            r'(\d+)\.jpeg'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(1))
        
        return 1
    
    def export_predictions_to_mat(self, challenge_test_images, challenge_predictions, file_path):
        try:
            import scipy.io as sio
            
            predictions_list = []
            
            for image_path in challenge_test_images:
                image_name = os.path.basename(image_path)
                image_id = self.extract_image_id_from_filename(image_name)
                pred_boxes = challenge_predictions.get(image_name, [])
                
                for pred in pred_boxes:
                    prediction_row = [
                        image_id,
                        pred['ymin'],
                        pred['ymax'], 
                        pred['xmin'],
                        pred['xmax'],
                        pred['line']
                    ]
                    predictions_list.append(prediction_row)
            
            if predictions_list:
                predictions_array = np.array(predictions_list, dtype=np.int32)
            else:
                predictions_array = np.empty((0, 6), dtype=np.int32)
            
            mat_data = {
                'BD': predictions_array
            }
            
            sio.savemat(file_path, mat_data)
            
            return {
                'success': True,
                'images_count': len(challenge_test_images),
                'predictions_count': len(predictions_list)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def export_results_to_txt(self, performance_metrics, challenge_test_images, challenge_predictions, file_path):
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("=== TEAM 12 - RESULTATS CHALLENGE METRO ===\n\n")
                
                f.write("INFORMATIONS GENERALES:\n")
                f.write(f"Images traitées: {len(challenge_test_images)}\n")
                total_predictions = sum(len(preds) for preds in challenge_predictions.values())
                f.write(f"Détections totales: {total_predictions}\n")
                f.write(f"Moyenne par image: {total_predictions/len(challenge_test_images):.2f}\n\n")
                
                if performance_metrics:
                    detection = performance_metrics['detection']
                    classification = performance_metrics['classification']
                    totals = performance_metrics['totals']
                    
                    f.write("METRIQUES DE DETECTION:\n")
                    f.write(f"Precision: {detection['precision']:.3f}\n")
                    f.write(f"Rappel: {detection['recall']:.3f}\n")
                    f.write(f"F1-Score: {detection['f1']:.3f}\n")
                    f.write(f"Accuracy: {detection['accuracy']:.3f}\n")
                    f.write(f"TP: {detection['tp']}, FP: {detection['fp']}, FN: {detection['fn']}\n\n")
                    
                    f.write("METRIQUES DE CLASSIFICATION:\n")
                    f.write(f"Precision: {classification['precision']:.3f}\n")
                    f.write(f"Rappel: {classification['recall']:.3f}\n")
                    f.write(f"F1-Score: {classification['f1']:.3f}\n")
                    f.write(f"Accuracy: {classification['accuracy']:.3f}\n")
                    f.write(f"TP: {classification['tp']}, FP: {classification['fp']}, FN: {classification['fn']}\n\n")
                    
                    if 'macro' in performance_metrics and 'weighted' in performance_metrics:
                        macro = performance_metrics['macro']
                        weighted = performance_metrics['weighted']
                        
                        f.write("MOYENNES:\n")
                        f.write(f"Macro     - P: {macro['precision']:.3f}, R: {macro['recall']:.3f}, F1: {macro['f1']:.3f}, A: {macro['accuracy']:.3f} ({macro['classes']} classes)\n")
                        f.write(f"Weighted  - P: {weighted['precision']:.3f}, R: {weighted['recall']:.3f}, F1: {weighted['f1']:.3f}, A: {weighted['accuracy']:.3f} ({weighted['support']} signs)\n\n")
                    
                    f.write("PERFORMANCE PAR LIGNE:\n")
                    f.write("Ligne | Precision | Recall | F1-Score | Accuracy | Support\n")
                    f.write("------|-----------|--------|----------|----------|--------\n")
                    
                    by_line = performance_metrics['by_line']
                    for line_num in sorted(by_line.keys()):
                        if by_line[line_num]['gt_count'] > 0 or by_line[line_num]['pred_count'] > 0:
                            stats = by_line[line_num]
                            f.write(f"{line_num:5d} | {stats['precision']:9.3f} | {stats['recall']:6.3f} | {stats['f1']:8.3f} | {stats['accuracy']:8.3f} | {stats['gt_count']:7d}\n")
                    
                    f.write(f"\nTOTAUX:\n")
                    f.write(f"Boîtes vérité terrain: {totals['gt_boxes']}\n")
                    f.write(f"Boîtes prédites: {totals['pred_boxes']}\n")
                
                else:
                    f.write("AUCUNE METRIQUE DISPONIBLE (pas de vérités terrain)\n\n")
                
                f.write(f"\n=== FIN RAPPORT TEAM 12 ===\n")
            
            return {
                'success': True,
                'images_count': len(challenge_test_images),
                'predictions_count': total_predictions
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            } 