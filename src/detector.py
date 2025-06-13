"""
Module principal de détection des panneaux de métro - VERSION AMÉLIORÉE AVEC YOLO
"""

import cv2
import numpy as np
import os
from collections import Counter
from src.preprocessing import ImagePreprocessor
from src.yolo_segmentation import YOLOMetroSegmenter
from src.classification import LineClassifier
from src.constants import DETECTOR_PARAMS


class MetroSignDetector:
    def __init__(self, seg_params=None, class_params=None, yolo_model_path=None):
        self.preprocessor = ImagePreprocessor()
        self.segmenter = YOLOMetroSegmenter(
            model_path=yolo_model_path,
            confidence_threshold=seg_params.get('confidence_threshold', 0.5) if seg_params else 0.5
        )
        self.classifier = LineClassifier(class_params)
        self.params = DETECTOR_PARAMS.copy()
        
        self._load_model()
    
    def _load_model(self):
        model_path = self.params['model_path']
        
        if os.path.exists(model_path):
            print(f"Chargement du modèle de classification...")
            self.classifier.load_model(model_path)
        else:
            print(f"Aucun modèle de classification pré-entraîné trouvé")
    
    def detect_signs(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
        
        prep_result = self.preprocessor.preprocess(image)
        bgr_image = prep_result['bgr']
        
        print(f"Détection YOLO sur {os.path.basename(image_path)}...")
        rois = self.segmenter.segment(bgr_image)
        print(f"YOLO a détecté {len(rois)} zones d'intérêt")
        
        detections = self._process_rois(rois, bgr_image)
        detections = self._post_process_detections(detections)
        
        return {
            'detections': detections,
            'processed_image': bgr_image
        }
    
    def detect_signs_from_array(self, image_array):
        """
        Détecter les panneaux de métro directement depuis un array numpy
        
        Args:
            image_array: np.ndarray - Image en format BGR
            
        Returns:
            dict: Résultats de détection avec même format que detect_signs
        """
        if image_array is None:
            raise ValueError("Image array est None")
        
        if len(image_array.shape) != 3 or image_array.shape[2] != 3:
            raise ValueError("L'image doit être en couleur (3 canaux)")
        
        # Préprocessing
        prep_result = self.preprocessor.preprocess(image_array)
        bgr_image = prep_result['bgr']
        
        # Segmentation YOLO
        rois = self.segmenter.segment(bgr_image)
        
        # Classification et post-processing
        detections = self._process_rois(rois, bgr_image)
        detections = self._post_process_detections(detections)
        
        return {
            'detections': detections,
            'processed_image': bgr_image
        }
    
    def _process_rois(self, rois, bgr_image):
        detections = []
        params = self.params
        
        for i, roi_info in enumerate(rois):
            xmin, ymin, xmax, ymax = roi_info['bbox']
            yolo_confidence = roi_info['confidence']
            
            roi_bgr = bgr_image[ymin:ymax, xmin:xmax]
            
            if roi_bgr.size == 0:
                continue
            
            width, height = xmax - xmin, ymax - ymin
            
            if not self._is_valid_roi(width, height):
                print(f"  ROI {i+1} rejetée: taille invalide ({width}x{height})")
                continue
            
            print(f"  Classification ROI {i+1} ({width}x{height})...")
            class_result = self.classifier.classify(roi_bgr)
            
            combined_confidence = (yolo_confidence * 0.6 + class_result['confidence'] * 0.4)
            
            if combined_confidence < params['min_confidence']:
                print(f"  ROI {i+1} rejetée: confiance trop faible ({combined_confidence:.3f})")
                continue
            
            aspect_ratio = width / height if height > 0 else 0
            quality_score = self._calculate_quality_score(width, height, aspect_ratio, combined_confidence)
            
            detection = {
                'bbox': (xmin, ymin, xmax, ymax),
                'line_num': class_result['line_num'],
                'confidence': combined_confidence,
                'yolo_confidence': yolo_confidence,
                'classification_confidence': class_result['confidence'],
                'color_prediction': class_result['color_prediction'],
                'digit_prediction': class_result['digit_prediction'],
                'ensemble_prediction': class_result.get('ensemble_prediction'),
                'all_scores': class_result.get('all_scores', {}),
                'quality_score': quality_score
            }
            
            detections.append(detection)
            print(f"  ROI {i+1} acceptée: Ligne {class_result['line_num']} (conf: {combined_confidence:.3f})")
        
        return detections
    
    def _is_valid_roi(self, width, height):
        params = self.params
        
        if width < params['min_roi_size'] or height < params['min_roi_size']:
            return False
        if width > params['max_roi_size'] or height > params['max_roi_size']:
            return False
        
        aspect_ratio = width / height if height > 0 else 0
        if aspect_ratio < params['min_aspect_ratio'] or aspect_ratio > params['max_aspect_ratio']:
            return False
        
        return True
    
    def detect_batch(self, image_paths, progress_callback=None):
        results = {}
        total_detections = 0
        processing_errors = 0
        
        for i, image_path in enumerate(image_paths):
            try:
                result = self.detect_signs(image_path)
                results[image_path] = result
                total_detections += len(result['detections'])
                
                if progress_callback:
                    progress_callback(i + 1, len(image_paths), image_path, result)
                    
            except Exception as e:
                print(f"Erreur sur {image_path}: {str(e)}")
                results[image_path] = {'error': str(e)}
                processing_errors += 1
        
        # Ajouter des statistiques globales
        results['_statistics'] = {
            'total_images': len(image_paths),
            'successful_images': len(image_paths) - processing_errors,
            'processing_errors': processing_errors,
            'total_detections': total_detections,
            'average_detections_per_image': total_detections / max(1, len(image_paths) - processing_errors)
        }
        
        return results
    
    def train_classifier(self, training_data):
        print(f"Entraînement du classificateur sur {len(training_data)} ROIs...")
        
        if training_data:
            self.classifier.train_advanced_classifier(training_data)
            
            os.makedirs('models', exist_ok=True)
            model_path = self.params['model_path']
            self.classifier.save_model(model_path)
            
            print(f"Modèle sauvegardé: {model_path}")
        else:
            print("Aucune donnée d'entraînement fournie")
    
    def visualize_detections(self, image_path, detections):
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        colors = self.params['visualization_colors']
        levels = self.params['confidence_levels']
        
        for det in detections:
            xmin, ymin, xmax, ymax = det['bbox']
            line_num = det['line_num']
            confidence = det['confidence']
            
            if confidence > levels['high']:
                color = colors['high']
            elif confidence > levels['medium']:
                color = colors['medium']
            elif confidence > levels['low']:
                color = colors['low']
            else:
                color = colors['poor']
            
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            
            info_lines = [f"L{line_num} ({confidence:.2f})"]
        
            if 'yolo_confidence' in det and 'classification_confidence' in det:
                yolo_conf = det['yolo_confidence']
                class_conf = det['classification_confidence']
                info_lines.append(f"YOLO:{yolo_conf:.2f} Class:{class_conf:.2f}")
            
            if 'color_prediction' in det and det['color_prediction'][0] is not None:
                color_pred, color_conf = det['color_prediction']
                info_lines.append(f"C:{color_pred}({color_conf:.2f})")
            
            if 'digit_prediction' in det and det['digit_prediction'][0] is not None:
                digit_pred, digit_conf = det['digit_prediction']
                info_lines.append(f"D:{digit_pred}({digit_conf:.2f})")
            
            y_offset = ymin - 10
            for j, text_line in enumerate(info_lines):
                text_y = max(15, y_offset - j * 20)
                cv2.putText(image, text_line, (xmin, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image
    
    def get_detection_statistics(self, detections):
        if not detections:
            return {'total': 0}
        
        line_counts = Counter([det['line_num'] for det in detections])
        confidences = [det['confidence'] for det in detections]
        levels = self.params['confidence_levels']

        high_conf = sum(1 for c in confidences if c > levels['high'])
        medium_conf = sum(1 for c in confidences if levels['medium'] < c <= levels['high'])
        low_conf = sum(1 for c in confidences if c <= levels['medium'])
        
        return {
            'total': len(detections),
            'line_distribution': dict(line_counts),
            'confidence_stats': {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences),
                'high_confidence': high_conf,
                'medium_confidence': medium_conf,
                'low_confidence': low_conf
            },
            'most_detected_line': line_counts.most_common(1)[0] if line_counts else None
        }
    
    def _calculate_quality_score(self, width, height, aspect_ratio, confidence):
        params = self.params
        ideal_size = params['ideal_size']
        
        size_diff = abs(max(width, height) - ideal_size) / ideal_size
        size_score = max(0.1, 1.0 - size_diff * 0.5)
        
        ratio_score = max(0.1, 1.0 - abs(aspect_ratio - 1.0) * 0.3)
        
        quality_score = (size_score * 0.3 + ratio_score * 0.3 + confidence * 0.4)
        return min(1.0, quality_score)
    
    def _post_process_detections(self, detections):
        if not detections:
            return detections
        
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        filtered_detections = []
        iou_threshold = self.params['nms_iou_threshold']
        
        for detection in detections:
            bbox = detection['bbox']
            is_redundant = False
            
            for existing in filtered_detections:
                if self._calculate_bbox_iou(bbox, existing['bbox']) > iou_threshold:
                    is_redundant = True
                    break
            
            if not is_redundant:
                filtered_detections.append(detection)
        
        high_quality_detections = [
            det for det in filtered_detections 
            if det.get('quality_score', 0) > self.params['quality_threshold']
        ]
        
        max_detections = self.params['max_detections_per_image']
        if len(high_quality_detections) > max_detections:
            high_quality_detections.sort(key=lambda x: x['confidence'], reverse=True)
            high_quality_detections = high_quality_detections[:max_detections]
        
        return high_quality_detections
    
    def _calculate_bbox_iou(self, bbox1, bbox2):
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1 = max(x1_1, x1_2)
        y1 = max(y1_1, y1_2)
        x2 = min(x2_1, x2_2)
        y2 = min(y2_1, y2_2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0 