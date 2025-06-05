"""
Module principal de détection des panneaux de métro - VERSION AMÉLIORÉE
"""

import cv2
import numpy as np
import os
from .preprocessing import ImagePreprocessor
from .segmentation import MetroSegmenter
from .classification import AdvancedLineClassifier
from .constants import DEFAULT_PATHS


class MetroSignDetector:
    """Détecteur complet de panneaux de métro avec classificateur avancé"""
    
    def __init__(self, seg_params=None, class_params=None):
        """
        Initialise le détecteur avancé
        
        Args:
            seg_params (dict): Paramètres de segmentation
            class_params (dict): Paramètres de classification
        """
        self.preprocessor = ImagePreprocessor()
        self.segmenter = MetroSegmenter(seg_params)
        self.classifier = AdvancedLineClassifier(class_params)
        
        # Essayer de charger le modèle avancé en priorité
        advanced_model_path = 'models/advanced_digit_classifier.pkl'
        legacy_model_path = 'models/digit_classifier.pkl'
        
        if os.path.exists(advanced_model_path):
            print(f"📚 Chargement du modèle avancé...")
            self.classifier.load_model(advanced_model_path)
        elif os.path.exists(legacy_model_path):
            print(f"📚 Chargement du modèle classique...")
            self.classifier.load_model(legacy_model_path)
        else:
            print(f"📁 Aucun modèle pré-entraîné trouvé")
    
    def detect_signs(self, image_path):
        """
        Détecte tous les panneaux de métro dans une image
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            dict: {
                'detections': liste des détections,
                'processed_image': image traitée
            }
        """
        # Charger l'image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
        
        # Traitement sans redimensionnement (selon spécifications)
        prep_result = self.preprocessor.preprocess(image)
        lab_image = prep_result['processed']
        bgr_image = prep_result['bgr']
        
        # Segmentation
        rois = self.segmenter.segment(lab_image)
        
        # Classification avancée
        detections = []
        for roi_info in rois:
            xmin, ymin, xmax, ymax = roi_info['bbox']
            
            # Extraire la ROI en BGR pour la classification
            roi_bgr = bgr_image[ymin:ymax, xmin:xmax]
            
            if roi_bgr.size == 0:
                continue
            
            # FILTRES DE QUALITÉ POUR RÉDUIRE LES FAUX POSITIFS
            width = xmax - xmin
            height = ymax - ymin
            
            # 1. Filtre de taille (éviter très petites/grandes régions)
            if width < 25 or height < 25:
                continue  # Trop petit
            if width > 250 or height > 250:
                continue  # Trop grand
                
            # 2. Filtre de ratio aspect (numéros plutôt carrés)
            aspect_ratio = width / height if height > 0 else 0
            if aspect_ratio < 0.4 or aspect_ratio > 2.5:
                continue  # Ratio aberrant
            
            # Classification avancée
            class_result = self.classifier.classify(roi_bgr)
            
            # 3. SEUIL DE CONFIANCE PLUS STRICT
            if class_result['confidence'] < 0.35:  # Augmenté de 0.1 à 0.35
                continue
            
            # Créer la détection avec informations détaillées
            detection = {
                'bbox': (xmin, ymin, xmax, ymax),
                'line_num': class_result['line_num'],
                'confidence': class_result['confidence'],
                'color_prediction': class_result['color_prediction'],
                'digit_prediction': class_result['digit_prediction'],
                'ensemble_prediction': class_result.get('ensemble_prediction'),
                'all_scores': class_result.get('all_scores', {}),
                'quality_score': self._calculate_quality_score(width, height, aspect_ratio, class_result['confidence'])
            }
            
            detections.append(detection)
        
        # 4. POST-PROCESSING : garder seulement les meilleures détections
        detections = self._post_process_detections(detections)
        
        return {
            'detections': detections,
            'processed_image': bgr_image
        }
    
    def detect_batch(self, image_paths, progress_callback=None):
        """
        Détecte les panneaux dans un lot d'images avec statistiques
        
        Args:
            image_paths: Liste des chemins d'images
            progress_callback: Fonction appelée pour chaque image (optionnel)
            
        Returns:
            dict: Résultats par image avec statistiques globales
        """
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
                print(f"⚠️  Erreur sur {image_path}: {str(e)}")
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
    
    def train_advanced_classifier(self, training_data):
        """
        Entraîne le classificateur avancé
        
        Args:
            training_data: Liste de tuples (roi_image, line_num) - ROIs déjà extraites
        """
        print(f"🧠 Entraînement du classificateur avancé sur {len(training_data)} ROIs...")
        
        if training_data:
            # Entraîner directement sur les ROIs fournies
            self.classifier.train_advanced_classifier(training_data)
            
            # Sauvegarder le modèle avancé
            os.makedirs('models', exist_ok=True)
            model_path = 'models/advanced_digit_classifier.pkl'
            self.classifier.save_model(model_path)
            
            print(f"✅ Modèle avancé sauvegardé: {model_path}")
        else:
            print("❌ Aucune donnée d'entraînement fournie")
    
    def train_classifier_legacy(self, training_data):
        """
        Méthode d'entraînement legacy (pour compatibilité)
        
        Args:
            training_data: Liste de tuples (image_path, annotations)
        """
        print("🔄 Entraînement en mode legacy...")
        
        # Préparer les données d'entraînement
        train_samples = []
        
        for image_path, annotations in training_data:
            # Charger et prétraiter l'image
            image = cv2.imread(image_path)
            if image is None:
                continue
                
            prep_result = self.preprocessor.preprocess(image)
            bgr_image = prep_result['bgr']
            
            # Extraire les ROIs annotées (sans redimensionnement)
            for ann in annotations:
                xmin = ann['xmin']
                ymin = ann['ymin']
                xmax = ann['xmax']
                ymax = ann['ymax']
                line_num = ann['line']
                
                # Vérifier les coordonnées
                if xmax > xmin and ymax > ymin:
                    roi = bgr_image[ymin:ymax, xmin:xmax]
                    if roi.size > 0:
                        train_samples.append((roi, line_num))
        
        # Entraîner le classifieur avancé
        if train_samples:
            self.train_advanced_classifier(train_samples)
        else:
            print("❌ Aucun échantillon d'entraînement extrait")
    
    def visualize_detections(self, image_path, detections):
        """
        Visualise les détections sur l'image avec informations détaillées
        
        Args:
            image_path: Chemin de l'image originale
            detections: Liste des détections
            
        Returns:
            Image avec les détections dessinées
        """
        # Charger l'image originale
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Dessiner chaque détection
        for i, det in enumerate(detections):
            xmin, ymin, xmax, ymax = det['bbox']
            line_num = det['line_num']
            confidence = det['confidence']
            
            # Couleur de la boîte selon la confiance
            if confidence > 0.8:
                color = (0, 255, 0)  # Vert - confiance élevée
            elif confidence > 0.6:
                color = (0, 165, 255)  # Orange - confiance moyenne
            elif confidence > 0.4:
                color = (0, 255, 255)  # Jaune - confiance faible
            else:
                color = (0, 0, 255)  # Rouge - confiance très faible
            
            # Dessiner la boîte
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Préparer le texte d'information
            main_text = f"L{line_num} ({confidence:.2f})"
            
            # Informations détaillées
            info_lines = [main_text]
            if 'color_prediction' in det and det['color_prediction'][0] is not None:
                color_pred, color_conf = det['color_prediction']
                info_lines.append(f"C:{color_pred}({color_conf:.2f})")
            
            if 'digit_prediction' in det and det['digit_prediction'][0] is not None:
                digit_pred, digit_conf = det['digit_prediction']
                info_lines.append(f"D:{digit_pred}({digit_conf:.2f})")
            
            # Dessiner le texte
            y_offset = ymin - 10
            for j, text_line in enumerate(info_lines):
                text_y = max(15, y_offset - j * 20)
                cv2.putText(image, text_line, (xmin, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return image
    
    def get_detection_statistics(self, detections):
        """
        Calcule des statistiques sur les détections
        
        Args:
            detections: Liste des détections
            
        Returns:
            dict: Statistiques détaillées
        """
        if not detections:
            return {'total': 0}
        
        from collections import Counter
        
        # Statistiques de base
        line_counts = Counter([det['line_num'] for det in detections])
        confidences = [det['confidence'] for det in detections]
        
        # Statistiques de confiance
        high_conf = sum(1 for c in confidences if c > 0.8)
        medium_conf = sum(1 for c in confidences if 0.6 < c <= 0.8)
        low_conf = sum(1 for c in confidences if c <= 0.6)
        
        stats = {
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
        
        return stats 
    
    def _calculate_quality_score(self, width, height, aspect_ratio, confidence):
        """Calcule un score de qualité pour une détection"""
        # Score basé sur plusieurs critères
        size_score = 1.0
        
        # Pénaliser les tailles extrêmes
        ideal_size = 80  # Taille idéale estimée
        size_diff = abs(max(width, height) - ideal_size) / ideal_size
        size_score = max(0.1, 1.0 - size_diff * 0.5)
        
        # Score de ratio (idéal proche de 1.0)
        ratio_score = 1.0 - abs(aspect_ratio - 1.0) * 0.3
        ratio_score = max(0.1, ratio_score)
        
        # Score de confiance (déjà normalisé 0-1)
        conf_score = confidence
        
        # Score composite
        quality_score = (size_score * 0.3 + ratio_score * 0.3 + conf_score * 0.4)
        return min(1.0, quality_score)
    
    def _post_process_detections(self, detections):
        """Post-traitement pour réduire les faux positifs"""
        if not detections:
            return detections
        
        # 1. Supprimer les détections qui se chevauchent trop (NMS simplifié)
        filtered_detections = []
        
        # Trier par confiance décroissante
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        for detection in detections:
            bbox = detection['bbox']
            is_redundant = False
            
            for existing in filtered_detections:
                existing_bbox = existing['bbox']
                
                # Calculer IoU
                iou = self._calculate_bbox_iou(bbox, existing_bbox)
                
                # Si IoU > 0.5, c'est probablement la même région
                if iou > 0.5:
                    is_redundant = True
                    break
            
            if not is_redundant:
                filtered_detections.append(detection)
        
        # 2. Filtrer par score de qualité
        high_quality_detections = [
            det for det in filtered_detections 
            if det.get('quality_score', 0) > 0.4
        ]
        
        # 3. Limiter le nombre total (éviter trop de détections)
        max_detections = 8  # Max 8 panneaux par image
        if len(high_quality_detections) > max_detections:
            # Garder les meilleures
            high_quality_detections.sort(key=lambda x: x['confidence'], reverse=True)
            high_quality_detections = high_quality_detections[:max_detections]
        
        return high_quality_detections
    
    def _calculate_bbox_iou(self, bbox1, bbox2):
        """Calcule l'IoU entre deux bounding boxes"""
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