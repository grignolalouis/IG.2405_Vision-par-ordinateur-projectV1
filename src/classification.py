"""
Module de classification des numéros de ligne - VERSION AMÉLIORÉE
Utilise des techniques avancées pour une meilleure précision
"""

import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from skimage.feature import hog, local_binary_pattern
import pickle
import os
from collections import Counter
from .constants import CLASS_PARAMS, METRO_COLORS


class AdvancedLineClassifier:
    """Classificateur avancé pour les numéros de ligne de métro"""
    
    def __init__(self, params=None):
        """
        Initialise le classificateur avancé
        
        Args:
            params (dict): Paramètres personnalisés (optionnel)
        """
        self.params = CLASS_PARAMS.copy()
        if params:
            self.params.update(params)
            
        # Modèles d'ensemble
        self.color_classifier = None
        self.digit_classifier = None
        self.ensemble_classifier = None
        
        # Scalers
        self.color_scaler = StandardScaler()
        self.digit_scaler = StandardScaler()
        self.ensemble_scaler = StandardScaler()
        
        # État de l'entraînement
        self.is_trained = False
        self.color_references = self._compute_color_references()
        
        # Statistiques d'entraînement
        self.training_stats = {}
        
    def _compute_color_references(self):
        """Calcule les références de couleur dans plusieurs espaces"""
        references = {}
        
        for line_num, color_info in METRO_COLORS.items():
            rgb = np.array(color_info['rgb'], dtype=np.uint8).reshape(1, 1, 3)
            
            # Convertir dans différents espaces colorimétriques
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
            
            references[line_num] = {
                'rgb': color_info['rgb'],
                'hsv': hsv[0, 0],
                'lab': lab[0, 0]
            }
        
        return references
    
    def extract_advanced_color_features(self, roi_image):
        """
        Extrait des caractéristiques de couleur avancées
        
        Args:
            roi_image: Image BGR de la ROI
            
        Returns:
            np.array: Vecteur de caractéristiques de couleur
        """
        h, w = roi_image.shape[:2]
        features = []
        
        # Créer des masques pour différentes zones
        center_mask = np.zeros((h, w), dtype=np.uint8)
        ring_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Zone centrale (logo/chiffre)
        cv2.circle(center_mask, (w//2, h//2), int(min(w, h) * 0.3), 255, -1)
        
        # Zone anneau (couleur de fond)
        cv2.circle(ring_mask, (w//2, h//2), int(min(w, h) * 0.45), 255, -1)
        cv2.circle(ring_mask, (w//2, h//2), int(min(w, h) * 0.3), 0, -1)
        
        # Convertir dans différents espaces colorimétriques
        hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(roi_image, cv2.COLOR_BGR2Lab)
        
        # Extraire des statistiques pour chaque zone et espace colorimétrique
        for space, image in [('bgr', roi_image), ('hsv', hsv), ('lab', lab)]:
            for zone_name, mask in [('center', center_mask), ('ring', ring_mask)]:
                if np.sum(mask) > 0:
                    for channel in range(3):
                        channel_data = image[:, :, channel][mask > 0]
                        if len(channel_data) > 0:
                            # Statistiques de base
                            features.extend([
                                np.mean(channel_data),
                                np.std(channel_data),
                                np.median(channel_data),
                                np.percentile(channel_data, 25),
                                np.percentile(channel_data, 75)
                            ])
                        else:
                            features.extend([0, 0, 0, 0, 0])
        
        # Histogrammes de couleur pour la zone anneau
        if np.sum(ring_mask) > 0:
            for i, (space, image) in enumerate([('hsv', hsv), ('lab', lab)]):
                for channel in range(3):
                    hist = cv2.calcHist([image], [channel], ring_mask, [16], [0, 256])
                    features.extend(hist.flatten())
        
        return np.array(features, dtype=np.float32)
    
    def extract_advanced_digit_features(self, roi_image):
        """
        Extrait des caractéristiques avancées pour les chiffres
        
        Args:
            roi_image: Image BGR de la ROI
            
        Returns:
            np.array: Vecteur de caractéristiques combinées
        """
        # Prétraitement amélioré
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        
        # Extraire la zone centrale
        h, w = gray.shape
        center_h, center_w = int(h * 0.6), int(w * 0.6)
        y_start, x_start = (h - center_h) // 2, (w - center_w) // 2
        center = gray[y_start:y_start+center_h, x_start:x_start+center_w]
        
        # Améliorer le contraste
        center = cv2.equalizeHist(center)
        
        # Binarisation multiple
        binary_otsu = cv2.threshold(center, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        binary_adaptive = cv2.adaptiveThreshold(center, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY, 11, 2)
        
        # Choisir la meilleure binarisation
        if np.sum(binary_otsu == 255) > np.sum(binary_adaptive == 255):
            binary = binary_otsu
        else:
            binary = binary_adaptive
        
        # Déterminer orientation (chiffre blanc ou noir)
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
        
        # Morphologie pour nettoyer
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Redimensionner à taille standard
        binary_resized = cv2.resize(binary, self.params['digit_size'])
        gray_resized = cv2.resize(center, self.params['digit_size'])
        
        features = []
        
        # 1. Caractéristiques HOG
        hog_features = hog(
            binary_resized,
            orientations=self.params['hog_orientations'],
            pixels_per_cell=self.params['hog_pixels_per_cell'],
            cells_per_block=self.params['hog_cells_per_block'],
            visualize=False,
            feature_vector=True
        )
        features.extend(hog_features)
        
        # 2. Caractéristiques LBP (Local Binary Pattern)
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray_resized, n_points, radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, 
                                  range=(0, n_points + 2), density=True)
        features.extend(lbp_hist)
        
        # 3. Caractéristiques géométriques
        contours, _ = cv2.findContours(binary_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Plus grand contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Moments géométriques
            moments = cv2.moments(largest_contour)
            if moments['m00'] != 0:
                # Centroïde
                cx = moments['m10'] / moments['m00']
                cy = moments['m01'] / moments['m00']
                
                # Moments centraux normalisés
                hu_moments = cv2.HuMoments(moments).flatten()
                
                # Caractéristiques géométriques
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                # Rectangle englobant
                x, y, w, h = cv2.boundingRect(largest_contour)
                aspect_ratio = w / h if h > 0 else 0
                extent = area / (w * h) if w * h > 0 else 0
                
                # Convex hull
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                features.extend([
                    cx / self.params['digit_size'][0],  # Position X normalisée
                    cy / self.params['digit_size'][1],  # Position Y normalisée
                    area / (self.params['digit_size'][0] * self.params['digit_size'][1]),  # Aire normalisée
                    perimeter / (2 * (self.params['digit_size'][0] + self.params['digit_size'][1])),  # Périmètre normalisé
                    aspect_ratio,
                    extent,
                    solidity
                ])
                features.extend(np.log(np.abs(hu_moments) + 1e-7))  # Moments de Hu (log)
            else:
                # Pas de contour valide
                features.extend([0] * 14)  # 7 caractéristiques + 7 moments de Hu
        else:
            features.extend([0] * 14)
        
        return np.array(features, dtype=np.float32)
    
    def classify_by_advanced_color(self, color_features):
        """
        Classification avancée par couleur avec distances multiples
        
        Args:
            color_features: Vecteur de caractéristiques de couleur
            
        Returns:
            dict: Scores pour chaque ligne
        """
        scores = {}
        
        if self.color_classifier is not None and hasattr(self.color_classifier, 'predict_proba'):
            # Utiliser le classificateur entraîné
            features_scaled = self.color_scaler.transform([color_features])
            probabilities = self.color_classifier.predict_proba(features_scaled)[0]
            
            for i, line_num in enumerate(self.color_classifier.classes_):
                scores[line_num] = probabilities[i]
        else:
            # Fallback: comparaison directe avec les références
            for line_num in METRO_COLORS.keys():
                scores[line_num] = 1.0 / (1.0 + len(METRO_COLORS))  # Score uniforme
        
        return scores
    
    def train_advanced_classifier(self, training_data):
        """
        Entraîne les classificateurs avancés
        
        Args:
            training_data: Liste de tuples (image, label)
        """
        if not training_data:
            print("❌ Pas de données d'entraînement disponibles")
            return
        
        print(f" Entraînement sur {len(training_data)} échantillons...")
        
        # Extraire les caractéristiques
        color_features = []
        digit_features = []
        labels = []
        
        for roi_image, label in training_data:
            try:
                # Caractéristiques de couleur
                color_feat = self.extract_advanced_color_features(roi_image)
                color_features.append(color_feat)
                
                # Caractéristiques de chiffre
                digit_feat = self.extract_advanced_digit_features(roi_image)
                digit_features.append(digit_feat)
                
                labels.append(label)
                
            except Exception as e:
                print(f"Erreur sur échantillon: {e}")
                continue
        
        if not color_features:
            print("Aucune caractéristique extraite")
            return
        
        color_features = np.array(color_features)
        digit_features = np.array(digit_features)
        labels = np.array(labels)
        
        print(f"Caractéristiques couleur: {color_features.shape}")
        print(f"Caractéristiques chiffre: {digit_features.shape}")
        
        color_features_scaled = self.color_scaler.fit_transform(color_features)
        digit_features_scaled = self.digit_scaler.fit_transform(digit_features)
        
        self.color_classifier = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        self.color_classifier.fit(color_features_scaled, labels)
        
        self.digit_classifier = SVC(
            kernel='rbf', 
            C=10.0, 
            gamma='scale', 
            probability=True,
            random_state=42
        )
        self.digit_classifier.fit(digit_features_scaled, labels)
        
        combined_features = np.hstack([color_features_scaled, digit_features_scaled])
        combined_features_scaled = self.ensemble_scaler.fit_transform(combined_features)
        
        rf_ensemble = RandomForestClassifier(n_estimators=150, random_state=42)
        svm_ensemble = SVC(kernel='rbf', C=5.0, probability=True, random_state=42)
        
        self.ensemble_classifier = VotingClassifier(
            estimators=[('rf', rf_ensemble), ('svm', svm_ensemble)],
            voting='soft',
            weights=[0.6, 0.4]  # Pondération RF > SVM
        )
        self.ensemble_classifier.fit(combined_features_scaled, labels)
        
        self.is_trained = True
        
        self._compute_training_stats(color_features_scaled, digit_features_scaled, 
                                   combined_features_scaled, labels)
        
        print(f"Classificateur avancé entraîné avec succès")
    
    def _compute_training_stats(self, color_features, digit_features, combined_features, labels):
        """Calcule les statistiques de performance sur les données d'entraînement"""
        try:

            color_pred = self.color_classifier.predict(color_features)
            digit_pred = self.digit_classifier.predict(digit_features)
            ensemble_pred = self.ensemble_classifier.predict(combined_features)
            
            color_acc = np.mean(color_pred == labels)
            digit_acc = np.mean(digit_pred == labels)
            ensemble_acc = np.mean(ensemble_pred == labels)
            
            class_counts = Counter(labels)
            
            self.training_stats = {
                'color_accuracy': color_acc,
                'digit_accuracy': digit_acc,
                'ensemble_accuracy': ensemble_acc,
                'class_distribution': dict(class_counts),
                'total_samples': len(labels)
            }
            
            print(f"Précision couleur: {color_acc:.3f}")
            print(f"Précision chiffre: {digit_acc:.3f}")
            print(f"Précision ensemble: {ensemble_acc:.3f}")
            
        except Exception as e:
            print(f"⚠️  Erreur calcul stats: {e}")
    
    def classify(self, roi_image):
        """
        Classification finale avec fusion avancée
        
        Args:
            roi_image: Image BGR de la ROI
            
        Returns:
            dict: Résultat de classification avec confiance
        """
        if not self.is_trained:
            return self._classify_color_only(roi_image)
        
        try:
            color_features = self.extract_advanced_color_features(roi_image)
            digit_features = self.extract_advanced_digit_features(roi_image)
            
            color_features_scaled = self.color_scaler.transform([color_features])
            digit_features_scaled = self.digit_scaler.transform([digit_features])
            combined_features = np.hstack([color_features_scaled, digit_features_scaled])
            combined_features_scaled = self.ensemble_scaler.transform(combined_features)

            color_proba = self.color_classifier.predict_proba(color_features_scaled)[0]
            digit_proba = self.digit_classifier.predict_proba(digit_features_scaled)[0]
            ensemble_proba = self.ensemble_classifier.predict_proba(combined_features_scaled)[0]

            color_classes = self.color_classifier.classes_
            digit_classes = self.digit_classifier.classes_
            ensemble_classes = self.ensemble_classifier.classes_
            
            final_scores = {}
            all_classes = set(color_classes) | set(digit_classes) | set(ensemble_classes)
            
            for line_num in all_classes:
                score = 0
                weight_sum = 0
                
                if line_num in color_classes:
                    idx = np.where(color_classes == line_num)[0][0]
                    score += 0.3 * color_proba[idx]
                    weight_sum += 0.3
                
                if line_num in digit_classes:
                    idx = np.where(digit_classes == line_num)[0][0]
                    score += 0.3 * digit_proba[idx]
                    weight_sum += 0.3
                
                if line_num in ensemble_classes:
                    idx = np.where(ensemble_classes == line_num)[0][0]
                    score += 0.4 * ensemble_proba[idx]
                    weight_sum += 0.4
                
                if weight_sum > 0:
                    final_scores[line_num] = score / weight_sum
                else:
                    final_scores[line_num] = 0
            
            best_line = max(final_scores.keys(), key=lambda x: final_scores[x])
            best_confidence = final_scores[best_line]
            
            color_pred = color_classes[np.argmax(color_proba)]
            digit_pred = digit_classes[np.argmax(digit_proba)]
            ensemble_pred = ensemble_classes[np.argmax(ensemble_proba)]
            
            return {
                'line_num': int(best_line),
                'confidence': float(best_confidence),
                'color_prediction': (int(color_pred), float(np.max(color_proba))),
                'digit_prediction': (int(digit_pred), float(np.max(digit_proba))),
                'ensemble_prediction': (int(ensemble_pred), float(np.max(ensemble_proba))),
                'all_scores': {int(k): float(v) for k, v in final_scores.items()}
            }
            
        except Exception as e:
            print(f"Erreur classification: {e}")
            return self._classify_color_only(roi_image)
    
    def _classify_color_only(self, roi_image):
        """Classification de fallback basée uniquement sur la couleur"""
        try:
            lab = cv2.cvtColor(roi_image, cv2.COLOR_BGR2Lab)
            mean_lab = np.mean(lab.reshape(-1, 3), axis=0)
            
            best_line = 1
            best_distance = float('inf')
            
            for line_num, refs in self.color_references.items():
                distance = np.linalg.norm(mean_lab - refs['lab'])
                if distance < best_distance:
                    best_distance = distance
                    best_line = line_num
            
            confidence = max(0.1, 1.0 - (best_distance / 100))
            
            return {
                'line_num': int(best_line),
                'confidence': float(confidence),
                'color_prediction': (int(best_line), float(confidence)),
                'digit_prediction': (None, 0.0),
                'ensemble_prediction': (None, 0.0)
            }
            
        except Exception as e:
            print(f"Erreur classification fallback: {e}")
            return {
                'line_num': 1,
                'confidence': 0.1,
                'color_prediction': (1, 0.1),
                'digit_prediction': (None, 0.0),
                'ensemble_prediction': (None, 0.0)
            }
    
    def save_model(self, filepath):
        """Sauvegarde le modèle complet"""
        if self.is_trained:
            model_data = {
                'color_classifier': self.color_classifier,
                'digit_classifier': self.digit_classifier,
                'ensemble_classifier': self.ensemble_classifier,
                'color_scaler': self.color_scaler,
                'digit_scaler': self.digit_scaler,
                'ensemble_scaler': self.ensemble_scaler,
                'color_references': self.color_references,
                'training_stats': self.training_stats,
                'params': self.params
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Modèle avancé sauvegardé: {filepath}")
        else:
            print(" Aucun modèle entraîné à sauvegarder")
    
    def load_model(self, filepath):
        """Charge un modèle pré-entraîné"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.color_classifier = model_data.get('color_classifier')
                self.digit_classifier = model_data.get('digit_classifier') 
                self.ensemble_classifier = model_data.get('ensemble_classifier')
                self.color_scaler = model_data.get('color_scaler', StandardScaler())
                self.digit_scaler = model_data.get('digit_scaler', StandardScaler())
                self.ensemble_scaler = model_data.get('ensemble_scaler', StandardScaler())
                self.color_references = model_data.get('color_references', self.color_references)
                self.training_stats = model_data.get('training_stats', {})
                self.params.update(model_data.get('params', {}))
                
                self.is_trained = (self.color_classifier is not None and 
                                 self.digit_classifier is not None and
                                 self.ensemble_classifier is not None)
                
                if self.is_trained:
                    print(f"Modèle avancé chargé: {filepath}")
                    if self.training_stats:
                        print(f"Stats d'entraînement: {self.training_stats}")
                else:
                    print(f"Modèle partiellement chargé: {filepath}")
                    
            except Exception as e:
                print(f"Erreur chargement modèle: {e}")
                self.is_trained = False
        else:
            print(f"Fichier modèle non trouvé: {filepath}")


# Alias pour compatibilité avec l'ancien système
LineClassifier = AdvancedLineClassifier 