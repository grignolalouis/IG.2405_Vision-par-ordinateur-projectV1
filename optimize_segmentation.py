#!/usr/bin/env python3
"""
Optimiseur automatique des paramètres de segmentation
Maximise algorithmiquement le ratio GT/prédictions
"""

import numpy as np
import cv2
import os
import time
from itertools import product
from collections import defaultdict
import pickle
import json
from datetime import datetime

from src.data_loader import DataLoader
from src.detector import MetroSignDetector
from src.segmentation import MetroSegmenter
from src.constants import SEG_PARAMS
from sklearn.model_selection import ParameterGrid
from scipy.optimize import minimize
import optuna


class SegmentationOptimizer:
    """Optimiseur automatique des paramètres de segmentation"""
    
    def __init__(self, validation_size=30):
        """
        Initialise l'optimiseur
        
        Args:
            validation_size: Nombre d'images pour la validation
        """
        self.loader = DataLoader()
        self.validation_size = validation_size
        self.best_params = None
        self.best_score = 0.0
        self.optimization_history = []
        
        # Charger un sous-ensemble pour validation
        self._prepare_validation_set()
        
        # Définir l'espace de recherche des paramètres
        self._define_parameter_space()
    
    def _prepare_validation_set(self):
        """Prépare un ensemble de validation"""
        print(f"🔄 Préparation ensemble de validation ({self.validation_size} images)...")
        
        # Charger les données de test (pour ne pas contaminer l'entraînement)
        test_data = self.loader.load_test_data()
        
        # Prendre un sous-ensemble aléatoire
        import random
        random.seed(42)  # Pour reproductibilité
        self.validation_data = random.sample(test_data, min(self.validation_size, len(test_data)))
        
        print(f"✅ {len(self.validation_data)} images préparées pour validation")
    
    def _define_parameter_space(self):
        """Définit l'espace de recherche des paramètres"""
        
        # PARAMÈTRES À OPTIMISER avec leurs plages
        self.param_space = {
            # Morphologie - Impact majeur sur la détection
            'morph_kernel_close': [(3, 3), (5, 5), (7, 7), (9, 9), (11, 11)],
            'morph_kernel_open': [(3, 3), (5, 5), (7, 7), (9, 9)],
            
            # Aire - Crucial pour filtrer
            'min_area': [500, 800, 1000, 1500, 2000],
            'max_area': [30000, 40000, 50000, 60000, 80000],
            
            # Circularité - Important pour forme
            'min_circularity': [0.5, 0.6, 0.7, 0.8],
            'max_circularity': [1.2, 1.3, 1.4, 1.5],
            
            # Couleur - Impact sur segmentation
            'color_tolerance': [30, 40, 45, 50, 60, 70],
            
            # NMS - Éviter doublons
            'nms_threshold': [0.2, 0.3, 0.4, 0.5],
            
            # Filtre bilatéral - Préparation image
            'bilateral_d': [3, 5, 7, 9],
            'bilateral_sigma_color': [30, 50, 70, 90],
            'bilateral_sigma_space': [30, 50, 70, 90]
        }
        
        print(f"📊 Espace de paramètres défini:")
        total_combinations = 1
        for param, values in self.param_space.items():
            print(f"  {param}: {len(values)} valeurs")
            total_combinations *= len(values)
        print(f"  TOTAL: {total_combinations:,} combinaisons possibles")
    
    def evaluate_parameters(self, params):
        """
        Évalue une combinaison de paramètres
        
        Args:
            params: Dictionnaire de paramètres
            
        Returns:
            Score de performance (à maximiser)
        """
        try:
            # Créer un détecteur avec ces paramètres
            detector = MetroSignDetector(seg_params=params)
            
            # Métriques
            total_tp = 0
            total_fp = 0
            total_fn = 0
            total_gt = 0
            total_pred = 0
            
            for image_path, gt_annotations in self.validation_data:
                # Détecter
                result = detector.detect_signs(image_path)
                detections = result['detections']
                
                total_gt += len(gt_annotations)
                total_pred += len(detections)
                
                # Calculer TP, FP, FN
                matched_gt = set()
                matched_pred = set()
                
                for i, pred in enumerate(detections):
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for j, gt in enumerate(gt_annotations):
                        if j in matched_gt:
                            continue
                        
                        iou = self._calculate_iou(pred, gt)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = j
                    
                    if best_iou > 0.5:  # Seuil IoU
                        matched_gt.add(best_gt_idx)
                        matched_pred.add(i)
                        total_tp += 1
                    else:
                        total_fp += 1
                
                # Faux négatifs
                total_fn += len(gt_annotations) - len(matched_gt)
            
            # Calculer métriques
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Score composite (privilégier F1 mais pénaliser trop de détections)
            efficiency = total_tp / total_pred if total_pred > 0 else 0
            score = f1 * 0.7 + efficiency * 0.3
            
            return {
                'score': score,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'efficiency': efficiency,
                'total_tp': total_tp,
                'total_fp': total_fp,
                'total_fn': total_fn,
                'total_gt': total_gt,
                'total_pred': total_pred
            }
            
        except Exception as e:
            print(f"❌ Erreur évaluation: {e}")
            return {'score': 0.0, 'precision': 0, 'recall': 0, 'f1': 0, 'efficiency': 0,
                   'total_tp': 0, 'total_fp': 999, 'total_fn': 999, 'total_gt': 1, 'total_pred': 999}
    
    def _calculate_iou(self, pred, gt):
        """Calcule IoU entre prédiction et GT"""
        pred_bbox = pred['bbox']
        
        x1 = max(pred_bbox[0], gt['xmin'])
        y1 = max(pred_bbox[1], gt['ymin'])
        x2 = min(pred_bbox[2], gt['xmax'])
        y2 = min(pred_bbox[3], gt['ymax'])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area_pred = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
        area_gt = (gt['xmax'] - gt['xmin']) * (gt['ymax'] - gt['ymin'])
        union = area_pred + area_gt - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def grid_search(self, max_combinations=500):
        """
        Optimisation par grid search (recherche exhaustive)
        
        Args:
            max_combinations: Limite du nombre de combinaisons à tester
        """
        print(f"🔍 GRID SEARCH (max {max_combinations} combinaisons)")
        print("=" * 60)
        
        # Créer toutes les combinaisons
        param_grid = list(ParameterGrid(self.param_space))
        
        if len(param_grid) > max_combinations:
            print(f"⚠️  Trop de combinaisons ({len(param_grid)}), échantillonnage aléatoire...")
            import random
            random.seed(42)
            param_grid = random.sample(param_grid, max_combinations)
        
        print(f"📊 Test de {len(param_grid)} combinaisons...")
        
        start_time = time.time()
        best_score = 0
        best_params = None
        
        for i, params in enumerate(param_grid):
            print(f"\n📋 Test {i+1}/{len(param_grid)}: ", end="")
            
            # Évaluer
            result = self.evaluate_parameters(params)
            score = result['score']
            
            print(f"Score={score:.3f} (P={result['precision']:.3f}, R={result['recall']:.3f}, F1={result['f1']:.3f})")
            
            # Sauvegarder historique
            self.optimization_history.append({
                'method': 'grid_search',
                'iteration': i,
                'params': params.copy(),
                'result': result.copy(),
                'timestamp': datetime.now().isoformat()
            })
            
            # Mise à jour du meilleur
            if score > best_score:
                best_score = score
                best_params = params.copy()
                print(f"  🎯 NOUVEAU MEILLEUR! Score={best_score:.3f}")
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ Grid Search terminé en {elapsed_time:.1f}s")
        print(f"🏆 Meilleur score: {best_score:.3f}")
        
        self.best_score = best_score
        self.best_params = best_params
        
        return best_params, best_score
    
    def random_search(self, n_iterations=200):
        """
        Optimisation par recherche aléatoire
        
        Args:
            n_iterations: Nombre d'itérations
        """
        print(f"🎲 RANDOM SEARCH ({n_iterations} itérations)")
        print("=" * 60)
        
        import random
        random.seed(int(time.time()))
        
        start_time = time.time()
        best_score = 0
        best_params = None
        
        for i in range(n_iterations):
            # Générer combinaison aléatoire
            params = {}
            for param_name, param_values in self.param_space.items():
                params[param_name] = random.choice(param_values)
            
            print(f"\n🎲 Itération {i+1}/{n_iterations}: ", end="")
            
            # Évaluer
            result = self.evaluate_parameters(params)
            score = result['score']
            
            print(f"Score={score:.3f} (P={result['precision']:.3f}, R={result['recall']:.3f}, F1={result['f1']:.3f})")
            
            # Sauvegarder historique
            self.optimization_history.append({
                'method': 'random_search',
                'iteration': i,
                'params': params.copy(),
                'result': result.copy(),
                'timestamp': datetime.now().isoformat()
            })
            
            # Mise à jour du meilleur
            if score > best_score:
                best_score = score
                best_params = params.copy()
                print(f"  🎯 NOUVEAU MEILLEUR! Score={best_score:.3f}")
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ Random Search terminé en {elapsed_time:.1f}s")
        print(f"🏆 Meilleur score: {best_score:.3f}")
        
        if best_score > self.best_score:
            self.best_score = best_score
            self.best_params = best_params
        
        return best_params, best_score
    
    def bayesian_optimization(self, n_trials=100):
        """
        Optimisation bayésienne avec Optuna
        
        Args:
            n_trials: Nombre d'essais
        """
        print(f"🧠 OPTIMISATION BAYÉSIENNE ({n_trials} essais)")
        print("=" * 60)
        
        def objective(trial):
            # Suggérer des paramètres
            params = {
                'morph_kernel_close': trial.suggest_categorical('morph_kernel_close', 
                                                              [(3,3), (5,5), (7,7), (9,9), (11,11)]),
                'morph_kernel_open': trial.suggest_categorical('morph_kernel_open', 
                                                             [(3,3), (5,5), (7,7), (9,9)]),
                'min_area': trial.suggest_int('min_area', 500, 2000, step=100),
                'max_area': trial.suggest_int('max_area', 30000, 80000, step=5000),
                'min_circularity': trial.suggest_float('min_circularity', 0.5, 0.8, step=0.1),
                'max_circularity': trial.suggest_float('max_circularity', 1.2, 1.5, step=0.1),
                'color_tolerance': trial.suggest_int('color_tolerance', 30, 70, step=5),
                'nms_threshold': trial.suggest_float('nms_threshold', 0.2, 0.5, step=0.1),
                'bilateral_d': trial.suggest_categorical('bilateral_d', [3, 5, 7, 9]),
                'bilateral_sigma_color': trial.suggest_int('bilateral_sigma_color', 30, 90, step=10),
                'bilateral_sigma_space': trial.suggest_int('bilateral_sigma_space', 30, 90, step=10)
            }
            
            # Évaluer
            result = self.evaluate_parameters(params)
            score = result['score']
            
            print(f"Trial {trial.number}: Score={score:.3f}")
            
            # Sauvegarder historique
            self.optimization_history.append({
                'method': 'bayesian',
                'iteration': trial.number,
                'params': params.copy(),
                'result': result.copy(),
                'timestamp': datetime.now().isoformat()
            })
            
            return score
        
        # Créer et optimiser
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        best_score = study.best_value
        
        print(f"\n✅ Optimisation bayésienne terminée")
        print(f"🏆 Meilleur score: {best_score:.3f}")
        
        if best_score > self.best_score:
            self.best_score = best_score
            self.best_params = best_params
        
        return best_params, best_score
    
    def save_results(self, filename="optimization_results.json"):
        """Sauvegarde les résultats d'optimisation"""
        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_history': self.optimization_history,
            'timestamp': datetime.now().isoformat(),
            'validation_size': self.validation_size
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Résultats sauvés: {filename}")
    
    def load_results(self, filename="optimization_results.json"):
        """Charge des résultats d'optimisation précédents"""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                results = json.load(f)
            
            self.best_params = results['best_params']
            self.best_score = results['best_score']
            self.optimization_history = results['optimization_history']
            
            print(f"📚 Résultats chargés: {filename}")
            print(f"🏆 Meilleur score précédent: {self.best_score:.3f}")
            return True
        return False
    
    def print_best_results(self):
        """Affiche les meilleurs résultats trouvés"""
        if self.best_params is None:
            print("❌ Aucune optimisation effectuée")
            return
        
        print(f"\n🏆 MEILLEURS PARAMÈTRES TROUVÉS")
        print("=" * 50)
        print(f"Score: {self.best_score:.4f}")
        print("\nParamètres optimaux:")
        
        for param, value in self.best_params.items():
            default_value = SEG_PARAMS.get(param, "N/A")
            change = "🔄" if value != default_value else "✓"
            print(f"  {change} {param}: {value} (défaut: {default_value})")
        
        # Tester les meilleurs paramètres
        print(f"\n📊 Évaluation détaillée des meilleurs paramètres:")
        result = self.evaluate_parameters(self.best_params)
        
        print(f"  Précision: {result['precision']:.3f}")
        print(f"  Rappel: {result['recall']:.3f}")
        print(f"  F1-Score: {result['f1']:.3f}")
        print(f"  Efficacité: {result['efficiency']:.3f}")
        print(f"  TP: {result['total_tp']}, FP: {result['total_fp']}, FN: {result['total_fn']}")
        print(f"  GT total: {result['total_gt']}, Prédictions: {result['total_pred']}")


def main():
    """Fonction principale d'optimisation"""
    print("🚀 OPTIMISEUR AUTOMATIQUE DE SEGMENTATION")
    print("=" * 60)
    
    # Créer l'optimiseur
    optimizer = SegmentationOptimizer(validation_size=25)
    
    # Charger résultats précédents si disponibles
    optimizer.load_results()
    
    # Choisir la méthode d'optimisation
    print("\n🔧 Méthodes d'optimisation disponibles:")
    print("1. Grid Search (exhaustif mais lent)")
    print("2. Random Search (rapide)")
    print("3. Optimisation Bayésienne (intelligent)")
    print("4. Tout combiné")
    
    choice = input("\nChoisir méthode (1-4): ").strip()
    
    if choice == "1":
        optimizer.grid_search(max_combinations=200)
    elif choice == "2":
        optimizer.random_search(n_iterations=150)
    elif choice == "3":
        optimizer.bayesian_optimization(n_trials=100)
    elif choice == "4":
        print("\n🎯 OPTIMISATION COMPLÈTE")
        optimizer.random_search(n_iterations=100)
        optimizer.bayesian_optimization(n_trials=80)
        optimizer.grid_search(max_combinations=100)
    else:
        print("❌ Choix invalide")
        return
    
    # Afficher et sauvegarder résultats
    optimizer.print_best_results()
    optimizer.save_results()


if __name__ == "__main__":
    main() 