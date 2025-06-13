"""
Module de formatage des métriques pour l'affichage.
Sépare la logique de formatage de l'interface utilisateur.
"""

from src.constants import METRO_COLORS
import os


class MetricsFormatter:
    """Classe responsable du formatage des métriques pour l'affichage."""
    
    def __init__(self):
        pass
    
    def format_single_image_metrics(self, image_name, predictions):
        """
        Formater les métriques pour une image unique.
        
        Args:
            image_name: Nom de l'image
            predictions: Liste des prédictions
            
        Returns:
            Texte formaté pour affichage
        """
        metrics_text = f""" ANALYSE IMAGE UNIQUE
        
  IMAGE: {image_name}
  DETECTIONS: {len(predictions)}
  
  DETAIL:"""
        
        for i, pred in enumerate(predictions):
            line_info = METRO_COLORS.get(pred['line'], {'name': 'Inconnue'})
            metrics_text += f"\n  {i+1}. Ligne {pred['line']} ({line_info['name']})"
            metrics_text += f"\n      Confiance: {pred['confidence']:.3f}"
            metrics_text += f"\n      Position: ({pred['xmin']}, {pred['ymin']}) -> ({pred['xmax']}, {pred['ymax']})"
            metrics_text += f"\n      Taille: {pred['xmax']-pred['xmin']}x{pred['ymax']-pred['ymin']} px\n"
        
        if not predictions:
            metrics_text += "\n  Aucun panneau detecte."
        
        return metrics_text
    
    def format_challenge_metrics(self, test_images, predictions, resize_factor=1.0):
        """
        Formater les métriques pour un dataset challenge.
        
        Args:
            test_images: Liste des images de test
            predictions: Dictionnaire des prédictions
            resize_factor: Facteur de redimensionnement appliqué
            
        Returns:
            Texte formaté pour affichage
        """
        total_detections = sum(len(preds) for preds in predictions.values())
        
        # Informations sur le redimensionnement
        resize_info = ""
        if resize_factor != 1.0:
            resize_info = f"\n  REDIMENSIONNEMENT: {resize_factor:.2f}x"
            if resize_factor < 1.0:
                resize_info += " (Réduction)"
            else:
                resize_info += " (Agrandissement)"
        
        metrics_text = f""" DATASET CHALLENGE
        
  IMAGES: {len(test_images)}
  DETECTIONS: {total_detections}
  MOYENNE: {total_detections/len(test_images):.1f}{resize_info}
  
  REPARTITION:"""
        
        # Calculer la répartition par ligne
        line_counts = {}
        total_confidence = 0
        for preds in predictions.values():
            for pred in preds:
                line = pred['line']
                line_counts[line] = line_counts.get(line, 0) + 1
                total_confidence += pred['confidence']
        
        for line_num in sorted(line_counts.keys()):
            line_info = METRO_COLORS.get(line_num, {'name': 'Inconnue'})
            metrics_text += f"\n  Ligne {line_num:2d} ({line_info['name']:12s}): {line_counts[line_num]:3d}"
        
        if total_detections > 0:
            avg_confidence = total_confidence / total_detections
            metrics_text += f"\n\n  CONFIANCE MOYENNE: {avg_confidence:.3f}"
            
        # Avertissement sur le redimensionnement
        if resize_factor != 1.0:
            metrics_text += f"\n\n  COORDONNEES RECONVERTIES"
            metrics_text += f"\n  Détection sur images {resize_factor:.2f}x"
            metrics_text += f"\n  Coordonnées dans l'espace original"
        
        return metrics_text
    
    def format_pipeline_metrics(self, performance_metrics, test_images, train_images, 
                               current_image_info=None, ground_truth=None):
        """
        Formater les métriques complètes du pipeline.
        
        Args:
            performance_metrics: Dictionnaire des métriques de performance
            test_images: Liste des images de test
            train_images: Liste des images d'entraînement
            current_image_info: Informations sur l'image courante (optionnel)
            ground_truth: Dictionnaire des vérités terrain (optionnel)
            
        Returns:
            Texte formaté pour affichage
        """
        metrics = performance_metrics
        
        # Calculer les métriques de classification
        total_correct_classifications = 0
        total_classifications = 0
        classification_tp = classification_fp = classification_fn = 0
        
        for line_stats in metrics['line_stats'].values():
            classification_tp += line_stats['tp']
            classification_fp += line_stats['fp'] 
            classification_fn += line_stats['fn']
        
        total_classifications = classification_tp + classification_fp
        classification_precision = classification_tp / (classification_tp + classification_fp) if (classification_tp + classification_fp) > 0 else 0
        classification_recall = classification_tp / (classification_tp + classification_fn) if (classification_tp + classification_fn) > 0 else 0
        classification_f1 = 2 * classification_precision * classification_recall / (classification_precision + classification_recall) if (classification_precision + classification_recall) > 0 else 0
        
        global_text = f""" METRIQUES PIPELINE
        
  DETECTION:
  Precision:       {metrics['precision']:.3f}
  Rappel:          {metrics['recall']:.3f}
  F1-Score:        {metrics['f1']:.3f}
  
  CLASSIFICATION:
  Precision:       {classification_precision:.3f}
  Rappel:          {classification_recall:.3f}
  F1-Score:        {classification_f1:.3f}
  
  STATISTIQUES:
  Vrais Positifs:  {metrics['tp']}
  Faux Positifs:   {metrics['fp']}
  Faux Negatifs:   {metrics['fn']}
  
  PERFORMANCE PAR LIGNE:"""
        
        for line_num in sorted(METRO_COLORS.keys()):
            if line_num in metrics['line_stats']:
                stats = metrics['line_stats'][line_num]
                tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                line_info = METRO_COLORS[line_num]
                global_text += f"\n  Ligne {line_num:2d} ({line_info['name']:12s}): P={precision:.2f} R={recall:.2f} F1={f1:.2f}"
        
        # Section OVERALL - Métriques générales du dataset
        total_detections_attempted = metrics['tp'] + metrics['fp']
        
        # Calculer le total de GT seulement pour les images de test
        if ground_truth is not None:
            total_ground_truth_test = 0
            for image_path in test_images:
                image_name = os.path.basename(image_path)
                if image_name in ground_truth:
                    total_ground_truth_test += len(ground_truth[image_name])
        else:
            # Fallback: utiliser les métriques calculées (qui sont déjà sur test seulement)
            total_ground_truth_test = metrics['tp'] + metrics['fn']
        
        correctly_detected_and_classified = classification_tp
        
        overall_precision = correctly_detected_and_classified / total_detections_attempted if total_detections_attempted > 0 else 0
        overall_recall = correctly_detected_and_classified / total_ground_truth_test if total_ground_truth_test > 0 else 0
        overall_accuracy = correctly_detected_and_classified / total_ground_truth_test if total_ground_truth_test > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        global_text += f"""
        
  OVERALL - PERFORMANCE GENERALE (Test seulement):
  Precision:       {overall_precision:.3f}
  Rappel:          {overall_recall:.3f}
  Accuracy:        {overall_accuracy:.3f}
  F1-Score:        {overall_f1:.3f}
  
  Correctement détectés et classifiés: {correctly_detected_and_classified}/{total_ground_truth_test}
  Taux de réussite global: {overall_accuracy*100:.1f}%"""
        
        # Ajouter les informations sur l'image courante si disponibles
        if current_image_info:
            global_text += f"\n\n IMAGE COURANTE: {current_image_info['name']}"
            global_text += f"\n  Ground Truth: {current_image_info['gt_count']} panneaux"
            global_text += f"\n  Predictions: {current_image_info['pred_count']} panneaux"
            
            if current_image_info['avg_confidence'] is not None:
                global_text += f"\n  Confiance moyenne: {current_image_info['avg_confidence']:.3f}"
        
        return global_text
    
    def format_default_message(self):
        """Formater le message par défaut quand aucune analyse n'est disponible."""
        return """ SYSTEME DETECTION PANNEAUX METRO
        
  Bienvenue dans le système de détection automatique
  des panneaux de métro parisien utilisant YOLO + 
  Classification par apprentissage automatique.
  
  ═══════════════════════════════════════════════════
  
  PIPELINE COMPLET
  Démonstration complète du système :
  • Entraînement du classificateur sur données train
  • Test automatique sur dataset de validation
  • Segmentation YOLO + Classification des ROIs
  • Calcul des métriques de performance
  • Affichage des résultats avec navigation
  
  IMAGE UNIQUE  
  Analyse d'une image personnalisée :
  • Chargement d'une image quelconque
  • Détection automatique des panneaux métro
  • Segmentation + Classification en temps réel
  • Affichage des résultats avec confiance
  
  CHALLENGE + REDIM
  Test sur dataset personnalisé :
  • Chargement d'un nouveau dossier d'images
  • Option : Chargement vérités terrain (.mat)
  • Choix facteur redimensionnement (0.1x à 2.0x)
  • Pipeline complet avec reconversion coordonnées
  • Métriques détaillées si GT disponibles
  
  ═══════════════════════════════════════════════════
  
  Sélectionnez une action pour commencer."""
    
    def format_no_analysis_message(self):
        """Formater le message quand aucune analyse n'est en cours."""
        return "Aucune analyse.\nChoisir une action." 