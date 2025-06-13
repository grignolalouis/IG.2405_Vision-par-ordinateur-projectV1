from src.constants import METRO_COLORS
import os

class MetricsFormatter:
    def __init__(self):
        pass

    def format_single_image_metrics(self, image_name, predictions):
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
        total_detections = sum(len(preds) for preds in predictions.values())
        
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
            
        if resize_factor != 1.0:
            metrics_text += f"\n\n  COORDONNEES RECONVERTIES"
            metrics_text += f"\n  Détection sur images {resize_factor:.2f}x"
            metrics_text += f"\n  Coordonnées dans l'espace original"
        
        return metrics_text
    
    def format_pipeline_metrics(self, performance_metrics, test_images, train_images, 
                               current_image_info=None, ground_truth=None):

        metrics = performance_metrics
        
        detection_metrics = metrics['detection']
        classification_metrics = metrics['classification']
        by_line_metrics = metrics['by_line']
        totals = metrics['totals']
        
        global_text = f""" METRIQUES PIPELINE
        
  DETECTION:
  Precision:       {detection_metrics['precision']:.3f}
  Rappel:          {detection_metrics['recall']:.3f}
  Accuracy:        {detection_metrics['accuracy']:.3f}
  F1-Score:        {detection_metrics['f1']:.3f}
  
  CLASSIFICATION:
  Precision:       {classification_metrics['precision']:.3f}
  Rappel:          {classification_metrics['recall']:.3f}
  Accuracy:        {classification_metrics['accuracy']:.3f}
  F1-Score:        {classification_metrics['f1']:.3f}
  
  STATISTIQUES:
  Vrais Positifs:  {detection_metrics['tp']}
  Faux Positifs:   {detection_metrics['fp']}
  Faux Negatifs:   {detection_metrics['fn']}
  
  PERFORMANCE PAR LIGNE:"""
        
        for line_num in sorted(METRO_COLORS.keys()):
            if line_num in by_line_metrics:
                stats = by_line_metrics[line_num]
                tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
                support = stats['gt_count'] 
                precision = stats['precision']
                recall = stats['recall']
                accuracy = stats['accuracy']
                f1 = stats['f1']
                line_info = METRO_COLORS[line_num]
                global_text += f"\n  Ligne {line_num:2d} ({line_info['name']:12s}): P={precision:.2f} R={recall:.2f} A={accuracy:.2f} F1={f1:.2f} (n={support})"
        
        if 'macro' in metrics and 'weighted' in metrics:
            macro = metrics['macro']
            weighted = metrics['weighted']
            
            global_text += f"\n\n  MOYENNES:"
            global_text += f"\n  Macro        P={macro['precision']:.3f} R={macro['recall']:.3f} A={macro['accuracy']:.3f} F1={macro['f1']:.3f} ({macro['classes']} classes)"
            global_text += f"\n  Weighted     P={weighted['precision']:.3f} R={weighted['recall']:.3f} A={weighted['accuracy']:.3f} F1={weighted['f1']:.3f} ({weighted['support']} signs)"
        
        total_detections_attempted = totals['pred_boxes']
        total_ground_truth_test = totals['gt_boxes']
        correctly_detected_and_classified = classification_metrics['tp']
        
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
        
        if current_image_info:
            global_text += f"\n\n IMAGE COURANTE: {current_image_info['name']}"
            global_text += f"\n  Ground Truth: {current_image_info['gt_count']} panneaux"
            global_text += f"\n  Predictions: {current_image_info['pred_count']} panneaux"
            
            if current_image_info['avg_confidence'] is not None:
                global_text += f"\n  Confiance moyenne: {current_image_info['avg_confidence']:.3f}"
        
        return global_text
    
    def format_default_message(self):
        return """ 
  Système de détection automatique des panneaux de métro parisien
  ═══════════════════════════════════════════════════
  
  CHALLENGE
  Test sur dataset personnalisé :
  • Appuyez sur le bouton "Challenge" pour lancer
  • Chargement automatique du dossier challenge/BD_CHALLENGE
  • Détection automatique des vérités terrain (.mat)
  • Choix facteur redimensionnement (0.1x à 2.0x)
  • Pipeline complet automatique
  • Métriques détaillées et export des résultats + visualisation
  
  DÉMONSTRATION PIPELINE
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
  
  ═══════════════════════════════════════════════════
  
  Appuyez sur le bouton "Challenge" pour commencer l'évaluation."""
    
    def format_no_analysis_message(self):
        return "Aucune analyse.\nChoisir une action." 