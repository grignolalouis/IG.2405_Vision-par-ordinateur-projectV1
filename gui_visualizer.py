"""
POINT D'ENTRÉE PRINCIPAL DU PROJET MÉTRO PARISIEN
Interface graphique avec pipeline automatique complet:
1. Entraînement automatique au démarrage (images ID multiples de 3)
2. Sauvegarde du modèle entraîné
3. Prédiction automatique sur le jeu de test (images ID non multiples de 3)
4. Visualisation comparative des résultats
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import scipy.io as sio
from src.detector import MetroSignDetector
from src.data_loader import DataLoader
from src.constants import METRO_COLORS
import threading
import time
import pickle


class MetroProjectMainGUI:
    """Interface graphique principale du projet métro avec pipeline automatique"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🚇 PROJET MÉTRO PARISIEN - Détection Automatique (TEAM1)")
        self.root.geometry("1500x1000")
        
        # Variables du système
        self.detector = MetroSignDetector()
        self.loader = DataLoader()
        self.ground_truth = {}
        self.predictions = {}
        self.train_images = []
        self.test_images = []
        self.current_index = 0
        
        # État du pipeline
        self.training_completed = False
        self.model_saved = False
        self.prediction_completed = False
        
        # Métriques de performance
        self.performance_metrics = {}
        
        # Créer l'interface
        self.create_interface()
        
        # Démarrer automatiquement le pipeline complet
        self.root.after(1000, self.start_automatic_pipeline)
        
    def create_interface(self):
        """Crée l'interface graphique complète"""
        
        # Configuration principale
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # === TITRE ET STATUT ===
        title_frame = ttk.LabelFrame(main_frame, text="🎯 PROJET MÉTRO PARISIEN - TEAM1", padding="10")
        title_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        title_label = ttk.Label(title_frame, 
                               text="Détection automatique des panneaux métro (lignes 1-14, excluant 3bis et 7bis)",
                               font=("Arial", 12, "bold"))
        title_label.grid(row=0, column=0, pady=5)
        
        self.status_var = tk.StringVar(value="🚀 Initialisation du système...")
        self.status_label = ttk.Label(title_frame, textvariable=self.status_var, 
                                     font=("Arial", 10), foreground="blue")
        self.status_label.grid(row=1, column=0, pady=5)
        
        # Barre de progression globale
        self.progress_bar = ttk.Progressbar(title_frame, mode='determinate', length=600)
        self.progress_bar.grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E))
        
        self.progress_text_var = tk.StringVar(value="")
        self.progress_text_label = ttk.Label(title_frame, textvariable=self.progress_text_var)
        self.progress_text_label.grid(row=3, column=0, pady=2)
        
        # === STATUT DU PIPELINE ===
        pipeline_frame = ttk.LabelFrame(main_frame, text="📋 État du Pipeline", padding="10")
        pipeline_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        self.train_status_var = tk.StringVar(value="⏳ Entraînement: En attente")
        ttk.Label(pipeline_frame, textvariable=self.train_status_var, font=("Arial", 10)).grid(row=0, column=0, sticky=tk.W, pady=2)
        
        self.save_status_var = tk.StringVar(value="⏳ Sauvegarde: En attente")
        ttk.Label(pipeline_frame, textvariable=self.save_status_var, font=("Arial", 10)).grid(row=1, column=0, sticky=tk.W, pady=2)
        
        self.pred_status_var = tk.StringVar(value="⏳ Prédiction: En attente")
        ttk.Label(pipeline_frame, textvariable=self.pred_status_var, font=("Arial", 10)).grid(row=2, column=0, sticky=tk.W, pady=2)
        
        # Informations sur les données
        data_info_frame = ttk.LabelFrame(pipeline_frame, text="📊 Données", padding="5")
        data_info_frame.grid(row=3, column=0, pady=10, sticky=(tk.W, tk.E))
        
        self.train_count_var = tk.StringVar(value="Images d'entraînement: -")
        ttk.Label(data_info_frame, textvariable=self.train_count_var).grid(row=0, column=0, sticky=tk.W)
        
        self.test_count_var = tk.StringVar(value="Images de test: -")
        ttk.Label(data_info_frame, textvariable=self.test_count_var).grid(row=1, column=0, sticky=tk.W)
        
        # === MÉTRIQUES DE PERFORMANCE ===
        metrics_frame = ttk.LabelFrame(main_frame, text="📈 Métriques de Performance", padding="10")
        metrics_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Métriques globales
        self.metrics_text = tk.Text(metrics_frame, height=15, width=40, font=("Consolas", 9))
        self.metrics_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        metrics_scroll = ttk.Scrollbar(metrics_frame, orient=tk.VERTICAL, command=self.metrics_text.yview)
        metrics_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.metrics_text.configure(yscrollcommand=metrics_scroll.set)
        
        # === VISUALISATION ===
        viz_frame = ttk.LabelFrame(main_frame, text="🖼️ Visualisation des Résultats", padding="10")
        viz_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Contrôles de navigation
        nav_frame = ttk.Frame(viz_frame)
        nav_frame.grid(row=0, column=0, pady=5)
        
        ttk.Button(nav_frame, text="◀◀ -10", command=lambda: self.jump_images(-10)).grid(row=0, column=0, padx=2)
        ttk.Button(nav_frame, text="◀ Préc", command=self.prev_image).grid(row=0, column=1, padx=2)
        
        self.image_info_var = tk.StringVar(value="Image 0/0")
        ttk.Label(nav_frame, textvariable=self.image_info_var, font=("Arial", 10, "bold")).grid(row=0, column=2, padx=10)
        
        ttk.Button(nav_frame, text="Suiv ▶", command=self.next_image).grid(row=0, column=3, padx=2)
        ttk.Button(nav_frame, text="+10 ▶▶", command=lambda: self.jump_images(10)).grid(row=0, column=4, padx=2)
        
        # Mode d'affichage
        mode_frame = ttk.LabelFrame(viz_frame, text="Mode d'affichage", padding="5")
        mode_frame.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))
        
        self.display_mode = tk.StringVar(value="comparison")
        ttk.Radiobutton(mode_frame, text="🎯 Vérité terrain", variable=self.display_mode, 
                       value="gt", command=self.update_display).grid(row=0, column=0, padx=5)
        ttk.Radiobutton(mode_frame, text="🤖 Prédictions", variable=self.display_mode, 
                       value="pred", command=self.update_display).grid(row=0, column=1, padx=5)
        ttk.Radiobutton(mode_frame, text="📊 Comparaison", variable=self.display_mode, 
                       value="comparison", command=self.update_display).grid(row=0, column=2, padx=5)
        
        # Canvas pour l'image
        self.canvas = tk.Canvas(viz_frame, width=700, height=500, bg="lightgray")
        self.canvas.grid(row=2, column=0, pady=5)
        
        # === ACTIONS ===
        actions_frame = ttk.LabelFrame(main_frame, text="🎮 Actions", padding="10")
        actions_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(actions_frame, text="🔄 Relancer Pipeline", 
                  command=self.restart_pipeline).grid(row=0, column=0, padx=5)
        ttk.Button(actions_frame, text="💾 Exporter Résultats MAT", 
                  command=self.export_results_mat).grid(row=0, column=1, padx=5)
        ttk.Button(actions_frame, text="📊 Rapport Détaillé", 
                  command=self.show_detailed_report).grid(row=0, column=2, padx=5)
        ttk.Button(actions_frame, text="📁 Ouvrir Dossier Images", 
                  command=self.open_images_folder).grid(row=0, column=3, padx=5)
        
        # Configuration redimensionnement
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
    def start_automatic_pipeline(self):
        """Démarre le pipeline automatique complet"""
        def pipeline_thread():
            try:
                # ÉTAPE 1: Chargement des données
                self.update_status("🔄 Chargement des données...", 10)
                train_data = self.loader.load_training_data()
                test_data = self.loader.load_test_data()
                
                # Sauvegarder les listes d'images
                self.train_images = [path for path, _ in train_data]
                self.test_images = [path for path, _ in test_data]
                
                # Charger la ground truth
                for path, annotations in train_data + test_data:
                    self.ground_truth[os.path.basename(path)] = annotations
                
                # Mettre à jour les compteurs
                self.train_count_var.set(f"Images d'entraînement: {len(train_data)} (IDs multiples de 3)")
                self.test_count_var.set(f"Images de test: {len(test_data)} (IDs non multiples de 3)")
                
                # ÉTAPE 2: Entraînement
                self.update_status("🧠 Entraînement du modèle...", 30)
                self.train_status_var.set("🔄 Entraînement: En cours...")
                self.root.update()
                
                if train_data:
                    # Utiliser la nouvelle méthode d'entraînement avancée
                    self.detector.train_classifier_legacy(train_data)
                    self.training_completed = True
                    self.train_status_var.set(f"✅ Entraînement: Terminé ({len(train_data)} images)")
                else:
                    self.train_status_var.set("❌ Entraînement: Échec (pas de données)")
                
                # ÉTAPE 3: Sauvegarde du modèle
                self.update_status("💾 Sauvegarde du modèle...", 50)
                self.save_status_var.set("🔄 Sauvegarde: En cours...")
                self.root.update()
                
                model_path = "models/metro_detector_trained.pkl"
                os.makedirs("models", exist_ok=True)
                try:
                    with open(model_path, 'wb') as f:
                        pickle.dump(self.detector.classifier, f)
                    self.model_saved = True
                    self.save_status_var.set(f"✅ Sauvegarde: Modèle sauvé dans {model_path}")
                except Exception as e:
                    self.save_status_var.set(f"❌ Sauvegarde: Erreur ({str(e)})")
                
                # ÉTAPE 4: Prédiction sur le test
                self.update_status("🔍 Prédiction sur le jeu de test...", 60)
                self.pred_status_var.set("🔄 Prédiction: En cours...")
                self.root.update()
                
                total_test = len(test_data)
                for i, (image_path, _) in enumerate(test_data):
                    try:
                        result = self.detector.detect_signs(image_path)
                        image_name = os.path.basename(image_path)
                        
                        # Stocker les prédictions (coordonnées déjà à l'échelle originale)
                        self.predictions[image_name] = []
                        for det in result['detections']:
                            xmin, ymin, xmax, ymax = det['bbox']
                            self.predictions[image_name].append({
                                'xmin': int(xmin),
                                'ymin': int(ymin),
                                'xmax': int(xmax),
                                'ymax': int(ymax),
                                'line': det['line_num'],
                                'confidence': det['confidence']
                            })
                        
                        # Mise à jour progression
                        progress = 60 + int((i + 1) / total_test * 30)
                        self.update_status(f"🔍 Prédiction: {i+1}/{total_test} images", progress)
                        
                    except Exception as e:
                        print(f"Erreur prédiction {image_path}: {e}")
                
                self.prediction_completed = True
                self.pred_status_var.set(f"✅ Prédiction: Terminée ({total_test} images)")
                
                # ÉTAPE 5: Calcul des métriques
                self.update_status("📊 Calcul des métriques...", 95)
                self.calculate_performance_metrics()
                
                # ÉTAPE 6: Finalisation
                self.update_status("✅ Pipeline terminé avec succès!", 100)
                
                # Charger la première image de test pour affichage
                if self.test_images:
                    self.current_index = 0
                    self.load_image(self.test_images[0])
                
            except Exception as e:
                self.update_status(f"❌ Erreur pipeline: {str(e)}", 0)
                messagebox.showerror("Erreur Pipeline", f"Erreur dans le pipeline automatique:\n{str(e)}")
        
        # Lancer dans un thread séparé
        thread = threading.Thread(target=pipeline_thread, daemon=True)
        thread.start()
    
    def update_status(self, status, progress):
        """Met à jour la barre de statut"""
        self.status_var.set(status)
        self.progress_bar['value'] = progress
        self.progress_text_var.set(f"{progress}%")
        self.root.update()
    
    def calculate_performance_metrics(self):
        """Calcule les métriques de performance détaillées"""
        tp = fp = fn = tn = 0
        correct_classifications = 0
        total_detections = 0
        line_stats = {line: {'tp': 0, 'fp': 0, 'fn': 0} for line in METRO_COLORS.keys()}
        
        for image_name in self.test_images:
            image_name_base = os.path.basename(image_name)
            gt_boxes = self.ground_truth.get(image_name_base, [])
            pred_boxes = self.predictions.get(image_name_base, [])
            
            # Calcul IoU et matching
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
                
                if best_iou > 0.5:  # Seuil IoU
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(i)
                    tp += 1
                    
                    # Vérifier si la classification est correcte
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
            
            # Faux négatifs
            fn += len(gt_boxes) - len(matched_gt)
            for j, gt in enumerate(gt_boxes):
                if j not in matched_gt:
                    line_stats[gt['line']]['fn'] += 1
        
        # Calculer les métriques
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = correct_classifications / total_detections if total_detections > 0 else 0
        
        self.performance_metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'total_gt': sum(len(boxes) for boxes in self.ground_truth.values()),
            'total_pred': sum(len(boxes) for boxes in self.predictions.values()),
            'line_stats': line_stats
        }
    
    def calculate_iou(self, box1, box2):
        """Calcule l'IoU entre deux boîtes"""
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
    
    def jump_images(self, delta):
        """Saute de plusieurs images"""
        if self.test_images:
            new_index = max(0, min(len(self.test_images) - 1, self.current_index + delta))
            self.current_index = new_index
            self.load_image(self.test_images[self.current_index])
    
    def prev_image(self):
        """Image précédente"""
        self.jump_images(-1)
    
    def next_image(self):
        """Image suivante"""
        self.jump_images(1)
    
    def load_image(self, image_path):
        """Charge et affiche une image"""
        self.current_image_path = image_path
        
        self.current_image = cv2.imread(image_path)
        if self.current_image is None:
            messagebox.showerror("Erreur", f"Impossible de charger l'image: {image_path}")
            return
        
        self.update_display()
        
        if self.current_index >= 0:
            self.image_info_var.set(f"Image {self.current_index + 1}/{len(self.test_images)}")
        else:
            self.image_info_var.set(f"Image: {os.path.basename(image_path)}")
    
    def update_display(self):
        """Met à jour l'affichage de l'image avec les annotations"""
        if not hasattr(self, 'current_image') or self.current_image is None:
            return
            
        image_name = os.path.basename(self.current_image_path)
        mode = self.display_mode.get()
        
        # Copier l'image originale
        display_img = self.current_image.copy()
        
        # Obtenir les annotations
        gt_boxes = self.ground_truth.get(image_name, [])
        pred_boxes = self.predictions.get(image_name, [])
        
        # Dessiner selon le mode
        if mode == "gt" or mode == "comparison":
            # Dessiner vérité terrain en vert
            for gt in gt_boxes:
                cv2.rectangle(display_img, 
                             (gt['xmin'], gt['ymin']), 
                             (gt['xmax'], gt['ymax']), 
                             (0, 255, 0), 2)
                cv2.putText(display_img, f"GT: {gt['line']}", 
                           (gt['xmin'], gt['ymin']-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if mode == "pred" or mode == "comparison":
            # Dessiner prédictions avec matching IoU intelligent
            for i, pred in enumerate(pred_boxes):
                # COULEURS CORRIGÉES pour BGR→RGB conversion
                color = (0, 0, 255)  # ROUGE en BGR (devient rouge en RGB)
                label_suffix = " (FP)"  # Faux Positif
                
                # Vérifier IoU avec GT pour le matching
                if mode == "comparison":
                    best_iou = 0
                    matched_gt = None
                    
                    for gt in gt_boxes:
                        iou = self.calculate_iou(pred, gt)
                        if iou > best_iou:
                            best_iou = iou
                            matched_gt = gt
                    
                    if best_iou > 0.5:  # Seuil IoU
                        if pred['line'] == matched_gt['line']:
                            color = (255, 0, 0)  # BLEU en BGR (devient bleu en RGB)
                            label_suffix = f" (TP, IoU={best_iou:.2f})"
                        else:
                            color = (0, 255, 255)  # JAUNE en BGR (reste jaune en RGB)
                            label_suffix = f" (WC, IoU={best_iou:.2f})"  # Wrong Class
                    else:
                        label_suffix = f" (FP, IoU={best_iou:.2f})"
                else:
                    color = (255, 0, 0)  # BLEU en BGR pour prédictions seules
                    label_suffix = ""
                
                # Dessiner avec épaisseur différente selon le type
                thickness = 3 if color == (255, 0, 0) else 2  # Plus épais pour les TP (bleu)
                
                cv2.rectangle(display_img, 
                             (pred['xmin'], pred['ymin']), 
                             (pred['xmax'], pred['ymax']), 
                             color, thickness)
                
                # Texte avec plus d'informations
                text = f"P{i+1}: L{pred['line']} ({pred['confidence']:.2f}){label_suffix}"
                cv2.putText(display_img, text, 
                           (pred['xmin'], pred['ymax']+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Convertir pour affichage Tkinter
        display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        
        # Redimensionner si nécessaire pour l'affichage
        h, w = display_img_rgb.shape[:2]
        max_size = 650
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            display_img_rgb = cv2.resize(display_img_rgb, (new_w, new_h))
        
        # Afficher dans le canvas
        pil_img = Image.fromarray(display_img_rgb)
        self.photo = ImageTk.PhotoImage(pil_img)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        # Mettre à jour les métriques
        self.update_metrics_display()
    
    def update_metrics_display(self):
        """Met à jour l'affichage des métriques"""
        if self.performance_metrics:
            metrics = self.performance_metrics
            
            # Texte des métriques globales
            global_text = f"""📊 MÉTRIQUES GLOBALES
            
🎯 DÉTECTION:
  Précision:       {metrics['precision']:.3f}
  Rappel:          {metrics['recall']:.3f}
  F1-Score:        {metrics['f1']:.3f}
  
🔢 CLASSIFICATION:
  Précision:       {metrics['accuracy']:.3f}
  
📈 STATISTIQUES:
  Vrais Positifs:  {metrics['tp']}
  Faux Positifs:   {metrics['fp']}
  Faux Négatifs:   {metrics['fn']}
  
📊 DONNÉES:
  GT total:        {metrics['total_gt']}
  Prédictions:     {metrics['total_pred']}
  Images test:     {len(self.test_images)}
  Images train:    {len(self.train_images)}
  
🚇 PERFORMANCE PAR LIGNE:"""
            
            # Ajouter stats par ligne
            for line_num in sorted(METRO_COLORS.keys()):
                if line_num in metrics['line_stats']:
                    stats = metrics['line_stats'][line_num]
                    tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    global_text += f"\n  Ligne {line_num:2d}: P={precision:.2f} R={recall:.2f} (TP={tp}, FP={fp}, FN={fn})"
            
            # Stats image courante
            if hasattr(self, 'current_image_path') and self.current_image_path:
                image_name = os.path.basename(self.current_image_path)
                gt_boxes = self.ground_truth.get(image_name, [])
                pred_boxes = self.predictions.get(image_name, [])
                
                global_text += f"\n\n🖼️ IMAGE COURANTE: {image_name}"
                global_text += f"\n  GT: {len(gt_boxes)} panneaux"
                global_text += f"\n  Prédictions: {len(pred_boxes)} panneaux"
                
                if pred_boxes:
                    avg_conf = sum(p['confidence'] for p in pred_boxes) / len(pred_boxes)
                    global_text += f"\n  Confiance moy: {avg_conf:.3f}"
        else:
            global_text = "Calcul des métriques en cours..."
        
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(1.0, global_text)
    
    def restart_pipeline(self):
        """Relance le pipeline complet"""
        if messagebox.askyesno("Confirmer", "Relancer le pipeline complet (entraînement + test)?"):
            # Réinitialiser l'état
            self.training_completed = False
            self.model_saved = False
            self.prediction_completed = False
            self.predictions.clear()
            self.performance_metrics.clear()
            
            # Relancer
            self.start_automatic_pipeline()
    
    def export_results_mat(self):
        """Exporte les résultats au format MAT comme demandé par le projet"""
        if not self.predictions:
            messagebox.showwarning("Attention", "Aucune prédiction à exporter")
            return
        
        try:
            # Créer le tableau de résultats au format requis
            results_list = []
            
            for image_path in self.test_images:
                image_name = os.path.basename(image_path)
                
                # Extraire l'ID de l'image
                image_id = self.loader._extract_image_id(image_name)
                if image_id is None:
                    continue
                
                # Ajouter les prédictions pour cette image
                pred_boxes = self.predictions.get(image_name, [])
                
                if pred_boxes:
                    for pred in pred_boxes:
                        # Format correct: [image_id, y1, y2, x1, x2, line_pred]
                        results_list.append([
                            image_id,
                            pred['ymin'],  # y1
                            pred['ymax'],  # y2
                            pred['xmin'],  # x1
                            pred['xmax'],  # x2
                            pred['line']
                        ])
                else:
                    # Image sans détection - ajouter une ligne vide ou ignorer
                    # Selon les exigences du projet
                    pass
            
            # Convertir en array numpy
            if results_list:
                results_array = np.array(results_list)
                
                # Sauvegarder au format MAT
                output_file = "results_test_TEAM1.mat"
                sio.savemat(output_file, {'BD': results_array})
                
                messagebox.showinfo("Succès", 
                                  f"Résultats exportés dans {output_file}\n"
                                  f"Format: [image_id, y1, y2, x1, x2, line_pred]\n"
                                  f"{len(results_list)} détections sur {len(self.test_images)} images")
            else:
                messagebox.showwarning("Attention", "Aucune détection à exporter")
                
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'export MAT: {str(e)}")
    
    def show_detailed_report(self):
        """Affiche un rapport détaillé dans une nouvelle fenêtre"""
        if not self.performance_metrics:
            messagebox.showwarning("Attention", "Calcul des métriques en cours...")
            return
        
        # Créer une nouvelle fenêtre
        report_window = tk.Toplevel(self.root)
        report_window.title("📊 Rapport Détaillé - Projet Métro TEAM1")
        report_window.geometry("800x600")
        
        # Zone de texte avec scrollbar
        text_frame = ttk.Frame(report_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_area = tk.Text(text_frame, wrap=tk.WORD, font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_area.yview)
        text_area.configure(yscrollcommand=scrollbar.set)
        
        text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Générer le rapport
        metrics = self.performance_metrics
        
        report_text = f"""🚇 RAPPORT DÉTAILLÉ - PROJET MÉTRO PARISIEN (TEAM1)
{'='*70}

📋 CONFIGURATION DU PROJET:
  • Lignes détectées: 1-14 (excluant 3bis et 7bis)
  • Split train/test: Multiples de 3 pour train, reste pour test
  • Images à taille originale (pas de redimensionnement)
  • Pipeline automatique complet

📊 RÉSULTATS GLOBAUX:
  • Images d'entraînement: {len(self.train_images)}
  • Images de test: {len(self.test_images)}
  • Total annotations GT: {metrics['total_gt']}
  • Total prédictions: {metrics['total_pred']}

🎯 MÉTRIQUES DE DÉTECTION:
  • Précision: {metrics['precision']:.4f} ({metrics['tp']}/{metrics['tp'] + metrics['fp']})
  • Rappel: {metrics['recall']:.4f} ({metrics['tp']}/{metrics['tp'] + metrics['fn']})
  • F1-Score: {metrics['f1']:.4f}

🔢 MÉTRIQUES DE CLASSIFICATION:
  • Précision classification: {metrics['accuracy']:.4f}

📈 DÉTAIL DES ERREURS:
  • Vrais Positifs (TP): {metrics['tp']}
  • Faux Positifs (FP): {metrics['fp']}
  • Faux Négatifs (FN): {metrics['fn']}

🚇 PERFORMANCE PAR LIGNE DE MÉTRO:
{'-'*50}
"""
        
        for line_num in sorted(METRO_COLORS.keys()):
            if line_num in metrics['line_stats']:
                stats = metrics['line_stats'][line_num]
                tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                color_info = METRO_COLORS[line_num]
                report_text += f"Ligne {line_num:2d} ({color_info['name']:15s}): "
                report_text += f"P={precision:.3f} R={recall:.3f} F1={f1:.3f} "
                report_text += f"(TP={tp:2d}, FP={fp:2d}, FN={fn:2d})\n"
        
        report_text += f"""
💾 FICHIERS GÉNÉRÉS:
  • Modèle entraîné: models/metro_detector_trained.pkl
  • Résultats test: results_test_TEAM1.mat
  
🏁 STATUT DU PIPELINE:
  • Entraînement: {'✅ Terminé' if self.training_completed else '❌ Échoué'}
  • Sauvegarde modèle: {'✅ Terminé' if self.model_saved else '❌ Échoué'}
  • Prédiction test: {'✅ Terminé' if self.prediction_completed else '❌ Échoué'}

{'='*70}
Rapport généré automatiquement par le système de détection métro TEAM1
"""
        
        text_area.insert(1.0, report_text)
        text_area.configure(state='disabled')
        
        # Bouton pour sauvegarder le rapport
        def save_report():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Fichiers texte", "*.txt"), ("Tous les fichiers", "*.*")]
            )
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                messagebox.showinfo("Succès", f"Rapport sauvegardé dans {file_path}")
        
        ttk.Button(report_window, text="💾 Sauvegarder rapport", 
                  command=save_report).pack(pady=10)
    
    def open_images_folder(self):
        """Ouvre le dossier des images dans l'explorateur"""
        import subprocess
        import platform
        
        folder_path = self.loader.image_dir
        
        if platform.system() == "Windows":
            subprocess.Popen(f'explorer "{folder_path}"')
        elif platform.system() == "Darwin":  # macOS
            subprocess.Popen(["open", folder_path])
        else:  # Linux
            subprocess.Popen(["xdg-open", folder_path])


def main():
    """Fonction principale"""
    root = tk.Tk()
    app = MetroProjectMainGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main() 