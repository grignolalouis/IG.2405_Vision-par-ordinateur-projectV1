"""
Interface utilisateur principale refactorisée.
Sépare clairement l'interface de la logique métier.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import os
import threading
from src.constants import GUI_PARAMS
from .image_renderer import ImageRenderer
from .metrics_formatter import MetricsFormatter
from .workflow_manager import WorkflowManager
from .data_processor import DataProcessor


class MetroProjectMainGUI:
    """Interface utilisateur principale pour le projet de détection de panneaux métro."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Detection Panneaux Metro - TEAM1")
        self.root.geometry(GUI_PARAMS['window_size'])
        
        # Composants métier
        self.workflow_manager = WorkflowManager()
        self.image_renderer = ImageRenderer()
        self.metrics_formatter = MetricsFormatter()
        self.data_processor = DataProcessor()
        
        # État de l'application
        self.ground_truth = {}
        self.predictions = {}
        self.train_images = []
        self.test_images = []
        self.current_index = 0
        self.current_image = None
        self.current_image_path = None
        self.performance_metrics = {}
        
        # Modes d'affichage
        self.single_image_mode = False
        self.challenge_mode = False
        self.resize_factor = 1.0
        
        self.create_interface()
        self.update_metrics_display()
        
    def create_interface(self):
        """Créer l'interface utilisateur complète."""
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.create_header(main_frame)
        self.create_action_panel(main_frame)
        self.create_main_content(main_frame)
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
    def create_header(self, parent):
        """Créer l'en-tête avec titre et barre de progression."""
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Titre principal
        title_label = ttk.Label(header_frame, text="Détection Panneaux Métro Parisien - TEAM1", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, sticky=tk.W)
        
        # Barre de progression avec texte de statut
        progress_frame = ttk.Frame(header_frame)
        progress_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(8, 0))
        
        self.status_var = tk.StringVar(value="Système prêt")
        status_label = ttk.Label(progress_frame, textvariable=self.status_var, 
                                font=("Arial", 10))
        status_label.grid(row=0, column=0, sticky=tk.W)
        
        self.progress = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.progress.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(4, 0))
        
        # Configuration pour expansion
        header_frame.columnconfigure(0, weight=1)
        progress_frame.columnconfigure(0, weight=1)
    
    def create_action_panel(self, parent):
        """Créer le panneau d'actions avec les boutons principaux."""
        action_frame = ttk.LabelFrame(parent, text="ACTIONS", padding="15")
        action_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Frame pour les boutons essentiels
        buttons_frame = ttk.Frame(action_frame)
        buttons_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Boutons principaux sur une seule ligne
        ttk.Button(buttons_frame, text="🚀 Pipeline Complet", 
                  command=self.start_complete_pipeline).grid(row=0, column=0, padx=(0, 10), sticky=(tk.W, tk.E))
        
        ttk.Button(buttons_frame, text="🖼️ Image Unique", 
                  command=self.load_single_image).grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        
        ttk.Button(buttons_frame, text="🔧 Challenge + Redim", 
                  command=self.load_challenge_with_resize).grid(row=0, column=2, padx=(10, 0), sticky=(tk.W, tk.E))
        
        # Configuration pour expansion uniforme
        action_frame.columnconfigure(0, weight=1)
        buttons_frame.columnconfigure(0, weight=1)
        buttons_frame.columnconfigure(1, weight=1)
        buttons_frame.columnconfigure(2, weight=1)
    
    def create_main_content(self, parent):
        """Créer le contenu principal avec panneaux gauche et droit."""
        content_frame = ttk.Frame(parent)
        content_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Panel gauche avec taille flexible
        left_panel = ttk.Frame(content_frame)
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        self.create_pipeline_panel(left_panel)
        self.create_metrics_panel(left_panel)
        
        # Panel droit pour visualisation avec plus d'espace
        right_panel = ttk.LabelFrame(content_frame, text="VISUALISATION", padding="10")
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.create_visualization_panel(right_panel)
        
        # Configuration des poids pour une répartition optimale
        content_frame.columnconfigure(0, weight=1, minsize=400)  # Panel gauche
        content_frame.columnconfigure(1, weight=2, minsize=600)  # Panel droit plus large
        content_frame.rowconfigure(0, weight=1)
        
        # Configuration du panel gauche
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(0, weight=0)  # Pipeline panel fixe
        left_panel.rowconfigure(1, weight=1)  # Metrics panel extensible
    
    def create_pipeline_panel(self, parent):
        """Créer le panneau d'état du pipeline."""
        pipeline_frame = ttk.LabelFrame(parent, text="État du Pipeline", padding="12")
        pipeline_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Status des étapes en format compact
        status_frame = ttk.Frame(pipeline_frame)
        status_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        
        self.train_status_var = tk.StringVar(value="Entraînement: En attente")
        ttk.Label(status_frame, textvariable=self.train_status_var, 
                 font=("Arial", 10)).grid(row=0, column=0, sticky=tk.W, pady=1)
        
        self.save_status_var = tk.StringVar(value="Sauvegarde: En attente")
        ttk.Label(status_frame, textvariable=self.save_status_var, 
                 font=("Arial", 10)).grid(row=1, column=0, sticky=tk.W, pady=1)
        
        self.pred_status_var = tk.StringVar(value="Prédiction: En attente")
        ttk.Label(status_frame, textvariable=self.pred_status_var, 
                 font=("Arial", 10)).grid(row=2, column=0, sticky=tk.W, pady=1)
        
        # Informations sur les données
        data_info_frame = ttk.LabelFrame(pipeline_frame, text="Données", padding="8")
        data_info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.train_count_var = tk.StringVar(value="Images entraînement: -")
        ttk.Label(data_info_frame, textvariable=self.train_count_var, 
                 font=("Arial", 10)).grid(row=0, column=0, sticky=tk.W, pady=1)
        
        self.test_count_var = tk.StringVar(value="Images test: -")
        ttk.Label(data_info_frame, textvariable=self.test_count_var,
                 font=("Arial", 10)).grid(row=1, column=0, sticky=tk.W, pady=1)
        
        # Configuration pour expansion
        pipeline_frame.columnconfigure(0, weight=1)
        status_frame.columnconfigure(0, weight=1)
        data_info_frame.columnconfigure(0, weight=1)
    
    def create_metrics_panel(self, parent):
        """Créer le panneau des métriques."""
        metrics_frame = ttk.LabelFrame(parent, text="RÉSULTATS & MÉTRIQUES", padding="12")
        metrics_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Zone de texte avec scrollbar pour les métriques
        text_frame = ttk.Frame(metrics_frame)
        text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.metrics_text = tk.Text(text_frame, 
                                   font=("Consolas", 10), 
                                   wrap=tk.WORD,
                                   bg='#f8f9fa',
                                   relief=tk.SUNKEN,
                                   borderwidth=1,
                                   padx=8,
                                   pady=8)
        self.metrics_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar verticale
        metrics_scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.metrics_text.yview)
        metrics_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.metrics_text.configure(yscrollcommand=metrics_scroll.set)
        
        # Configuration pour expansion complète
        metrics_frame.columnconfigure(0, weight=1)
        metrics_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
    
    def create_visualization_panel(self, parent):
        """Créer le panneau de visualisation."""
        # Contrôles de mode d'affichage
        mode_frame = ttk.LabelFrame(parent, text="Mode d'affichage", padding="10")
        mode_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.display_mode = tk.StringVar(value="comparison")
        
        # Boutons radio en ligne
        controls_frame = ttk.Frame(mode_frame)
        controls_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        ttk.Radiobutton(controls_frame, text="Vérité Terrain", variable=self.display_mode, 
                       value="gt", command=self.update_display).grid(row=0, column=0, padx=(0, 15), sticky=tk.W)
        ttk.Radiobutton(controls_frame, text="Prédictions", variable=self.display_mode, 
                       value="pred", command=self.update_display).grid(row=0, column=1, padx=15, sticky=tk.W)
        ttk.Radiobutton(controls_frame, text="Comparaison", variable=self.display_mode, 
                       value="comparison", command=self.update_display).grid(row=0, column=2, padx=(15, 0), sticky=tk.W)
        
        # Légende des couleurs
        legend_frame = ttk.LabelFrame(parent, text="Légende des couleurs", padding="10")
        legend_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.create_color_legend(legend_frame)
        
        # Contrôles de navigation
        nav_frame = ttk.Frame(parent)
        nav_frame.grid(row=2, column=0, pady=(0, 10), sticky=(tk.W, tk.E))
        
        # Boutons de navigation centrés
        nav_buttons_frame = ttk.Frame(nav_frame)
        nav_buttons_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        ttk.Button(nav_buttons_frame, text=f"<<-{GUI_PARAMS['jump_size']}", 
                  command=lambda: self.jump_images(-GUI_PARAMS['jump_size'])).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(nav_buttons_frame, text="< Précédent", 
                  command=self.prev_image).grid(row=0, column=1, padx=5)
        
        # Info image au centre
        self.image_info_var = tk.StringVar(value="Aucune image")
        info_label = ttk.Label(nav_buttons_frame, textvariable=self.image_info_var, 
                              font=("Arial", 11, "bold"))
        info_label.grid(row=0, column=2, padx=20)
        
        ttk.Button(nav_buttons_frame, text="Suivant >", 
                  command=self.next_image).grid(row=0, column=3, padx=5)
        ttk.Button(nav_buttons_frame, text=f"+{GUI_PARAMS['jump_size']}>>", 
                  command=lambda: self.jump_images(GUI_PARAMS['jump_size'])).grid(row=0, column=4, padx=(5, 0))
        
        # Canvas avec scrollbars
        canvas_frame = ttk.Frame(parent, relief=tk.SUNKEN, borderwidth=1)
        canvas_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.canvas = tk.Canvas(canvas_frame, bg="#f8f9fa", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbars
        canvas_scroll_y = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        canvas_scroll_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        canvas_scroll_x = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        canvas_scroll_x.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.canvas.configure(yscrollcommand=canvas_scroll_y.set, xscrollcommand=canvas_scroll_x.set)
        
        # Message par défaut
        self.canvas.create_text(400, 300, text="Aucune image chargée\nUtilisez les boutons d'action ci-dessus", 
                               font=("Arial", 12), fill="#6B7280", justify=tk.CENTER)
        
        # Configuration pour expansion
        parent.rowconfigure(3, weight=1)  # Changé de 2 à 3
        parent.columnconfigure(0, weight=1)
        
        mode_frame.columnconfigure(0, weight=1)
        controls_frame.columnconfigure(1, weight=1)  # Centre les contrôles
        
        legend_frame.columnconfigure(0, weight=1)
        
        nav_frame.columnconfigure(0, weight=1)
        nav_buttons_frame.columnconfigure(2, weight=1)  # Centre l'info image
        
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
    
    def create_color_legend(self, parent):
        """Créer la légende des couleurs pour les prédictions."""
        # Frame principal pour organiser la légende
        legend_container = ttk.Frame(parent)
        legend_container.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Définir les couleurs et leurs descriptions basées sur les constantes
        color_info = [
            (GUI_PARAMS['colors']['gt'], "Vérité Terrain", "■"),
            (GUI_PARAMS['colors']['pred_tp'], "Vrais Positifs", "■"),
            (GUI_PARAMS['colors']['pred_fp'], "Faux Positifs", "■"),
            (GUI_PARAMS['colors']['pred_wc'], "Mauvaise Classe", "■")
        ]
        
        # Créer les éléments de légende en ligne
        for i, (color_bgr, description, symbol) in enumerate(color_info):
            # Convertir BGR vers RGB pour Tkinter
            color_rgb = f"#{color_bgr[2]:02x}{color_bgr[1]:02x}{color_bgr[0]:02x}"
            
            # Frame pour chaque élément de légende
            item_frame = ttk.Frame(legend_container)
            item_frame.grid(row=0, column=i, padx=(0, 20), sticky=tk.W)
            
            # Carré coloré - utiliser un fond par défaut
            color_label = tk.Label(item_frame, text=symbol, 
                                 font=("Arial", 14, "bold"),
                                 fg=color_rgb,
                                 bg='SystemButtonFace')  # Couleur de fond par défaut du système
            color_label.grid(row=0, column=0, padx=(0, 5))
            
            # Description
            desc_label = ttk.Label(item_frame, text=description, 
                                 font=("Arial", 10))
            desc_label.grid(row=0, column=1)
        
        # Configuration pour centrer la légende
        legend_container.columnconfigure(0, weight=1)
        legend_container.columnconfigure(1, weight=1)
        legend_container.columnconfigure(2, weight=1)
        legend_container.columnconfigure(3, weight=1)
        parent.columnconfigure(0, weight=1)
    
    # === MÉTHODES D'ACTION ===
    
    def start_complete_pipeline(self):
        """Démarrer le pipeline complet via WorkflowManager."""
        # Préparer les callbacks pour les mises à jour
        progress_callbacks = {
            'status_callback': self.update_status,
            'train_status_callback': lambda msg: self.train_status_var.set(msg),
            'save_status_callback': lambda msg: self.save_status_var.set(msg),
            'pred_status_callback': lambda msg: self.pred_status_var.set(msg),
            'count_callbacks': {
                'train_count': lambda msg: self.train_count_var.set(msg),
                'test_count': lambda msg: self.test_count_var.set(msg)
            }
        }
        
        # Callback de completion pour appliquer les résultats
        def on_pipeline_complete(results):
            # Programmer l'exécution dans le thread principal
            self.root.after(0, lambda: self._apply_pipeline_results(results))
        
        # Lancer le pipeline avec le callback de completion
        self.workflow_manager.run_complete_pipeline(progress_callbacks, on_pipeline_complete)
    
    def load_single_image(self):
        """Charger et analyser une image unique."""
        file_path = filedialog.askopenfilename(
            title="Sélectionner image", 
            filetypes=GUI_PARAMS['supported_image_formats']
        )
        
        if file_path:
            try:
                self.update_status("Analyse image...", 30)
                
                # Utiliser WorkflowManager pour traiter l'image
                results = self.workflow_manager.process_single_image(file_path)
                
                # Mettre à jour l'état de l'application
                self._handle_single_image_results(results, file_path)
                
                self.update_status("Analyse terminée", 100)
                
            except Exception as e:
                messagebox.showerror("Erreur", str(e))
                self.update_status("Erreur analyse", 0)
    
    def load_challenge_with_resize(self):
        """Charger un dataset challenge avec interface de redimensionnement."""
        # Étape 1: Choisir le dossier d'images
        folder_path = filedialog.askdirectory(title="Choisir dossier dataset challenge")
        if not folder_path:
            return
        
        # Étape 2: Interface pour configuration
        config = self._show_resize_configuration_dialog(folder_path)
        if not config:
            return
        
        # Callback de completion pour appliquer les résultats
        def on_challenge_complete(results):
            if results:
                # Programmer l'exécution dans le thread principal
                self.root.after(0, lambda: self._apply_challenge_results(results))
            else:
                self.root.after(0, lambda: self.update_status("Challenge échoué", 0))
        
        # Étape 3: Traitement via WorkflowManager
        try:
            self.workflow_manager.process_challenge_dataset(
                folder_path, 
                config['resize_factor'], 
                config['gt_file_path'],
                self.update_status,
                on_challenge_complete
            )
            
        except Exception as e:
            messagebox.showerror("Erreur", str(e))
    
    def _apply_challenge_results(self, results):
        """Appliquer les résultats du challenge à l'interface."""
        # Mettre à jour l'état de l'application
        self.test_images = results['test_images']
        self.train_images = results['train_images']
        self.ground_truth = results['ground_truth']
        self.predictions = results['predictions']
        self.resize_factor = results['resize_factor']
        
        # Mode challenge
        self.single_image_mode = False
        self.challenge_mode = True
        
        # Métriques si disponibles
        if 'performance_metrics' in results:
            self.performance_metrics = results['performance_metrics']
        else:
            self.performance_metrics = {}
        
        # Charger la première image si disponible
        if self.test_images:
            self.current_index = 0
            self.load_image(self.test_images[0])
        
        # Mettre à jour l'affichage des métriques
        self.update_metrics_display()
    
    # === MÉTHODES DE NAVIGATION ===
    
    def jump_images(self, delta):
        """Naviguer par saut d'images."""
        if self.test_images:
            new_index = max(0, min(len(self.test_images) - 1, self.current_index + delta))
            self.current_index = new_index
            self.load_image(self.test_images[self.current_index])
    
    def prev_image(self):
        """Image précédente."""
        self.jump_images(-1)
    
    def next_image(self):
        """Image suivante."""
        self.jump_images(1)
    
    def load_image(self, image_path):
        """Charger une image pour affichage."""
        self.current_image_path = image_path
        
        self.current_image = cv2.imread(image_path)
        if self.current_image is None:
            messagebox.showerror("Erreur", f"Impossible charger: {image_path}")
            return
        
        self.update_display()
        
        if self.current_index >= 0:
            self.image_info_var.set(f"Image {self.current_index + 1}/{len(self.test_images)}")
        else:
            self.image_info_var.set(f"Image: {os.path.basename(image_path)}")
    
    def update_display(self):
        """Mettre à jour l'affichage de l'image avec les annotations."""
        if not hasattr(self, 'current_image') or self.current_image is None:
            return
            
        image_name = os.path.basename(self.current_image_path)
        mode = self.display_mode.get()
        
        gt_boxes = self.ground_truth.get(image_name, [])
        pred_boxes = self.predictions.get(image_name, [])
        
        # Utiliser ImageRenderer pour le rendu
        photo = self.image_renderer.render_image_with_boxes(
            self.current_image, gt_boxes, pred_boxes, mode
        )
        
        if photo:
            self.photo = photo  # Garder une référence
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        self.update_metrics_display()
    
    # === MÉTHODES DE MISE À JOUR ===
    
    def update_status(self, status, progress):
        """Mettre à jour le statut et la barre de progression."""
        self.status_var.set(status)
        self.progress['value'] = progress
        self.root.update()
    
    def update_metrics_display(self):
        """Mettre à jour l'affichage des métriques."""
        if self.single_image_mode:
            # Mode image unique
            image_name = os.path.basename(self.current_image_path) if self.current_image_path else ""
            pred_boxes = self.predictions.get(image_name, [])
            metrics_text = self.metrics_formatter.format_single_image_metrics(image_name, pred_boxes)
            
        elif self.challenge_mode:
            # Mode challenge avec redimensionnement
            metrics_text = self.metrics_formatter.format_challenge_metrics(
                self.test_images, self.predictions, self.resize_factor
            )
            
        elif self.performance_metrics:
            # Mode pipeline complet avec métriques
            current_image_info = None
            if hasattr(self, 'current_image_path') and self.current_image_path:
                current_image_info = self.data_processor.get_current_image_info(
                    self.current_image_path, self.ground_truth, self.predictions
                )
            
            metrics_text = self.metrics_formatter.format_pipeline_metrics(
                self.performance_metrics, self.test_images, self.train_images, current_image_info, self.ground_truth
            )
            
        else:
            # Message par défaut
            metrics_text = self.metrics_formatter.format_default_message()
        
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(1.0, metrics_text)
    
    # === MÉTHODES PRIVÉES ===
    
    def _handle_single_image_results(self, results, file_path):
        """Gérer les résultats d'une image unique."""
        self.test_images = results['test_images']
        self.train_images = results['train_images']
        self.ground_truth = results['ground_truth']
        self.predictions = results['predictions']
        self.single_image_mode = True
        self.challenge_mode = False
        self.current_index = 0
        
        image_name = os.path.basename(file_path)
        self.test_count_var.set(f"Image: {image_name}")
        self.train_count_var.set("Mode: Image unique")
        
        self.load_image(file_path)
        self.update_metrics_display()
    

    def _show_resize_configuration_dialog(self, folder_path):
        """Afficher la boîte de dialogue de configuration du redimensionnement."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Configuration Challenge + Redimensionnement")
        dialog.geometry("650x650")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Centrer la fenêtre
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (650 // 2)
        y = (dialog.winfo_screenheight() // 2) - (650 // 2)
        dialog.geometry(f"650x650+{x}+{y}")
        
        # Variable pour stocker le résultat
        result = {'cancelled': True}
        
        # Header
        header_frame = ttk.Frame(dialog)
        header_frame.pack(fill=tk.X, padx=20, pady=20)
        
        ttk.Label(header_frame, text="🔧 CONFIGURATION CHALLENGE + REDIMENSIONNEMENT", 
                 font=("Arial", 14, "bold")).pack(anchor=tk.W)
        
        ttk.Label(header_frame, text=f"Dossier sélectionné: {folder_path}", 
                 font=("Arial", 10), foreground="gray").pack(anchor=tk.W, pady=(5, 0))
        
        # Separator
        ttk.Separator(dialog, orient='horizontal').pack(fill=tk.X, padx=20, pady=10)
        
        # Configuration du redimensionnement
        resize_frame = ttk.LabelFrame(dialog, text="Facteur de Redimensionnement", padding="15")
        resize_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Variable pour le facteur
        resize_var = tk.DoubleVar(value=1.0)
        
        # Slider pour le facteur
        ttk.Label(resize_frame, text="Facteur de redimensionnement:").pack(anchor=tk.W)
        
        scale_frame = ttk.Frame(resize_frame)
        scale_frame.pack(fill=tk.X, pady=5)
        
        scale = ttk.Scale(scale_frame, from_=0.1, to=2.0, variable=resize_var, 
                         orient=tk.HORIZONTAL, length=400)
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        scale_label = ttk.Label(scale_frame, text="1.00x", width=8)
        scale_label.pack(side=tk.RIGHT, padx=(10, 0))
        
        def update_scale_label(*args):
            scale_label.config(text=f"{resize_var.get():.2f}x")
        
        resize_var.trace('w', update_scale_label)
        
        # Boutons de preset
        preset_frame = ttk.Frame(resize_frame)
        preset_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(preset_frame, text="Presets rapides:").pack(anchor=tk.W)
        
        buttons_frame = ttk.Frame(preset_frame)
        buttons_frame.pack(fill=tk.X, pady=5)
        
        def set_resize_factor(factor):
            resize_var.set(factor)
        
        for factor in [0.5, 0.75, 1.0, 1.5, 2.0]:
            ttk.Button(buttons_frame, text=f"{factor}x", width=8,
                      command=lambda f=factor: set_resize_factor(f)).pack(side=tk.LEFT, padx=2)
        
        # Configuration des vérités terrain
        gt_frame = ttk.LabelFrame(dialog, text="Vérités Terrain (Optionnel)", padding="15")
        gt_frame.pack(fill=tk.X, padx=20, pady=10)
        
        gt_var = tk.StringVar()
        
        gt_check_var = tk.BooleanVar()
        gt_check = ttk.Checkbutton(gt_frame, text="Charger fichier de vérités terrain (.mat)", 
                                  variable=gt_check_var)
        gt_check.pack(anchor=tk.W)
        
        gt_file_frame = ttk.Frame(gt_frame)
        gt_file_frame.pack(fill=tk.X, pady=5)
        
        gt_entry = ttk.Entry(gt_file_frame, textvariable=gt_var, state='disabled')
        gt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        def browse_gt_file():
            file_path = filedialog.askopenfilename(
                title="Choisir fichier de vérités terrain",
                filetypes=[("Fichiers MAT", "*.mat"), ("Tous les fichiers", "*.*")]
            )
            if file_path:
                gt_var.set(file_path)
        
        gt_browse_btn = ttk.Button(gt_file_frame, text="Parcourir", 
                                  command=browse_gt_file, state='disabled')
        gt_browse_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        def toggle_gt_widgets():
            state = 'normal' if gt_check_var.get() else 'disabled'
            gt_entry.config(state=state)
            gt_browse_btn.config(state=state)
            if not gt_check_var.get():
                gt_var.set('')
        
        gt_check_var.trace('w', lambda *args: toggle_gt_widgets())
        
        # Avertissements
        warning_frame = ttk.LabelFrame(dialog, text="⚠️ Avertissements", padding="15")
        warning_frame.pack(fill=tk.X, padx=20, pady=10)
        
        warning_text = tk.Text(warning_frame, height=6, wrap=tk.WORD, 
                              font=("Arial", 9), state='disabled')
        warning_text.pack(fill=tk.BOTH, expand=True)
        
        def update_warnings(*args):
            warning_text.config(state='normal')
            warning_text.delete(1.0, tk.END)
            
            factor = resize_var.get()
            if factor < 1.0:
                warning_text.insert(tk.END, f"• RÉDUCTION ({factor:.2f}x): Risque de perdre les petits panneaux\n")
                warning_text.insert(tk.END, "• Les détections YOLO peuvent être moins précises\n")
            elif factor > 1.0:
                warning_text.insert(tk.END, f"• AGRANDISSEMENT ({factor:.2f}x): Peut améliorer la détection\n")
                warning_text.insert(tk.END, "• Temps de traitement plus long\n")
            else:
                warning_text.insert(tk.END, "• TAILLE ORIGINALE: Aucun impact sur les performances YOLO\n")
            
            warning_text.insert(tk.END, "• Les coordonnées seront automatiquement reconverties\n")
            
            if gt_check_var.get() and gt_var.get():
                warning_text.insert(tk.END, "• Métriques de performance seront calculées\n")
            else:
                warning_text.insert(tk.END, "• Mode analyse simple (pas de métriques GT)\n")
            
            warning_text.config(state='disabled')
        
        resize_var.trace('w', update_warnings)
        gt_check_var.trace('w', update_warnings)
        gt_var.trace('w', update_warnings)
        update_warnings()
        
        # Boutons d'action
        action_frame = ttk.Frame(warning_frame)
        action_frame.pack(fill=tk.X, pady=10)
        
        def on_start():
            result.update({
                'cancelled': False,
                'resize_factor': resize_var.get(),
                'gt_file_path': gt_var.get() if gt_check_var.get() else None
            })
            dialog.destroy()
        
        def on_cancel():
            result['cancelled'] = True
            dialog.destroy()
        
        ttk.Button(action_frame, text="🚀 COMMENCER LA PRÉDICTION", 
                  command=on_start).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(action_frame, text="❌ Annuler", 
                  command=on_cancel).pack(side=tk.LEFT)
        
        # Raccourcis clavier
        dialog.bind('<Return>', lambda e: on_start())
        dialog.bind('<Escape>', lambda e: on_cancel())
        
        # Attendre la fermeture du dialogue
        dialog.wait_window()
        
        return None if result['cancelled'] else {
            'resize_factor': result['resize_factor'],
            'gt_file_path': result['gt_file_path']
        }

    def _apply_pipeline_results(self, results):
        """Appliquer les résultats du pipeline complet à l'interface."""
        if results:
            # Mettre à jour l'état de l'application
            self.test_images = results['test_images']
            self.train_images = results['train_images']
            self.ground_truth = results['ground_truth']
            self.predictions = results['predictions']
            self.performance_metrics = results['performance_metrics']
            
            # Réinitialiser les modes
            self.single_image_mode = False
            self.challenge_mode = False
            
            # Charger la première image si disponible
            if self.test_images:
                self.current_index = 0
                self.load_image(self.test_images[0])
            
            # Mettre à jour l'affichage des métriques
            self.update_metrics_display()
        else:
            self.update_status("Pipeline échoué", 0)


def main():
    """Point d'entrée principal."""
    root = tk.Tk()
    app = MetroProjectMainGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main() 