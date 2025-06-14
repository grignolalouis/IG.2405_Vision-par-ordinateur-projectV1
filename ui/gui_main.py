"""
Module de l'interface utilisateur principale du système de détection de panneaux de métro.

Ce module implémente la classe MetroProjectMainGUI qui constitue l'interface graphique
principale de l'application. Elle orchestre tous les composants de l'interface utilisateur
et gère les interactions avec les modules de traitement d'images.

Auteur: LGrignola
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
from .dialog_manager import DialogManager


class MetroProjectMainGUI: 
    def __init__(self, root):
        self.root = root
        self.root.title("Detection Panneaux Metro")
        self.root.geometry(GUI_PARAMS['window_size'])
        
        self.workflow_manager = WorkflowManager()
        self.image_renderer = ImageRenderer()
        self.metrics_formatter = MetricsFormatter()
        self.data_processor = DataProcessor()
        self.dialog_manager = DialogManager(root)
        
        self.ground_truth = {}
        self.predictions = {}
        self.challenge_predictions = {}
        self.challenge_test_images = []
        self.train_images = []
        self.test_images = []
        self.current_index = 0
        self.current_image = None
        self.current_image_path = None
        self.performance_metrics = {}
        
        self.single_image_mode = False
        self.challenge_mode = False
        self.resize_factor = 1.0
        
        self.train_status_var = tk.StringVar(value="Entraînement: En attente")
        self.save_status_var = tk.StringVar(value="Sauvegarde: En attente")
        self.pred_status_var = tk.StringVar(value="Prédiction: En attente")
        
        self.create_interface()
        self.update_metrics_display()
        
    def create_interface(self):
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
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        title_label = ttk.Label(header_frame, text="Détection et classification des panneaux du Métro Parisien", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, sticky=tk.W)
        
        progress_frame = ttk.Frame(header_frame)
        progress_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(8, 0))
        
        self.status_var = tk.StringVar(value="Système prêt")
        status_label = ttk.Label(progress_frame, textvariable=self.status_var, 
                                font=("Arial", 10))
        status_label.grid(row=0, column=0, sticky=tk.W)
        
        data_info_frame = ttk.Frame(progress_frame)
        data_info_frame.grid(row=0, column=1, sticky=tk.E, padx=(20, 0))
        
        self.train_count_var = tk.StringVar(value="Images entraînement: -")
        ttk.Label(data_info_frame, textvariable=self.train_count_var, 
                 font=("Arial", 9)).grid(row=0, column=0, padx=(0, 15), sticky=tk.W)
        
        self.test_count_var = tk.StringVar(value="Images test: -")
        ttk.Label(data_info_frame, textvariable=self.test_count_var,
                 font=("Arial", 9)).grid(row=0, column=1, sticky=tk.W)
        
        self.progress = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.progress.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(4, 0))
        
        header_frame.columnconfigure(0, weight=1)
        progress_frame.columnconfigure(0, weight=1)
        progress_frame.columnconfigure(1, weight=0)
    
    def create_action_panel(self, parent):
        action_frame = ttk.LabelFrame(parent, text="ACTIONS", padding="15")
        action_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        buttons_frame = ttk.Frame(action_frame)
        buttons_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        ttk.Button(buttons_frame, text="Challenge", 
                  command=self.load_challenge_with_resize).grid(row=0, column=0, padx=(0, 10), sticky=(tk.W, tk.E))
        
        ttk.Button(buttons_frame, text="Démonstration Pipeline", 
                  command=self.start_complete_pipeline).grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        
        ttk.Button(buttons_frame, text="Image Unique", 
                  command=self.load_single_image).grid(row=0, column=2, padx=(10, 0), sticky=(tk.W, tk.E))
        
        action_frame.columnconfigure(0, weight=1)
        buttons_frame.columnconfigure(0, weight=1)
        buttons_frame.columnconfigure(1, weight=1)
        buttons_frame.columnconfigure(2, weight=1)
    
    def create_main_content(self, parent):
        content_frame = ttk.Frame(parent)
        content_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        left_panel = ttk.Frame(content_frame)
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        self.create_metrics_panel(left_panel)
        
        right_panel = ttk.LabelFrame(content_frame, text="VISUALISATION", padding="10")
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.create_visualization_panel(right_panel)
        
        content_frame.columnconfigure(0, weight=1, minsize=400)
        content_frame.columnconfigure(1, weight=2, minsize=600)
        content_frame.rowconfigure(0, weight=1)
        
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(0, weight=1)
    

    def create_metrics_panel(self, parent):
        metrics_frame = ttk.LabelFrame(parent, text="RÉSULTATS & MÉTRIQUES", padding="12")
        metrics_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        buttons_frame = ttk.Frame(metrics_frame)
        buttons_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        
        buttons_frame.columnconfigure(0, weight=1)
        
        self.export_button = ttk.Button(buttons_frame, text=" Export .mat", 
                                        command=self.export_predictions_mat,
                                        width=12,
                                        state='disabled')
        self.export_button.grid(row=0, column=1, sticky=tk.E, padx=(0, 5))
        
        self.export_results_button = ttk.Button(buttons_frame, text=" Export Results", 
                                               command=self.export_results_txt,
                                               width=14,
                                               state='disabled')
        self.export_results_button.grid(row=0, column=2, sticky=tk.E, padx=(0, 5))
        
        info_button = ttk.Button(buttons_frame, text="ℹ Info", 
                                command=self.show_info_text,
                                width=10)
        info_button.grid(row=0, column=3, sticky=tk.E)
        
        text_frame = ttk.Frame(metrics_frame)
        text_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.metrics_text = tk.Text(text_frame, 
                                   font=("Consolas", 10), 
                                   wrap=tk.WORD,
                                   bg='#f8f9fa',
                                   relief=tk.SUNKEN,
                                   borderwidth=1,
                                   padx=8,
                                   pady=8)
        self.metrics_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        metrics_scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.metrics_text.yview)
        metrics_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.metrics_text.configure(yscrollcommand=metrics_scroll.set)
        
        metrics_frame.columnconfigure(0, weight=1)
        metrics_frame.rowconfigure(1, weight=1)
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
    
    def create_visualization_panel(self, parent):
        mode_frame = ttk.LabelFrame(parent, text="Mode d'affichage", padding="10")
        mode_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.display_mode = tk.StringVar(value="comparison")
        
        controls_frame = ttk.Frame(mode_frame)
        controls_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        ttk.Radiobutton(controls_frame, text="Vérité Terrain", variable=self.display_mode, 
                       value="gt", command=self.update_display).grid(row=0, column=0, padx=(0, 15), sticky=tk.W)
        ttk.Radiobutton(controls_frame, text="Prédictions", variable=self.display_mode, 
                       value="pred", command=self.update_display).grid(row=0, column=1, padx=15, sticky=tk.W)
        ttk.Radiobutton(controls_frame, text="Comparaison", variable=self.display_mode, 
                       value="comparison", command=self.update_display).grid(row=0, column=2, padx=(15, 0), sticky=tk.W)
        
        legend_frame = ttk.LabelFrame(parent, text="Légende des couleurs", padding="10")
        legend_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.create_color_legend(legend_frame)
        
        nav_frame = ttk.Frame(parent)
        nav_frame.grid(row=2, column=0, pady=(0, 10), sticky=(tk.W, tk.E))
        
        nav_buttons_frame = ttk.Frame(nav_frame)
        nav_buttons_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        ttk.Button(nav_buttons_frame, text=f"<<-{GUI_PARAMS['jump_size']}", 
                  command=lambda: self.jump_images(-GUI_PARAMS['jump_size'])).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(nav_buttons_frame, text="< Précédent", 
                  command=self.prev_image).grid(row=0, column=1, padx=5)
        
        self.image_info_var = tk.StringVar(value="Aucune image")
        info_label = ttk.Label(nav_buttons_frame, textvariable=self.image_info_var, 
                              font=("Arial", 11, "bold"))
        info_label.grid(row=0, column=2, padx=20)
        
        ttk.Button(nav_buttons_frame, text="Suivant >", 
                  command=self.next_image).grid(row=0, column=3, padx=5)
        ttk.Button(nav_buttons_frame, text=f"+{GUI_PARAMS['jump_size']}>>", 
                  command=lambda: self.jump_images(GUI_PARAMS['jump_size'])).grid(row=0, column=4, padx=(5, 0))
        
        canvas_frame = ttk.Frame(parent, relief=tk.SUNKEN, borderwidth=1)
        canvas_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.canvas = tk.Canvas(canvas_frame, bg="#f8f9fa", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        canvas_scroll_y = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        canvas_scroll_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        canvas_scroll_x = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        canvas_scroll_x.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.canvas.configure(yscrollcommand=canvas_scroll_y.set, xscrollcommand=canvas_scroll_x.set)
        
        self.canvas.create_text(400, 300, text="Aucune image chargée\nUtilisez les boutons d'action ci-dessus", 
                               font=("Arial", 12), fill="#6B7280", justify=tk.CENTER)
        
        parent.rowconfigure(3, weight=1)
        parent.columnconfigure(0, weight=1)
        
        mode_frame.columnconfigure(0, weight=1)
        controls_frame.columnconfigure(1, weight=1)
        
        legend_frame.columnconfigure(0, weight=1)
        
        nav_frame.columnconfigure(0, weight=1)
        nav_buttons_frame.columnconfigure(2, weight=1)
        
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
    
    def create_color_legend(self, parent):
        legend_container = ttk.Frame(parent)
        legend_container.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        color_info = [
            (GUI_PARAMS['colors']['gt'], "Vérité Terrain", "■"),
            (GUI_PARAMS['colors']['pred_tp'], "Vrais Positifs", "■"),
            (GUI_PARAMS['colors']['pred_fp'], "Faux Positifs", "■"),
            (GUI_PARAMS['colors']['pred_wc'], "Mauvaise Classe", "■")
        ]
        
        for i, (color_bgr, description, symbol) in enumerate(color_info):
            color_rgb = f"#{color_bgr[2]:02x}{color_bgr[1]:02x}{color_bgr[0]:02x}"
            
            item_frame = ttk.Frame(legend_container)
            item_frame.grid(row=0, column=i, padx=(0, 20), sticky=tk.W)
            
            color_label = tk.Label(item_frame, text=symbol, 
                                 font=("Arial", 14, "bold"),
                                 fg=color_rgb,
                                 bg='SystemButtonFace')
            color_label.grid(row=0, column=0, padx=(0, 5))
            
            desc_label = ttk.Label(item_frame, text=description, 
                                 font=("Arial", 10))
            desc_label.grid(row=0, column=1)
        
        legend_container.columnconfigure(0, weight=1)
        legend_container.columnconfigure(1, weight=1)
        legend_container.columnconfigure(2, weight=1)
        legend_container.columnconfigure(3, weight=1)
        parent.columnconfigure(0, weight=1)
    
    def start_complete_pipeline(self):
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
        
        def on_pipeline_complete(results):
            processed_results = self.workflow_manager.apply_pipeline_results(results)
            self.root.after(0, lambda: self._apply_pipeline_results(processed_results))
        
        self.workflow_manager.run_complete_pipeline(progress_callbacks, on_pipeline_complete)
    
    def load_single_image(self):
        file_path = filedialog.askopenfilename(
            title="Sélectionner image", 
            filetypes=GUI_PARAMS['supported_image_formats']
        )
        
        if file_path:
            try:
                self.update_status("Analyse image...", 30)
                
                results = self.workflow_manager.process_single_image(file_path)
                
                self._handle_single_image_results(results, file_path)
                
                self.update_status("Analyse terminée", 100)
                
            except Exception as e:
                messagebox.showerror("Erreur", str(e))
                self.update_status("Erreur analyse", 0)
    
    def load_challenge_with_resize(self):
        config = self.dialog_manager.show_resize_configuration_dialog()
        if not config:
            return
        
        def on_challenge_complete(results):
            if results:
                processed_results = self.workflow_manager.apply_challenge_results(results)
                self.root.after(0, lambda: self._apply_challenge_results(processed_results))
            else:
                self.root.after(0, lambda: self.update_status("Challenge échoué", 0))
        
        try:
            self.workflow_manager.process_challenge_dataset(
                config['resize_factor'], 
                self.update_status,
                on_challenge_complete
            )
            
        except Exception as e:
            messagebox.showerror("Erreur", str(e))
    
    def _apply_challenge_results(self, processed_results):
        self.test_images = processed_results['test_images']
        self.train_images = processed_results['train_images']
        self.ground_truth = processed_results['ground_truth']
        self.predictions = processed_results['predictions']
        self.resize_factor = processed_results['resize_factor']
        self.challenge_predictions = processed_results['challenge_predictions']
        self.challenge_test_images = processed_results['challenge_test_images']
        self.single_image_mode = processed_results['single_image_mode']
        self.challenge_mode = processed_results['challenge_mode']
        self.performance_metrics = processed_results['performance_metrics']
        
        if self.challenge_predictions:
            self.export_button.config(state='normal')
            self.export_results_button.config(state='normal')
        else:
            self.export_button.config(state='disabled')
            self.export_results_button.config(state='disabled')
        
        if self.test_images:
            self.current_index = 0
            self.load_image(self.test_images[0])
        
        self.update_metrics_display()
    
    def jump_images(self, delta):
        if self.test_images:
            new_index = max(0, min(len(self.test_images) - 1, self.current_index + delta))
            self.current_index = new_index
            self.load_image(self.test_images[self.current_index])
    
    def prev_image(self):
        self.jump_images(-1)
    
    def next_image(self):
        self.jump_images(1)
    
    def load_image(self, image_path):
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
        if not hasattr(self, 'current_image') or self.current_image is None:
            return
            
        image_name = os.path.basename(self.current_image_path)
        mode = self.display_mode.get()
        
        gt_boxes = self.ground_truth.get(image_name, [])
        pred_boxes = self.predictions.get(image_name, [])
        
        photo = self.image_renderer.render_image_with_boxes(
            self.current_image, gt_boxes, pred_boxes, mode
        )
        
        if photo:
            self.photo = photo
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        self.update_metrics_display()
    
    def update_status(self, status, progress):
        self.status_var.set(status)
        self.progress['value'] = progress
        self.root.update()
    
    def update_metrics_display(self):
        if self.single_image_mode:
            image_name = os.path.basename(self.current_image_path) if self.current_image_path else ""
            pred_boxes = self.predictions.get(image_name, [])
            metrics_text = self.metrics_formatter.format_single_image_metrics(image_name, pred_boxes)
            
        elif self.performance_metrics:
            current_image_info = None
            if hasattr(self, 'current_image_path') and self.current_image_path:
                current_image_info = self.data_processor.get_current_image_info(
                    self.current_image_path, self.ground_truth, self.predictions
                )
            
            metrics_text = self.metrics_formatter.format_pipeline_metrics(
                self.performance_metrics, self.test_images, self.train_images, current_image_info, self.ground_truth
            )
            
        elif self.challenge_mode:       
            metrics_text = self.metrics_formatter.format_challenge_metrics(
                self.test_images, self.predictions, self.resize_factor
            )
            
        else:
            metrics_text = self.metrics_formatter.format_default_message()
        
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(1.0, metrics_text)
    
    def show_info_text(self):
        info_text = self.metrics_formatter.format_default_message()
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(1.0, info_text)
    
    def export_predictions_mat(self):
        if not hasattr(self, 'challenge_predictions') or not self.challenge_predictions:
            messagebox.showwarning("Aucune prédiction challenge", 
                                 "Aucune prédiction de challenge disponible à exporter.\n"
                                 "Veuillez d'abord charger et analyser un dataset challenge.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Exporter prédictions challenge",
            defaultextension=".mat",
            initialfile="team12_challenge_predictions.mat",
            filetypes=[("Fichiers MAT", "*.mat"), ("Tous les fichiers", "*.*")]
        )
        
        if not file_path:
            return
        
        result = self.data_processor.export_predictions_to_mat(
            self.challenge_test_images, 
            self.challenge_predictions, 
            file_path
        )
        
        if result['success']:
            messagebox.showinfo("Export réussi", 
                              f"Prédictions challenge exportées avec succès !\n\n"
                              f"Fichier: {file_path}\n"
                              f"Images: {result['images_count']}\n"
                              f"Prédictions: {result['predictions_count']}\n"
                              f"Format: [image_id, y1, y2, x1, x2, line_num]")
        else:
            messagebox.showerror("Erreur d'export", 
                               f"Erreur lors de l'export:\n{result['error']}")
    
    def export_results_txt(self):
        if not hasattr(self, 'challenge_predictions') or not self.challenge_predictions:
            messagebox.showwarning("Aucun résultat challenge", 
                                 "Aucun résultat de challenge disponible à exporter.\n"
                                 "Veuillez d'abord charger et analyser un dataset challenge.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Exporter résultats et métriques",
            defaultextension=".txt",
            initialfile="team12_metrics_results.txt",
            filetypes=[("Fichiers texte", "*.txt"), ("Tous les fichiers", "*.*")]
        )
        
        if not file_path:
            return
        
        result = self.data_processor.export_results_to_txt(
            self.performance_metrics if hasattr(self, 'performance_metrics') else None,
            self.challenge_test_images, 
            self.challenge_predictions, 
            file_path
        )
        
        if result['success']:
            messagebox.showinfo("Export réussi", 
                              f"Résultats et métriques exportés avec succès !\n\n"
                              f"Fichier: {file_path}\n"
                              f"Images: {result['images_count']}\n"
                              f"Prédictions: {result['predictions_count']}\n"
                              f"Contenu: Métriques détaillées + Prédictions par image")
        else:
            messagebox.showerror("Erreur d'export", 
                               f"Erreur lors de l'export:\n{result['error']}")

    def _handle_single_image_results(self, results, file_path):
        processed_results = self.workflow_manager.apply_single_image_results(results, file_path)
        
        self.test_images = processed_results['test_images']
        self.train_images = processed_results['train_images']
        self.ground_truth = processed_results['ground_truth']
        self.predictions = processed_results['predictions']
        self.challenge_predictions = processed_results['challenge_predictions']
        self.challenge_test_images = processed_results['challenge_test_images']
        self.single_image_mode = processed_results['single_image_mode']
        self.challenge_mode = processed_results['challenge_mode']
        self.current_index = processed_results['current_index']
        
        self.export_button.config(state='disabled')
        self.export_results_button.config(state='disabled')
        
        image_name = processed_results['image_name']
        self.test_count_var.set(f"Image: {image_name}")
        self.train_count_var.set("Mode: Image unique")
        
        self.load_image(file_path)
        self.update_metrics_display()
    



    def _apply_pipeline_results(self, processed_results):
        if processed_results:
            self.test_images = processed_results['test_images']
            self.train_images = processed_results['train_images']
            self.ground_truth = processed_results['ground_truth']
            self.predictions = processed_results['predictions']
            self.performance_metrics = processed_results['performance_metrics']
            self.challenge_predictions = processed_results['challenge_predictions']
            self.challenge_test_images = processed_results['challenge_test_images']
            self.single_image_mode = processed_results['single_image_mode']
            self.challenge_mode = processed_results['challenge_mode']
            
            self.export_button.config(state='disabled')
            self.export_results_button.config(state='disabled')
            
            if self.test_images:
                self.current_index = 0
                self.load_image(self.test_images[0])
            
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