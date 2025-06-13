import os
import cv2
import threading
import pickle
from tkinter import messagebox
from src.detector import MetroSignDetector
from src.data_loader import DataLoader
from src.constants import GUI_PARAMS
from .data_processor import DataProcessor


class WorkflowManager:
    def __init__(self, status_callback=None):
        self.detector = MetroSignDetector()
        self.loader = DataLoader()
        self.data_processor = DataProcessor()
        self.status_callback = status_callback
        
        self.training_completed = False
        self.model_saved = False
        self.prediction_completed = False
    
    def run_complete_pipeline(self, progress_callbacks=None, completion_callback=None):
        def pipeline_thread():
            try:
                results = {
                    'train_images': [],
                    'test_images': [],
                    'ground_truth': {},
                    'predictions': {},
                    'performance_metrics': {}
                }
                
                self._update_status("Chargement donnees...", 10, progress_callbacks)
                train_data = self.loader.load_training_data()
                test_data = self.loader.load_test_data()
                
                results['train_images'] = [path for path, _ in train_data]
                results['test_images'] = [path for path, _ in test_data]
                
                for path, annotations in train_data + test_data:
                    results['ground_truth'][os.path.basename(path)] = annotations
                
                self._update_counts(len(train_data), len(test_data), progress_callbacks)
                
                self._update_status("Entrainement...", 30, progress_callbacks)
                self._update_train_status("Entrainement: En cours", progress_callbacks)
                
                if train_data:
                    train_samples = self._prepare_training_samples(train_data)
                    self.detector.train_classifier(train_samples)
                    self.training_completed = True
                    self._update_train_status(f"Entrainement: Termine ({len(train_samples)} ROIs)", progress_callbacks)
                else:
                    self._update_train_status("Entrainement: Echec", progress_callbacks)
                
                self._update_status("Sauvegarde...", 50, progress_callbacks)
                self._update_save_status("Sauvegarde: En cours", progress_callbacks)
                
                os.makedirs("models", exist_ok=True)
                try:
                    with open(GUI_PARAMS['model_save_path'], 'wb') as f:
                        pickle.dump(self.detector.classifier, f)
                    self.model_saved = True
                    self._update_save_status("Sauvegarde: Termine", progress_callbacks)
                except Exception as e:
                    self._update_save_status("Sauvegarde: Erreur", progress_callbacks)
                
                self._update_status("Prediction test...", 60, progress_callbacks)
                self._update_pred_status("Prediction: En cours", progress_callbacks)
                
                results['predictions'] = self._run_predictions_on_test_data(test_data, progress_callbacks)
                
                self.prediction_completed = True
                self._update_pred_status(f"Prediction: Termine ({len(test_data)} images)", progress_callbacks)
                
                self._update_status("Calcul metriques...", 95, progress_callbacks)
                results['performance_metrics'] = self.data_processor.calculate_performance_metrics(
                    results['test_images'], results['ground_truth'], results['predictions']
                )
                
                self._update_status("Pipeline termine", 100, progress_callbacks)
                
                if completion_callback:
                    completion_callback(results)
                
                return results
                
            except Exception as e:
                self._update_status(f"Erreur pipeline: {str(e)}", 0, progress_callbacks)
                messagebox.showerror("Erreur", f"Erreur pipeline: {str(e)}")
                if completion_callback:
                    completion_callback(None)
                return None
        
        thread = threading.Thread(target=pipeline_thread, daemon=True)
        thread.start()
        return thread
    
    def process_single_image(self, image_path):
        try:
            result = self.detector.processOneMetroImage(image_path)
            image_name = os.path.basename(image_path)
            
            predictions = {image_name: []}
            for det in result['detections']:
                xmin, ymin, xmax, ymax = det['bbox']
                predictions[image_name].append({
                    'xmin': int(xmin),
                    'ymin': int(ymin),
                    'xmax': int(xmax),
                    'ymax': int(ymax),
                    'line': det['line_num'],
                    'confidence': det['confidence']
                })
            
            return {
                'test_images': [image_path],
                'train_images': [],
                'ground_truth': {},
                'predictions': predictions,
                'single_image_mode': True
            }
            
        except Exception as e:
            raise Exception(f"Erreur analyse: {str(e)}")
    
    def process_challenge_dataset(self, resize_factor=1.0, progress_callback=None, completion_callback=None):
        
        def challenge_thread():
            try:
                folder_path = r"D:\Isep 2025-2026\IG.2405 - Vision par ordinateur\projetV1\challenge\BD_CHALLENGE"
                challenge_dir = r"D:\Isep 2025-2026\IG.2405 - Vision par ordinateur\projetV1\challenge"
                
                if progress_callback:
                    progress_callback("Vérification dossier challenge...", 5)
                
                if not os.path.exists(folder_path):
                    raise Exception(f"Dossier challenge non trouvé: {folder_path}")
                
                if progress_callback:
                    progress_callback("Chargement images...", 10)
                
                image_files = []
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    image_files.extend([f for f in os.listdir(folder_path) 
                                      if f.lower().endswith(ext)])
                
                if not image_files:
                    raise Exception(f"Aucune image trouvée dans le dossier: {folder_path}")
                
                original_images = [os.path.join(folder_path, f) for f in image_files]
                
                if progress_callback:
                    progress_callback("Recherche fichier vérités terrain...", 12)
                
                gt_file_path = None
                if os.path.exists(challenge_dir):
                    for file in os.listdir(challenge_dir):
                        if file.lower().endswith('.mat'):
                            gt_file_path = os.path.join(challenge_dir, file)
                            if progress_callback:
                                progress_callback(f"Fichier .mat trouvé: {file}", 13)
                            break
                
                ground_truth = {}
                if gt_file_path and os.path.exists(gt_file_path):
                    if progress_callback:
                        progress_callback(f"Chargement vérités terrain: {os.path.basename(gt_file_path)}", 15)
                    try:
                        ground_truth = self.data_processor.load_ground_truth_file(gt_file_path, original_images)
                        if progress_callback:
                            progress_callback(f"Vérités terrain chargées: {len(ground_truth)} images", 18)
                    except Exception as e:
                        if progress_callback:
                            progress_callback(f"Erreur chargement GT: {str(e)}", 18)
                        ground_truth = {}
                else:
                    if progress_callback:
                        progress_callback("Aucun fichier .mat trouvé - Mode analyse simple", 15)
                
                predictions = {}
                total_images = len(original_images)
                
                for i, original_path in enumerate(original_images):
                    original_image = cv2.imread(original_path)
                    if original_image is None:
                        continue
                    
                    image_name = os.path.basename(original_path)
                    
                    if resize_factor != 1.0:
                        h, w = original_image.shape[:2]
                        new_w = int(w * resize_factor)
                        new_h = int(h * resize_factor)
                        resized_image = cv2.resize(original_image, (new_w, new_h))
                    else:
                        resized_image = original_image.copy()
                    
                    result = self.detector.processOneMetroImage_from_array(resized_image)
                    
                    predictions[image_name] = []
                    for det in result['detections']:
                        xmin, ymin, xmax, ymax = det['bbox']
                        
                        
                        if resize_factor != 1.0:
                            xmin = int(xmin / resize_factor)
                            ymin = int(ymin / resize_factor)
                            xmax = int(xmax / resize_factor)
                            ymax = int(ymax / resize_factor)
                        
                        predictions[image_name].append({
                            'xmin': xmin,
                            'ymin': ymin,
                            'xmax': xmax,
                            'ymax': ymax,
                            'line': det['line_num'],
                            'confidence': det['confidence']
                        })
                    
                    if progress_callback:
                        progress = 20 + int((i + 1) / total_images * 75)
                        progress_callback(f"Analyse: {i+1}/{total_images}", progress)
                
                results = {
                    'test_images': original_images,
                    'train_images': [],
                    'ground_truth': ground_truth,
                    'predictions': predictions,
                    'challenge_mode': True,
                    'resize_factor': resize_factor
                }
                
                if ground_truth:
                    if progress_callback:
                        progress_callback("Calcul métriques...", 95)
                    results['performance_metrics'] = self.data_processor.calculate_performance_metrics(
                        results['test_images'], results['ground_truth'], results['predictions']
                    )
                
                if progress_callback:
                    progress_callback("Analyse terminée", 100)
                
                if completion_callback:
                    completion_callback(results)
                
                return results
                
            except Exception as e:
                if progress_callback:
                    progress_callback("Erreur traitement", 0)
                if completion_callback:
                    completion_callback(None)
                raise Exception(f"Erreur lors du traitement: {str(e)}")
        
        thread = threading.Thread(target=challenge_thread, daemon=True)
        thread.start()
        return thread
    
    def _prepare_training_samples(self, train_data):
        train_samples = []
        for image_path, annotations in train_data:
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            prep_result = self.detector.preprocessor.preprocess(image)
            bgr_image = prep_result['bgr']
            
            for ann in annotations:
                xmin, ymin, xmax, ymax = ann['xmin'], ann['ymin'], ann['xmax'], ann['ymax']
                line_num = ann['line']
                
                if xmax > xmin and ymax > ymin:
                    roi = bgr_image[ymin:ymax, xmin:xmax]
                    if roi.size > 0:
                        train_samples.append((roi, line_num))
        
        return train_samples
    
    def _run_predictions_on_test_data(self, test_data, progress_callbacks):
        predictions = {}
        total_test = len(test_data)
        
        for i, (image_path, _) in enumerate(test_data):
            try:
                result = self.detector.processOneMetroImage(image_path)
                image_name = os.path.basename(image_path)
                
                predictions[image_name] = []
                for det in result['detections']:
                    xmin, ymin, xmax, ymax = det['bbox']
                    predictions[image_name].append({
                        'xmin': int(xmin),
                        'ymin': int(ymin),
                        'xmax': int(xmax),
                        'ymax': int(ymax),
                        'line': det['line_num'],
                        'confidence': det['confidence']
                    })
                
                progress = 60 + int((i + 1) / total_test * 30)
                self._update_status(f"Prediction: {i+1}/{total_test}", progress, progress_callbacks)
                
            except Exception as e:
                pass
        
        return predictions
    
    def _update_status(self, message, progress, callbacks):

        if callbacks and 'status_callback' in callbacks:
            callbacks['status_callback'](message, progress)
    
    def _update_train_status(self, message, callbacks):

        if callbacks and 'train_status_callback' in callbacks:
            callbacks['train_status_callback'](message)
    
    def _update_save_status(self, message, callbacks):

        if callbacks and 'save_status_callback' in callbacks:
            callbacks['save_status_callback'](message)
    
    def _update_pred_status(self, message, callbacks):

        if callbacks and 'pred_status_callback' in callbacks:
            callbacks['pred_status_callback'](message)
    
    def _update_counts(self, train_count, test_count, callbacks):

        if callbacks and 'count_callbacks' in callbacks:
            count_cbs = callbacks['count_callbacks']
            if 'train_count' in count_cbs:
                count_cbs['train_count'](f"Images entrainement: {train_count}")
            if 'test_count' in count_cbs:
                count_cbs['test_count'](f"Images test: {test_count}")
    
    def apply_single_image_results(self, results, file_path):
        return {
            'test_images': results['test_images'],
            'train_images': results['train_images'],
            'ground_truth': results['ground_truth'],
            'predictions': results['predictions'],
            'challenge_predictions': {},
            'challenge_test_images': [],
            'single_image_mode': True,
            'challenge_mode': False,
            'current_index': 0,
            'image_name': os.path.basename(file_path)
        }
    
    def apply_challenge_results(self, results):
        processed_results = {
            'test_images': results['test_images'],
            'train_images': results['train_images'],
            'ground_truth': results['ground_truth'],
            'predictions': results['predictions'],
            'resize_factor': results['resize_factor'],
            'challenge_predictions': results['predictions'].copy(),
            'challenge_test_images': results['test_images'].copy(),
            'single_image_mode': False,
            'challenge_mode': True,
            'performance_metrics': results.get('performance_metrics', {})
        }
        return processed_results
    
    def apply_pipeline_results(self, results):
        if not results:
            return None
            
        return {
            'test_images': results['test_images'],
            'train_images': results['train_images'],
            'ground_truth': results['ground_truth'],
            'predictions': results['predictions'],
            'performance_metrics': results['performance_metrics'],
            'challenge_predictions': {},
            'challenge_test_images': [],
            'single_image_mode': False,
            'challenge_mode': False
        } 