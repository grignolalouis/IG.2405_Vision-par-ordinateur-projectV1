import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from src.constants import METRO_COLORS, GUI_PARAMS
from .evaluation import MetricsEvaluator


class ImageRenderer:
    def __init__(self):
        self.max_image_size = GUI_PARAMS['max_image_size']
        self.colors = GUI_PARAMS['colors']
        self.iou_threshold = GUI_PARAMS['iou_threshold']
        self.evaluator = MetricsEvaluator()
    
    def render_image_with_boxes(self, image, ground_truth_boxes=None, prediction_boxes=None, 
                               display_mode="comparison"):
        if image is None:
            return None
            
        display_img = image.copy()
        
        if display_mode == "gt" or display_mode == "comparison":
            if ground_truth_boxes:
                display_img = self._draw_ground_truth_boxes(display_img, ground_truth_boxes)
        
        if display_mode == "pred" or display_mode == "comparison":
            if prediction_boxes:
                display_img = self._draw_prediction_boxes(
                    display_img, prediction_boxes, ground_truth_boxes, display_mode
                )
        
        return self._prepare_for_display(display_img)
    
    def _draw_ground_truth_boxes(self, image, gt_boxes):
        for gt in gt_boxes:
            cv2.rectangle(image, 
                         (gt['xmin'], gt['ymin']), 
                         (gt['xmax'], gt['ymax']), 
                         self.colors['gt'], 2)
            cv2.putText(image, f"GT: {gt['line']}", 
                       (gt['xmin'], gt['ymin']-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['gt'], 2)
        return image
    
    def _draw_prediction_boxes(self, image, pred_boxes, gt_boxes=None, display_mode="pred"):
        for i, pred in enumerate(pred_boxes):
            color = self.colors['pred_fp']
            label_suffix = " (FP)" 
            thickness = 2
            
            if display_mode == "comparison" and gt_boxes:
                best_iou, matched_gt = self._find_best_matching_gt(pred, gt_boxes)
                
                if best_iou > self.iou_threshold: 
                    if pred['line'] == matched_gt['line']:
                        color = self.colors['pred_tp']
                        label_suffix = f" (TP, IoU={best_iou:.2f})"
                        thickness = 3
                    else:
                        color = self.colors['pred_wc']
                        label_suffix = f" (WC, IoU={best_iou:.2f})" 
                else:
                    label_suffix = f" (FP, IoU={best_iou:.2f})"
            else:
                color = self.colors['pred_tp']
                label_suffix = ""
                thickness = 3
            
            cv2.rectangle(image, 
                         (pred['xmin'], pred['ymin']), 
                         (pred['xmax'], pred['ymax']), 
                         color, thickness)

            text = f"P{i+1}: L{pred['line']} ({pred['confidence']:.2f}){label_suffix}"
            cv2.putText(image, text, 
                       (pred['xmin'], pred['ymax']+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image
    
    def _find_best_matching_gt(self, pred_box, gt_boxes):
        best_iou = 0
        matched_gt = None
        
        for gt in gt_boxes:
            iou = self.evaluator.calculate_iou(pred_box, gt)
            if iou > best_iou:
                best_iou = iou
                matched_gt = gt
        
        return best_iou, matched_gt
    
    def _prepare_for_display(self, image):
        display_img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        h, w = display_img_rgb.shape[:2]
        if max(h, w) > self.max_image_size:
            scale = self.max_image_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            display_img_rgb = cv2.resize(display_img_rgb, (new_w, new_h))
        
        pil_img = Image.fromarray(display_img_rgb)
        return ImageTk.PhotoImage(pil_img)
    
    def get_line_color_info(self, line_num):
        return METRO_COLORS.get(line_num, {'name': 'Inconnue', 'color': '#808080'}) 