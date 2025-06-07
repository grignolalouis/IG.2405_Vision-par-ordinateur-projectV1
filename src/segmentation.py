"""
Module de segmentation des panneaux de métro
"""

import cv2
import numpy as np
from src.constants import METRO_COLORS, SEG_PARAMS


class MetroSegmenter:
    def __init__(self, params=None):
        self.params = SEG_PARAMS.copy()
        if params:
            self.params.update(params)
            
        self._create_color_ranges()
    
    def _create_color_ranges(self):
        self.color_ranges = {}
        
        tolerance = self.params['color_tolerance']
        
        for line_num, color_info in METRO_COLORS.items():
            rgb = np.uint8([[color_info['rgb']]])
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)[0, 0]

            self.color_ranges[line_num] = {
                'lower': np.array([
                    max(0, lab[0] - tolerance),
                    max(0, lab[1] - tolerance), 
                    max(0, lab[2] - tolerance)
                ]),
                'upper': np.array([
                    min(255, lab[0] + tolerance),
                    min(255, lab[1] + tolerance),
                    min(255, lab[2] + tolerance)
                ])
            }
    
    def create_color_mask(self, lab_image, line_num):
        if line_num not in self.color_ranges:
            return np.zeros(lab_image.shape[:2], dtype=np.uint8)
            
        color_range = self.color_ranges[line_num]
        return cv2.inRange(lab_image, color_range['lower'], color_range['upper'])
    
    def apply_morphology(self, mask):
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            self.params['morph_kernel_close']
        )
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            self.params['morph_kernel_open']
        )
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
        
        return opened
    
    def filter_by_circularity(self, contours):
        filtered = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.params['min_area'] or area > self.params['max_area']:
                continue
                
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if (self.params['min_circularity'] <= circularity <= 
                self.params['max_circularity']):
                filtered.append(contour)
        
        return filtered
    
    def find_candidates(self, lab_image):
        candidates = []
        
        for line_num in METRO_COLORS.keys():
            mask = self.create_color_mask(lab_image, line_num)

            mask = self.apply_morphology(mask)
            
            contours, _ = cv2.findContours(
                mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            valid_contours = self.filter_by_circularity(contours)

            for contour in valid_contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                margin = self.params.get('roi_margin', 2)
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = w + 2 * margin
                h = h + 2 * margin
                
                candidates.append({
                    'bbox': (x, y, w, h),
                    'contour': contour,
                    'line_num': line_num,
                    'area': cv2.contourArea(contour)
                })
        
        return candidates
    
    def apply_nms(self, candidates):
        if not candidates:
            return []
            
        candidates = sorted(candidates, key=lambda x: x['area'], reverse=True)
        
        keep = []
        
        for i, cand1 in enumerate(candidates):
            x1, y1, w1, h1 = cand1['bbox']
            
            should_keep = True
            
            for cand2 in keep:
                x2, y2, w2, h2 = cand2['bbox']
                
                x_left = max(x1, x2)
                y_top = max(y1, y2)
                x_right = min(x1 + w1, x2 + w2)
                y_bottom = min(y1 + h1, y2 + h2)
                
                if x_right > x_left and y_bottom > y_top:

                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    area1 = w1 * h1
                    area2 = w2 * h2
                    union_area = area1 + area2 - intersection_area
                    
                    iou = intersection_area / union_area
                    
                    if iou > self.params['nms_threshold']:
                        should_keep = False
                        break
            
            if should_keep:
                keep.append(cand1)
        
        return keep #on enlève les doublons
    
    def segment(self, lab_image):
        candidates = self.find_candidates(lab_image)
        candidates = self.apply_nms(candidates)

        rois = []
        for cand in candidates:
            x, y, w, h = cand['bbox']
            rois.append({
                'bbox': (x, y, x + w, y + h),  
                'line_num_color': cand['line_num'], 
                'confidence': 1.0 
            })
        
        return rois 