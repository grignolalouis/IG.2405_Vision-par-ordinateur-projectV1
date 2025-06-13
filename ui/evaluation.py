import os
import numpy as np
from src.constants import GUI_PARAMS


class MetricsEvaluator:
    def __init__(self):
        self.iou_threshold = GUI_PARAMS['iou_threshold']
    
    def calculate_iou(self, box1, box2):
        x1_min, y1_min = box1['xmin'], box1['ymin']
        x1_max, y1_max = box1['xmax'], box1['ymax']
        x2_min, y2_min = box2['xmin'], box2['ymin']
        x2_max, y2_max = box2['xmax'], box2['ymax']
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        
        union_area = box1_area + box2_area - inter_area
        
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area
    
    def calculate_performance_metrics(self, test_images, ground_truth, predictions):
        total_gt = 0
        total_pred = 0
        tp_detection = 0
        tp_classification = 0
        
        line_metrics = {}
        for line_num in range(1, 15):
            line_metrics[line_num] = {
                'tp': 0, 'fp': 0, 'fn': 0,
                'gt_count': 0, 'pred_count': 0
            }
        
        for image_path in test_images:
            image_name = os.path.basename(image_path)
            gt_boxes = ground_truth.get(image_name, [])
            pred_boxes = predictions.get(image_name, [])
            
            total_gt += len(gt_boxes)
            total_pred += len(pred_boxes)
            
            for gt in gt_boxes:
                line_metrics[gt['line']]['gt_count'] += 1
            
            for pred in pred_boxes:
                line_metrics[pred['line']]['pred_count'] += 1
            
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
                
                if best_iou > 0.5:
                    tp_detection += 1
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(i)
                    
                    gt_matched = gt_boxes[best_gt_idx]
                    if pred['line'] == gt_matched['line']:
                        tp_classification += 1
                        line_metrics[pred['line']]['tp'] += 1
                    else:
                        line_metrics[pred['line']]['fp'] += 1
                        line_metrics[gt_matched['line']]['fn'] += 1
                else:
                    line_metrics[pred['line']]['fp'] += 1
            
            for j, gt in enumerate(gt_boxes):
                if j not in matched_gt:
                    line_metrics[gt['line']]['fn'] += 1
        
        fp_detection = total_pred - tp_detection
        fn_detection = total_gt - tp_detection
        
        fp_classification = total_pred - tp_classification
        fn_classification = total_gt - tp_classification
        
        precision_detection = tp_detection / total_pred if total_pred > 0 else 0
        recall_detection = tp_detection / total_gt if total_gt > 0 else 0
        f1_detection = 2 * precision_detection * recall_detection / (precision_detection + recall_detection) if (precision_detection + recall_detection) > 0 else 0
        accuracy_detection = tp_detection / (tp_detection + fn_detection) if (tp_detection + fn_detection) > 0 else 0
        
        precision_classification = tp_classification / total_pred if total_pred > 0 else 0
        recall_classification = tp_classification / total_gt if total_gt > 0 else 0
        f1_classification = 2 * precision_classification * recall_classification / (precision_classification + recall_classification) if (precision_classification + recall_classification) > 0 else 0
        accuracy_classification = tp_classification / (tp_classification + fn_classification) if (tp_classification + fn_classification) > 0 else 0
        
        for line_num in line_metrics:
            metrics = line_metrics[line_num]
            if metrics['tp'] + metrics['fp'] > 0:
                metrics['precision'] = metrics['tp'] / (metrics['tp'] + metrics['fp'])
            else:
                metrics['precision'] = 0
            
            if metrics['tp'] + metrics['fn'] > 0:
                metrics['recall'] = metrics['tp'] / (metrics['tp'] + metrics['fn'])
                metrics['accuracy'] = metrics['tp'] / (metrics['tp'] + metrics['fn'])
            else:
                metrics['recall'] = 0
                metrics['accuracy'] = 0
            
            if metrics['precision'] + metrics['recall'] > 0:
                metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
            else:
                metrics['f1'] = 0
        
        active_lines = [line_num for line_num in line_metrics 
                       if line_metrics[line_num]['gt_count'] > 0 or line_metrics[line_num]['pred_count'] > 0]
        
        if active_lines:
            macro_precision = sum(line_metrics[line_num]['precision'] for line_num in active_lines) / len(active_lines)
            macro_recall = sum(line_metrics[line_num]['recall'] for line_num in active_lines) / len(active_lines)
            macro_f1 = sum(line_metrics[line_num]['f1'] for line_num in active_lines) / len(active_lines)
            macro_accuracy = sum(line_metrics[line_num]['accuracy'] for line_num in active_lines) / len(active_lines)
            
            total_support = sum(line_metrics[line_num]['gt_count'] for line_num in active_lines)
            if total_support > 0:
                weighted_precision = sum(line_metrics[line_num]['precision'] * line_metrics[line_num]['gt_count'] 
                                       for line_num in active_lines) / total_support
                weighted_recall = sum(line_metrics[line_num]['recall'] * line_metrics[line_num]['gt_count'] 
                                    for line_num in active_lines) / total_support
                weighted_f1 = sum(line_metrics[line_num]['f1'] * line_metrics[line_num]['gt_count'] 
                                for line_num in active_lines) / total_support
                weighted_accuracy = sum(line_metrics[line_num]['accuracy'] * line_metrics[line_num]['gt_count'] 
                                      for line_num in active_lines) / total_support
            else:
                weighted_precision = weighted_recall = weighted_f1 = weighted_accuracy = 0
        else:
            macro_precision = macro_recall = macro_f1 = macro_accuracy = 0
            weighted_precision = weighted_recall = weighted_f1 = weighted_accuracy = 0
        
        return {
            'detection': {
                'precision': precision_detection,
                'recall': recall_detection,
                'f1': f1_detection,
                'accuracy': accuracy_detection,
                'tp': tp_detection,
                'fp': fp_detection,
                'fn': fn_detection
            },
            'classification': {
                'precision': precision_classification,
                'recall': recall_classification,
                'f1': f1_classification,
                'accuracy': accuracy_classification,
                'tp': tp_classification,
                'fp': fp_classification,
                'fn': fn_classification
            },
            'by_line': line_metrics,
            'macro': {
                'precision': macro_precision,
                'recall': macro_recall,
                'f1': macro_f1,
                'accuracy': macro_accuracy,
                'classes': len(active_lines)
            },
            'weighted': {
                'precision': weighted_precision,
                'recall': weighted_recall,
                'f1': weighted_f1,
                'accuracy': weighted_accuracy,
                'support': total_support
            },
            'totals': {
                'gt_boxes': total_gt,
                'pred_boxes': total_pred
            }
        }
    
    def get_current_image_info(self, current_image_path, ground_truth, predictions):
        if not current_image_path:
            return None
        
        image_name = os.path.basename(current_image_path)
        gt_boxes = ground_truth.get(image_name, [])
        pred_boxes = predictions.get(image_name, [])
        
        avg_confidence = None
        if pred_boxes:
            avg_confidence = sum(p['confidence'] for p in pred_boxes) / len(pred_boxes)
        
        return {
            'name': image_name,
            'gt_count': len(gt_boxes),
            'pred_count': len(pred_boxes),
            'avg_confidence': avg_confidence
        } 