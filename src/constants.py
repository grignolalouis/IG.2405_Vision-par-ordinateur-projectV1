"""
Constantes pour la détection des panneaux de métro parisien
"""

DATA_LOADER = {
    'default_data_dir': 'data',
    'image_subdir': 'BD_METRO',
    'docs_subdir': 'docs/progsPython',
    'train_mat_file': 'Apprentissage.mat',
    'test_mat_file': 'Test.mat',
    'train_id_divisor': 3,
    'supported_extensions': ('.jpg', '.jpeg', '.png'),
    'image_name_patterns': [
        r'IM \((\d+)\)\.JPG',
        r'IM\((\d+)\)\.JPG', 
        r'metro(\d+)\.jpg',
        r'image(\d+)\.jpg'
    ],
    'image_name_formats': [
        'IM ({}).JPG',
        'IM({}).JPG',
        'metro{:03d}.jpg',
        'image{}.jpg'
    ]
}

METRO_COLORS = {
    1: {'hex': '#FFBE00', 'rgb': (255, 190, 0), 'name': 'Jaune'},
    2: {'hex': '#0055C8', 'rgb': (0, 85, 200), 'name': 'Bleu'},
    3: {'hex': '#6E6E00', 'rgb': (110, 110, 0), 'name': 'Vert olive'},
    4: {'hex': '#A0006E', 'rgb': (160, 0, 110), 'name': 'Violet'},
    5: {'hex': '#FF5A00', 'rgb': (255, 90, 0), 'name': 'Orange'},
    6: {'hex': '#82DC73', 'rgb': (130, 220, 115), 'name': 'Vert clair'},
    7: {'hex': '#FF82B4', 'rgb': (255, 130, 180), 'name': 'Rose'},
    8: {'hex': '#D282BE', 'rgb': (210, 130, 190), 'name': 'Lilas'},
    9: {'hex': '#D2D200', 'rgb': (210, 210, 0), 'name': 'Jaune-vert'},
    10: {'hex': '#DC9600', 'rgb': (220, 150, 0), 'name': 'Ocre'},
    11: {'hex': '#5A230A', 'rgb': (90, 35, 10), 'name': 'Marron'},
    12: {'hex': '#00643C', 'rgb': (0, 100, 60), 'name': 'Vert foncé'},
    13: {'hex': '#82C8E6', 'rgb': (130, 200, 230), 'name': 'Bleu clair'},
    14: {'hex': '#640082', 'rgb': (100, 0, 130), 'name': 'Violet foncé'}
}

# Paramètres de segmentation - OPTIMISÉS AUTOMATIQUEMENT (Score: 0.883)
SEG_PARAMS = {
    'bilateral_d': 9,  # Optimisé: augmenté pour meilleur lissage
    'bilateral_sigma_color': 30,  # Optimisé: réduit pour préserver détails
    'bilateral_sigma_space': 90,  # Optimisé: augmenté pour lissage spatial
    'morph_kernel_close': (5, 5),  # Optimisé: taille équilibrée pour fermeture
    'morph_kernel_open': (5, 5),  # Optimisé: taille équilibrée pour ouverture
    'min_area': 1500,  # Optimisé: augmenté pour éviter petites détections
    'max_area': 50000,  # Optimisé: maintenu pour taille maximale
    'min_circularity': 0.8,  # Optimisé: augmenté pour forme plus stricte
    'max_circularity': 1.5,  # Optimisé: augmenté pour plus de tolérance
    'color_tolerance': 50,  # Optimisé: équilibré pour segmentation couleur
    'nms_threshold': 0.2  # Optimisé: réduit pour éviter doublons agressivement
}

# Paramètres pour la classification
CLASS_PARAMS = {
    'roi_margin': 2,  # Marge autour des ROI (pixels)
    'digit_size': (32, 32),  # Taille normalisée pour OCR
    'hog_cells_per_block': (2, 2),  # HOG cells per block
    'hog_pixels_per_cell': (8, 8),  # HOG pixels per cell
    'hog_orientations': 9  # HOG orientations
}

# Chemins par défaut
DEFAULT_PATHS = {
    'data_dir': 'data/BD_METRO',
    'train_csv': 'data/train_split.csv',
    'test_csv': 'data/test_split.csv',
    'apprentissage_mat': 'docs/progsPython/Apprentissage.mat',
    'test_mat': 'docs/progsPython/Test.mat',
    'default_test_image': 'data/BD_METRO/IM (1).JPG'
}

DETECTOR_PARAMS = {
    'min_roi_size': 25,
    'max_roi_size': 250,
    'min_aspect_ratio': 0.4,
    'max_aspect_ratio': 2.5,
    'min_confidence': 0.35,
    'nms_iou_threshold': 0.5,
    'quality_threshold': 0.4,
    'max_detections_per_image': 8,
    'ideal_size': 80,
    'confidence_levels': {
        'high': 0.8,
        'medium': 0.6,
        'low': 0.4
    },
    'visualization_colors': {
        'high': (0, 255, 0),      # Vert
        'medium': (0, 165, 255),  # Orange
        'low': (0, 255, 255),     # Jaune  
        'poor': (0, 0, 255)       # Rouge
    },
    'model_path': 'models/advanced_digit_classifier.pkl'
}

GUI_PARAMS = {
    'window_size': "1600x1100",
    'canvas_size': (800, 600),
    'max_image_size': 750,
    'iou_threshold': 0.5,
    'model_save_path': "models/metro_detector_trained.pkl",
    'results_file': "results_test_TEAM1.mat",
    'colors': {
        'gt': (255, 0, 0),        # Vert pour ground truth
        'pred_tp': (0, 255, 0),   # Bleu pour vrais positifs  
        'pred_fp': (0, 0, 255),   # Rouge pour faux positifs
        'pred_wc': (0, 255, 255)  # Jaune pour wrong class
    },
    'fonts': {
        'title': ("Arial", 14, "bold"),
        'subtitle': ("Arial", 11, "bold"),
        'normal': ("Arial", 10),
        'status': ("Arial", 10),
        'code': ("Consolas", 9),
        'report': ("Consolas", 10)
    },
    'progress_length': 700,
    'metrics_text_size': (18, 45),
    'jump_size': 10,
    'button_colors': {
        'primary': '#0078D4',
        'secondary': '#6B7280', 
        'success': '#10B981',
        'warning': '#F59E0B',
        'danger': '#EF4444'
    },
    'supported_image_formats': [
        ("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"),
        ("JPEG", "*.jpg *.jpeg"),
        ("PNG", "*.png"),
        ("Tous les fichiers", "*.*")
    ]
}

# Configuration YOLO
YOLO_CONFIG = {
    'dataset_dir': 'yolotrain',
    'train_split': 0.8,  # 80% pour l'entraînement, 20% pour la validation
    'image_extensions': ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'],
    'model_name': 'models/yolov8n.pt',  # Modèle YOLO dans le dossier models/
    'epochs': 100,
    'batch_size': 16,
    'img_size': 640,
    'confidence_threshold': 0.5,
    'iou_threshold': 0.45,
    'class_names': ['panneau_metro'], 
    'augmentation': {
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.2
    }
} 