"""
Script de test pour vérifier le système de détection
"""

import os
import cv2
import numpy as np
from src.detector import MetroSignDetector
from src.data_loader import DataLoader
from src.constants import METRO_COLORS
import matplotlib.pyplot as plt


def test_single_image():
    """Test sur une seule image"""
    print("=== Test sur une seule image ===")
    
    # Initialiser le détecteur
    detector = MetroSignDetector()
    
    # Chercher une image de test
    data_dir = "data/BD_METRO"
    if os.path.exists(data_dir):
        images = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
        if images:
            test_image = os.path.join(data_dir, images[0])
            print(f"Test sur: {test_image}")
            
            # Détecter
            result = detector.detect_signs(test_image)
            
            print(f"Nombre de détections: {len(result['detections'])}")
            for i, det in enumerate(result['detections']):
                print(f"  Détection {i+1}: Ligne {det['line_num']} (confiance: {det['confidence']:.2f})")
                
            # Visualiser
            vis_image = detector.visualize_detections(
                test_image, 
                result['detections'], 
                result['scale_factor']
            )
            
            # Afficher avec matplotlib
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            plt.title("Détections de panneaux de métro")
            plt.axis('off')
            plt.show()
            
            return True
    
    print("Aucune image trouvée pour le test")
    return False


def test_color_segmentation():
    """Test de la segmentation par couleur"""
    print("\n=== Test de la segmentation par couleur ===")
    
    from src.segmentation import MetroSegmenter
    from src.preprocessing import ImagePreprocessor
    
    # Créer une image de test avec des cercles de couleurs de métro
    test_image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Ajouter des cercles avec les couleurs des lignes
    positions = [
        (100, 100), (200, 100), (300, 100), (400, 100), (500, 100),
        (100, 200), (200, 200), (300, 200), (400, 200), (500, 200),
        (100, 300), (200, 300), (300, 300), (400, 300)
    ]
    
    for i, (line_num, color_info) in enumerate(METRO_COLORS.items()):
        if i < len(positions):
            x, y = positions[i]
            color = color_info['rgb'][::-1]  # RGB vers BGR
            cv2.circle(test_image, (x, y), 40, color, -1)
            cv2.circle(test_image, (x, y), 40, (0, 0, 0), 2)
            
            # Ajouter le numéro
            cv2.putText(
                test_image, str(line_num), 
                (x - 15, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (255, 255, 255), 2
            )
    
    # Prétraiter
    preprocessor = ImagePreprocessor()
    prep_result = preprocessor.preprocess(test_image)
    
    # Segmenter
    segmenter = MetroSegmenter()
    rois = segmenter.segment(prep_result['processed'])
    
    print(f"Nombre de ROIs détectées: {len(rois)}")
    
    # Visualiser
    vis_image = test_image.copy()
    for roi in rois:
        xmin, ymin, xmax, ymax = roi['bbox']
        line_num = roi['line_num_color']
        cv2.rectangle(vis_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(
            vis_image, f"L{line_num}",
            (xmin, ymin - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 2
        )
    
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    plt.title("Test de segmentation par couleur")
    plt.axis('off')
    plt.show()


def generate_sample_csv():
    """Génère des fichiers CSV d'exemple si nécessaire"""
    import pandas as pd
    
    # Supprimer les anciens fichiers s'ils existent
    for csv_file in ["data/train_split.csv", "data/test_split.csv"]:
        if os.path.exists(csv_file):
            os.remove(csv_file)
    
    print("\n=== Génération des fichiers CSV d'exemple ===")
    
    # Créer des données d'exemple
    train_data = []
    test_data = []
    
    # Simuler des annotations pour quelques images
    for i in range(1, 262):  # 261 images
        # Nombre aléatoire de panneaux par image (0 à 3)
        num_signs = np.random.randint(0, 4)
        
        for _ in range(num_signs):
            # Générer une boîte aléatoire réaliste
            xmin = np.random.randint(50, 500)
            ymin = np.random.randint(50, 400)
            width = np.random.randint(30, 80)  # Taille réaliste d'un panneau
            height = np.random.randint(30, 80)
            
            annotation = {
                'image_id': i,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmin + width,
                'ymax': ymin + height,
                'line': np.random.choice(list(METRO_COLORS.keys()))
            }
            
            # Répartir selon la règle 1/3 apprentissage, 2/3 test
            if i % 3 == 0:
                train_data.append(annotation)
            else:
                test_data.append(annotation)
    
    # Créer les DataFrames avec les bonnes colonnes
    if train_data:
        train_df = pd.DataFrame(train_data)
    else:
        # DataFrame vide avec les bonnes colonnes
        train_df = pd.DataFrame(columns=['image_id', 'xmin', 'ymin', 'xmax', 'ymax', 'line'])
    
    if test_data:
        test_df = pd.DataFrame(test_data)
    else:
        # DataFrame vide avec les bonnes colonnes
        test_df = pd.DataFrame(columns=['image_id', 'xmin', 'ymin', 'xmax', 'ymax', 'line'])
    
    # Sauvegarder
    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train_split.csv", index=False)
    test_df.to_csv("data/test_split.csv", index=False)
    
    print(f"Fichier train_split.csv créé avec {len(train_data)} annotations")
    print(f"Fichier test_split.csv créé avec {len(test_data)} annotations")
    
    # Afficher un échantillon pour vérification
    if len(train_df) > 0:
        print(f"\nÉchantillon train_split.csv:")
        print(train_df.head())
    if len(test_df) > 0:
        print(f"\nÉchantillon test_split.csv:")
        print(test_df.head())


def main():
    """Fonction principale de test"""
    print("Test du système de détection de panneaux de métro")
    print("=" * 50)
    
    # Générer les CSV si nécessaire
    generate_sample_csv()
    
    # Test de segmentation sur image synthétique
    test_color_segmentation()
    
    # Test sur image réelle si disponible
    if os.path.exists("data/BD_METRO"):
        test_single_image()
    else:
        print("\nDossier data/BD_METRO non trouvé. Placez vos images dans ce dossier.")
        print("Structure attendue:")
        print("  data/")
        print("    BD_METRO/")
        print("      metro001.jpg")
        print("      metro002.jpg")
        print("      ...")


if __name__ == "__main__":
    main() 