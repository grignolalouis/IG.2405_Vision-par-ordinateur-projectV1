#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de test pour vérifier le fonctionnement du système
de reconnaissance des lignes de métro.
"""

import sys
import os
import numpy as np
from PIL import Image

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.zones_interet import ZonesInteretDetector
from src.evaluation_zones import ZoneEvaluator
from src.detection_lignes import LigneDetector
from src.metro_recognition import MetroRecognitionSystem

def test_modules_individuels():
    """
    Test des modules individuels.
    """
    print("=== Test des modules individuels ===")
    
    # Image de test (créer une image synthétique)
    test_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    
    # Test module zones d'intérêt
    print("1. Test ZonesInteretDetector...")
    try:
        detector = ZonesInteretDetector(debug=False)
        zones = detector.extract_zones_interet(test_image)
        print(f"   ✓ Zones détectées: {len(zones)}")
    except Exception as e:
        print(f"   ✗ Erreur: {e}")
    
    # Test module évaluation
    print("2. Test ZoneEvaluator...")
    try:
        evaluator = ZoneEvaluator(debug=False)
        # Créer quelques zones de test
        zones_test = [(100, 100, 150, 150), (200, 200, 250, 250)]
        zones_eval = evaluator.evaluate_zones(test_image, zones_test)
        print(f"   ✓ Zones évaluées: {len(zones_eval)}")
    except Exception as e:
        print(f"   ✗ Erreur: {e}")
    
    # Test module détection lignes
    print("3. Test LigneDetector...")
    try:
        ligne_detector = LigneDetector(debug=False)
        # Créer des zones évaluées de test
        zones_eval_test = [
            {
                'zone': (100, 100, 150, 150),
                'roi': test_image[100:150, 100:150],
                'score': 0.8
            }
        ]
        detections = ligne_detector.detect_line_numbers(test_image, zones_eval_test)
        print(f"   ✓ Détections: {len(detections)}")
    except Exception as e:
        print(f"   ✗ Erreur: {e}")

def test_systeme_complet():
    """
    Test du système complet avec une image réelle si disponible.
    """
    print("\n=== Test du système complet ===")
    
    # Chercher une image de test
    test_image_path = None
    possible_paths = [
        'data/BD_METRO/IM (1).JPG',
        'data/BD_METRO/IM (2).JPG',
        'data/BD_METRO/IM (4).JPG'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            test_image_path = path
            break
    
    if test_image_path is None:
        print("   ⚠ Aucune image de test trouvée - test avec image synthétique")
        # Créer une image synthétique plus réaliste
        test_image = np.ones((600, 800, 3), dtype=np.uint8) * 128
        # Ajouter quelques cercles colorés
        import cv2
        cv2.circle(test_image, (200, 200), 30, (0, 255, 255), -1)  # Cercle jaune
        cv2.circle(test_image, (400, 300), 25, (255, 0, 0), -1)    # Cercle rouge
        
        # Sauvegarder temporairement
        test_image_path = 'temp_test_image.jpg'
        Image.fromarray(test_image).save(test_image_path)
    else:
        print(f"   📷 Utilisation de l'image: {test_image_path}")
    
    try:
        # Test du système complet
        system = MetroRecognitionSystem(debug=True)
        results = system.process_single_image(test_image_path, 1)
        
        print(f"   ✓ Traitement réussi!")
        print(f"   ✓ Détections: {len(results)}")
        
        if len(results) > 0:
            print("   Détections trouvées:")
            for i, detection in enumerate(results):
                img_num, x1, y1, x2, y2, ligne = detection
                print(f"     - Zone {i+1}: Ligne {int(ligne)} à ({x1},{y1})-({x2},{y2})")
        
        # Nettoyer le fichier temporaire si créé
        if test_image_path == 'temp_test_image.jpg':
            os.remove(test_image_path)
            
    except Exception as e:
        print(f"   ✗ Erreur: {e}")
        import traceback
        traceback.print_exc()

def test_dependances():
    """
    Test de la disponibilité des dépendances.
    """
    print("=== Test des dépendances ===")
    
    dependencies = [
        ('numpy', 'np'),
        ('cv2', 'cv2'),
        ('matplotlib.pyplot', 'plt'),
        ('PIL', 'PIL'),
        ('scipy.io', 'sio'),
        ('sklearn', 'sklearn')
    ]
    
    for dep_name, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"   ✓ {dep_name}")
        except ImportError:
            print(f"   ✗ {dep_name} - NON DISPONIBLE")
    
    # Test spécial pour Tesseract
    try:
        import pytesseract
        # Test simple
        pytesseract.get_tesseract_version()
        print(f"   ✓ pytesseract (Tesseract OCR disponible)")
    except:
        print(f"   ⚠ pytesseract - Tesseract OCR peut ne pas être installé")

def test_structure_fichiers():
    """
    Test de la structure des fichiers requis.
    """
    print("\n=== Test de la structure des fichiers ===")
    
    required_dirs = [
        'src/',
        'data/',
        'data/BD_METRO/',
        'docs/',
        'docs/progsPython/'
    ]
    
    required_files = [
        'src/__init__.py',
        'src/zones_interet.py',
        'src/evaluation_zones.py', 
        'src/detection_lignes.py',
        'src/metro_recognition.py',
        'main.py',
        'requirements.txt'
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"   ✓ {dir_path}")
        else:
            print(f"   ✗ {dir_path} - MANQUANT")
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✓ {file_path}")
        else:
            print(f"   ✗ {file_path} - MANQUANT")
    
    # Compter les images disponibles
    bd_metro_path = 'data/BD_METRO/'
    if os.path.exists(bd_metro_path):
        images = [f for f in os.listdir(bd_metro_path) if f.endswith('.JPG')]
        print(f"   📷 Images disponibles: {len(images)}/261")

def main():
    """
    Fonction principale de test.
    """
    print("="*60)
    print("TEST DU SYSTÈME DE RECONNAISSANCE MÉTRO PARISIEN")
    print("="*60)
    
    test_dependances()
    test_structure_fichiers()
    test_modules_individuels()
    test_systeme_complet()
    
    print("\n" + "="*60)
    print("TESTS TERMINÉS")
    print("="*60)

if __name__ == "__main__":
    main() 