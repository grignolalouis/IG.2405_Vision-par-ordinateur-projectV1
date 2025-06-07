import cv2
import numpy as np
import argparse
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.constants import DEFAULT_PATHS


def analyze_color_channels(image):
    """Analyse les canaux de couleur pour détecter l'encodage"""
    if len(image.shape) != 3 or image.shape[2] != 3:
        return "GRAYSCALE ou format non supporté"
    
    # Calculer les moyennes de chaque canal
    channel_means = np.mean(image, axis=(0, 1))
    
    # Calculer les écart-types pour voir la distribution
    channel_stds = np.std(image, axis=(0, 1))
    
    return {
        'means': channel_means,
        'stds': channel_stds,
        'dominant_channel': np.argmax(channel_means)
    }


def detect_color_space(image_path):
    """Détecte l'espace colorimétrique d'une image"""
    if not os.path.exists(image_path):
        return f"Erreur: Fichier {image_path} non trouvé"
    
    results = {}
    
    # Test 1: Chargement avec OpenCV (BGR)
    try:
        img_cv2 = cv2.imread(image_path)
        if img_cv2 is not None:
            results['opencv_bgr'] = {
                'loaded': True,
                'shape': img_cv2.shape,
                'analysis': analyze_color_channels(img_cv2)
            }
        else:
            results['opencv_bgr'] = {'loaded': False}
    except Exception as e:
        results['opencv_bgr'] = {'loaded': False, 'error': str(e)}
    
    # Test 2: Chargement avec PIL (RGB)
    try:
        img_pil = Image.open(image_path)
        img_pil_array = np.array(img_pil)
        results['pil_rgb'] = {
            'loaded': True,
            'mode': img_pil.mode,
            'shape': img_pil_array.shape,
            'analysis': analyze_color_channels(img_pil_array) if len(img_pil_array.shape) == 3 else None
        }
    except Exception as e:
        results['pil_rgb'] = {'loaded': False, 'error': str(e)}
    
    # Test 3: Chargement avec Matplotlib (RGB)
    try:
        img_plt = plt.imread(image_path)
        results['matplotlib_rgb'] = {
            'loaded': True,
            'shape': img_plt.shape,
            'dtype': str(img_plt.dtype),
            'analysis': analyze_color_channels(img_plt) if len(img_plt.shape) == 3 else None
        }
    except Exception as e:
        results['matplotlib_rgb'] = {'loaded': False, 'error': str(e)}
    
    return results


def compare_conversions(image_path):
    """Compare les conversions Lab avec différents flags"""
    img_cv2 = cv2.imread(image_path)
    if img_cv2 is None:
        return "Impossible de charger l'image"
    
    comparisons = {}
    
    # BGR → Lab
    try:
        lab_bgr = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2Lab)
        comparisons['bgr_to_lab'] = {
            'success': True,
            'lab_means': np.mean(lab_bgr, axis=(0, 1))
        }
    except Exception as e:
        comparisons['bgr_to_lab'] = {'success': False, 'error': str(e)}
    
    # RGB → Lab (avec l'image BGR, pour voir la différence)
    try:
        lab_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2Lab)
        comparisons['rgb_to_lab'] = {
            'success': True,
            'lab_means': np.mean(lab_rgb, axis=(0, 1))
        }
    except Exception as e:
        comparisons['rgb_to_lab'] = {'success': False, 'error': str(e)}
    
    return comparisons


def print_results(image_path, results, comparisons):
    """Affiche les résultats de manière lisible"""
    print(f"\n{'='*60}")
    print(f"ANALYSE DE L'ESPACE COLORIMÉTRIQUE")
    print(f"Image: {image_path}")
    print(f"{'='*60}")
    
    # Résultats de chargement
    for loader, data in results.items():
        print(f"\n{loader.upper().replace('_', ' ')}:")
        if data['loaded']:
            print(f"  ✅ Chargé avec succès")
            print(f"  📐 Shape: {data['shape']}")
            if 'mode' in data:
                print(f"  🎨 Mode: {data['mode']}")
            if 'dtype' in data:
                print(f"  📊 Type: {data['dtype']}")
            
            if data.get('analysis'):
                analysis = data['analysis']
                print(f"  📊 Moyennes des canaux: {analysis['means']}")
                print(f"  📈 Écart-types: {analysis['stds']}")
                print(f"  🎯 Canal dominant: {analysis['dominant_channel']} ({'BGR'[analysis['dominant_channel']] if loader == 'opencv_bgr' else 'RGB'[analysis['dominant_channel']]})")
        else:
            print(f"  ❌ Échec de chargement")
            if 'error' in data:
                print(f"  🚫 Erreur: {data['error']}")
    
    # Comparaison des conversions Lab
    print(f"\n{'='*40}")
    print("COMPARAISON CONVERSIONS LAB:")
    print(f"{'='*40}")
    
    for conversion, data in comparisons.items():
        conv_name = conversion.replace('_', ' → ').upper()
        print(f"\n{conv_name}:")
        if data['success']:
            print(f"  ✅ Conversion réussie")
            print(f"  🎨 Moyennes Lab: L={data['lab_means'][0]:.1f}, a={data['lab_means'][1]:.1f}, b={data['lab_means'][2]:.1f}")
        else:
            print(f"  ❌ Échec de conversion")
    
    # Recommandation
    print(f"\n{'='*40}")
    print("RECOMMANDATION:")
    print(f"{'='*40}")
    
    if results['opencv_bgr']['loaded']:
        print("🎯 Pour OpenCV: Utilisez cv2.COLOR_BGR2Lab")
        print("🎯 L'image est chargée en BGR par défaut")
    
    if results['pil_rgb']['loaded']:
        print("🎯 Pour PIL/Pillow: Utilisez cv2.COLOR_RGB2Lab")
        print("🎯 L'image est chargée en RGB par défaut")


def main():
    parser = argparse.ArgumentParser(description="Détecte l'espace colorimétrique d'une image")
    parser.add_argument(
        "image_path", 
        nargs='?',
        default=DEFAULT_PATHS['default_test_image'],
        help=f"Chemin vers l'image à analyser (défaut: {DEFAULT_PATHS['default_test_image']})"
    )
    
    args = parser.parse_args()
    
    # Vérifier si l'image par défaut existe
    if args.image_path == DEFAULT_PATHS['default_test_image'] and not os.path.exists(args.image_path):
        print(f"⚠️  Image par défaut non trouvée: {args.image_path}")
        print("💡 Spécifiez un chemin d'image ou placez une image test dans le dossier data/BD_METRO/")
        return
    
    results = detect_color_space(args.image_path)
    comparisons = compare_conversions(args.image_path)
    
    print_results(args.image_path, results, comparisons)


if __name__ == "__main__":
    main() 