#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de test pour vérifier les couleurs des lignes de métro

Ce script teste la conversion des couleurs officielles RATP 
en ranges HSV et valide les mappings de couleurs.
"""

import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import colorsys

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def hex_to_rgb(hex_color):
    """Convertit une couleur hexadécimale en RGB."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hsv(rgb):
    """Convertit RGB (0-255) en HSV (H:0-179, S:0-255, V:0-255) pour OpenCV."""
    r, g, b = [x/255.0 for x in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return int(h * 179), int(s * 255), int(v * 255)

def test_color_conversions():
    """Teste la conversion des couleurs officielles."""
    
    # Couleurs officielles RATP
    official_colors = {
        1: "#FFCE00",   # Jaune
        2: "#0064B0",   # Bleu
        3: "#9F9825",   # Olive/vert-jaune
        4: "#C04191",   # Violet/magenta
        5: "#F28E42",   # Orange
        6: "#83C491",   # Vert clair
        7: "#F3A4BA",   # Rose
        8: "#CEADD2",   # Violet clair
        9: "#D5C900",   # Jaune-vert
        10: "#E3B32A",  # Orange/marron
        11: "#8D5E2A",  # Marron
        12: "#00814F",  # Vert foncé
        13: "#98D4E2",  # Bleu clair
        14: "#662483"   # Violet foncé
    }
    
    print("=== CONVERSION COULEURS OFFICIELLES RATP ===")
    print(f"{'Ligne':<6} {'Hex':<8} {'RGB':<15} {'HSV (OpenCV)':<15}")
    print("-" * 50)
    
    hsv_ranges = {}
    
    for ligne, hex_color in official_colors.items():
        rgb = hex_to_rgb(hex_color)
        hsv = rgb_to_hsv(rgb)
        
        print(f"{ligne:<6} {hex_color:<8} {str(rgb):<15} {str(hsv)}")
        
        # Calculer une range HSV approximative (±10 pour H, ±50 pour S, ±50 pour V)
        h, s, v = hsv
        h_low = max(0, h - 10)
        h_high = min(179, h + 10)
        s_low = max(0, s - 50)
        s_high = min(255, s + 50)
        v_low = max(0, v - 50)
        v_high = min(255, v + 50)
        
        hsv_ranges[ligne] = [(h_low, s_low, v_low), (h_high, s_high, v_high)]
    
    return hsv_ranges

def create_color_palette():
    """Crée une palette visuelle des couleurs de lignes."""
    
    official_colors = {
        1: "#FFCE00", 2: "#0064B0", 3: "#9F9825", 4: "#C04191",
        5: "#F28E42", 6: "#83C491", 7: "#F3A4BA", 8: "#CEADD2",
        9: "#D5C900", 10: "#E3B32A", 11: "#8D5E2A", 12: "#00814F",
        13: "#98D4E2", 14: "#662483"
    }
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    
    # Créer la palette
    y_pos = 0
    for ligne, hex_color in official_colors.items():
        rgb = [x/255.0 for x in hex_to_rgb(hex_color)]
        
        # Rectangle de couleur
        rect = Rectangle((0, y_pos), 2, 0.8, 
                        facecolor=rgb, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        
        # Texte avec numéro de ligne
        ax.text(1, y_pos + 0.4, f"Ligne {ligne}", 
               ha='center', va='center', fontsize=12, fontweight='bold',
               color='white' if sum(rgb) < 1.5 else 'black')
        
        # Code couleur
        ax.text(2.5, y_pos + 0.4, hex_color, 
               ha='left', va='center', fontsize=10, 
               fontfamily='monospace')
        
        y_pos += 1
    
    ax.set_xlim(-0.5, 4)
    ax.set_ylim(-0.5, len(official_colors))
    ax.set_title('Palette officielle des couleurs RATP - Métro Parisien', 
                fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def test_color_detection():
    """Teste la détection des couleurs sur des échantillons synthétiques."""
    
    from src.detection_lignes import LigneDetector
    
    # Créer des échantillons de couleur
    detector = LigneDetector(debug=True)
    
    official_colors = {
        1: "#FFCE00", 2: "#0064B0", 3: "#9F9825", 4: "#C04191",
        5: "#F28E42", 6: "#83C491", 7: "#F3A4BA", 8: "#CEADD2",
        9: "#D5C900", 10: "#E3B32A", 11: "#8D5E2A", 12: "#00814F",
        13: "#98D4E2", 14: "#662483"
    }
    
    print("\n=== TEST DE DÉTECTION DES COULEURS ===")
    
    for ligne, hex_color in official_colors.items():
        # Créer un carré de couleur uniforme
        rgb = hex_to_rgb(hex_color)
        test_image = np.ones((100, 100, 3), dtype=np.uint8)
        test_image[:, :] = rgb
        
        # Tester la classification
        color_pred = detector._classify_by_color(test_image)
        
        predicted_line = color_pred.get('predicted_line')
        confidence = color_pred.get('confidence', 0)
        
        status = "✓" if predicted_line == ligne else "✗"
        print(f"Ligne {ligne:2d} ({hex_color}) -> Prédiction: {predicted_line:2d} "
              f"(confiance: {confidence:.3f}) {status}")

def suggest_hsv_ranges():
    """Suggère des ranges HSV optimisées basées sur les couleurs officielles."""
    
    print("\n=== SUGGESTIONS DE RANGES HSV OPTIMISÉES ===")
    
    official_colors = {
        1: "#FFCE00", 2: "#0064B0", 3: "#9F9825", 4: "#C04191",
        5: "#F28E42", 6: "#83C491", 7: "#F3A4BA", 8: "#CEADD2",
        9: "#D5C900", 10: "#E3B32A", 11: "#8D5E2A", 12: "#00814F",
        13: "#98D4E2", 14: "#662483"
    }
    
    print("# Ranges HSV suggérées pour detection_lignes.py :")
    print("self.line_color_ranges = {")
    
    for ligne, hex_color in official_colors.items():
        rgb = hex_to_rgb(hex_color)
        h, s, v = rgb_to_hsv(rgb)
        
        # Ajustement des ranges selon la couleur
        if h < 20 or h > 160:  # Rouge/magenta/violet
            h_range = 15
        elif 20 <= h <= 40:    # Jaune/orange
            h_range = 10
        else:                  # Vert/bleu
            h_range = 15
            
        # Range adaptative pour saturation et valeur
        s_range = min(50, max(30, s // 3))
        v_range = min(60, max(40, v // 4))
        
        h_low = max(0, h - h_range)
        h_high = min(179, h + h_range)
        s_low = max(0, s - s_range)
        s_high = min(255, s + s_range)
        v_low = max(0, v - v_range)
        v_high = min(255, v + v_range)
        
        print(f"    {ligne}: [({h_low}, {s_low}, {v_low}), ({h_high}, {s_high}, {v_high})],  # {hex_color}")
    
    print("}")

def main():
    """Fonction principale."""
    
    print("="*60)
    print("TEST DES COULEURS LIGNES METRO PARISIEN")
    print("="*60)
    
    # 1. Conversion des couleurs
    hsv_ranges = test_color_conversions()
    
    # 2. Palette visuelle
    create_color_palette()
    
    # 3. Test de détection
    try:
        test_color_detection()
    except ImportError as e:
        print(f"Impossible de tester la détection: {e}")
    
    # 4. Suggestions d'optimisation
    suggest_hsv_ranges()
    
    print("\n" + "="*60)
    print("TESTS TERMINÉS")
    print("="*60)

if __name__ == "__main__":
    main() 