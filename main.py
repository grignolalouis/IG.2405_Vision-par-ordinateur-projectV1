import sys
import os
import tkinter as tk
from tkinter import messagebox

# Ajouter le répertoire courant au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from gui_visualizer import MetroProjectMainGUI
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("Vérifiez que tous les modules sont présents dans le dossier src/")
    sys.exit(1)


def check_project_structure():
    """Vérifie que la structure du projet est correcte"""
    required_dirs = [
        "data/BD_METRO",
        "src",
        "docs",
        "models"
    ]
    
    required_files = [
        "src/__init__.py",
        "src/data_loader.py",
        "src/detector.py",
        "src/preprocessing.py",
        "src/segmentation.py",
        "src/classification.py",
        "src/constants.py"
    ]
    
    missing_items = []
    
    # Vérifier les dossiers
    for directory in required_dirs:
        if not os.path.exists(directory):
            missing_items.append(f"Dossier manquant: {directory}")
    
    # Vérifier les fichiers
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_items.append(f"Fichier manquant: {file_path}")
    
    # Créer le dossier models s'il n'existe pas
    if not os.path.exists("models"):
        os.makedirs("models")
        print("📁 Dossier 'models' créé")
    
    if missing_items:
        print("❌ Structure de projet incomplète:")
        for item in missing_items:
            print(f"  - {item}")
        return False
    
    return True


def check_image_directory():
    """Vérifie que le dossier d'images contient des fichiers"""
    image_dir = "data/BD_METRO"
    
    if not os.path.exists(image_dir):
        print(f"❌ Dossier d'images non trouvé: {image_dir}")
        return False
    
    image_files = []
    for file in os.listdir(image_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(file)
    
    if len(image_files) == 0:
        print(f"❌ Aucune image trouvée dans {image_dir}")
        return False
    
    print(f"✅ {len(image_files)} images trouvées dans {image_dir}")
    return True


def show_project_info():
    """Affiche les informations du projet"""
    print("=" * 80)
    print("🚇 PROJET MÉTRO PARISIEN - DÉTECTION AUTOMATIQUE (TEAM1)")
    print("=" * 80)
    print("📋 OBJECTIF:")
    print("   Détection automatique des panneaux de métro parisien")
    print("   Lignes supportées: 1-14 (excluant 3bis et 7bis)")
    print()
    print("⚙️  PIPELINE AUTOMATIQUE:")
    print("   1. 🔄 Chargement des données avec split train/test automatique")
    print("      • Train: Images avec ID multiple de 3 (3, 6, 9, ...)")
    print("      • Test:  Images avec ID non multiple de 3 (1, 2, 4, 5, 7, 8, ...)")
    print("   2. 🧠 Entraînement du modèle sur les données d'apprentissage")
    print("   3. 💾 Sauvegarde du modèle entraîné")
    print("   4. 🔍 Prédiction sur les données de test")
    print("   5. 📊 Calcul des métriques de performance")
    print("   6. 🖼️  Visualisation interactive des résultats")
    print("   7. 📤 Export des résultats au format MAT")
    print()
    print("🎯 SPÉCIFICATIONS TECHNIQUES:")
    print("   • Images traitées à taille originale (pas de redimensionnement)")
    print("   • Coordonnées exactes préservées")
    print("   • Interface graphique complète avec métriques détaillées")
    print("   • Export compatible avec les exigences du projet")
    print("=" * 80)
    print()


def main():
    """Fonction principale du projet"""
    
    # Afficher les informations du projet
    show_project_info()
    
    # Vérifier la structure du projet
    print("🔍 Vérification de la structure du projet...")
    if not check_project_structure():
        print("❌ Veuillez corriger la structure du projet avant de continuer.")
        input("Appuyez sur Entrée pour quitter...")
        return 1
    
    print("✅ Structure du projet validée")
    
    # Vérifier les images
    print("🔍 Vérification des images...")
    if not check_image_directory():
        print("⚠️  Attention: Aucune image trouvée.")
        print("   Placez vos images dans le dossier 'data/BD_METRO/'")
        if not messagebox.askyesno("Continuer", 
                                   "Aucune image trouvée. Voulez-vous quand même continuer?\n"
                                   "(Le système fonctionnera en mode démo)"):
            return 1
    
    # Lancer l'interface graphique
    print("🚀 Lancement de l'interface graphique...")
    print("📱 Le pipeline automatique va démarrer dans quelques secondes...")
    print()
    
    try:
        # Créer la fenêtre principale
        root = tk.Tk()
        
        # Configuration de la fenêtre
        root.title("🚇 Projet Métro Parisien - TEAM1")
        
        # Centrer la fenêtre
        root.eval('tk::PlaceWindow . center')
        
        # Créer l'application
        app = MetroProjectMainGUI(root)
        
        # Message de bienvenue
        print("✅ Interface graphique initialisée")
        print("🎯 Le pipeline automatique va commencer...")
        print("💡 Conseil: Consultez l'onglet 'État du Pipeline' pour suivre la progression")
        print()
        
        # Lancer la boucle principale
        root.mainloop()
        
        print("👋 Merci d'avoir utilisé le système de détection métro TEAM1!")
        return 0
        
    except Exception as e:
        error_msg = f"❌ Erreur lors du lancement: {str(e)}"
        print(error_msg)
        
        # Afficher l'erreur dans une boîte de dialogue si possible
        try:
            messagebox.showerror("Erreur", error_msg)
        except:
            pass
        
        return 1


if __name__ == "__main__":
    """Point d'entrée du script"""
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Arrêt demandé par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Erreur critique: {e}")
        sys.exit(1) 