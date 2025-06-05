# -*- coding: utf-8 -*-
"""
Script de préparation des données pour le projet de reconnaissance de signalisation du métro

Ce script convertit et prépare les données existantes au format requis par notre pipeline:
- Conversion des fichiers MATLAB existants
- Création des fichiers d'annotations
- Vérification de l'intégrité des données

@author: Projet IG2405 - Vision par ordinateur
"""

import sys
import os
import numpy as np
import scipy.io as sio
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import shutil

# Ajouter le dossier parent au path
sys.path.append('..')
sys.path.append('../src')

def check_data_availability():
    """Vérifie la disponibilité des données"""
    print("=== VÉRIFICATION DES DONNÉES ===")
    
    # Vérifier les dossiers
    data_dir = Path("../data/BD_METRO")
    docs_dir = Path("../docs")
    progs_dir = Path("../docs/progsPython")
    
    print(f"Dossier des images: {data_dir}")
    print(f"Existe: {data_dir.exists()}")
    
    if data_dir.exists():
        images = list(data_dir.glob("*.JPG"))
        print(f"Nombre d'images trouvées: {len(images)}")
        
        # Vérifier quelques images d'exemple
        for i in [1, 3, 6, 9]:
            img_path = data_dir / f"IM ({i}).JPG"
            print(f"  IM ({i}).JPG: {'✓' if img_path.exists() else '✗'}")
    
    print(f"\nDossier docs: {docs_dir}")
    print(f"Existe: {docs_dir.exists()}")
    
    print(f"\nDossier progsPython: {progs_dir}")
    print(f"Existe: {progs_dir.exists()}")
    
    if progs_dir.exists():
        # Vérifier les fichiers de programme
        files = list(progs_dir.glob("*"))
        print(f"Fichiers dans progsPython: {len(files)}")
        for file in files:
            print(f"  - {file.name}")
        
        # Vérifier les fichiers MAT dans progsPython
        mat_files = list(progs_dir.glob("*.mat"))
        if mat_files:
            print(f"\nFichiers MAT trouvés dans progsPython:")
            for mat_file in mat_files:
                print(f"  - {mat_file.name}")
                try:
                    data = sio.loadmat(str(mat_file))
                    print(f"    Clés: {list(data.keys())}")
                    if 'BD' in data:
                        BD = data['BD']
                        print(f"    Shape BD: {BD.shape}")
                        if BD.size > 0:
                            print(f"    Échantillon: {BD[0] if len(BD) > 0 else 'Vide'}")
                except Exception as e:
                    print(f"    Erreur lors de la lecture: {e}")

def copy_annotation_files():
    """Copie les fichiers d'annotations depuis progsPython vers docs"""
    print("\n=== COPIE DES FICHIERS D'ANNOTATIONS ===")
    
    progs_dir = Path("../docs/progsPython")
    docs_dir = Path("../docs")
    
    # Fichiers à copier
    files_to_copy = ["Apprentissage.mat", "Test.mat"]
    
    copied_files = []
    
    for filename in files_to_copy:
        source_path = progs_dir / filename
        dest_path = docs_dir / filename
        
        if source_path.exists():
            try:
                # Copier le fichier
                shutil.copy2(str(source_path), str(dest_path))
                print(f"✓ Copié: {filename}")
                copied_files.append(filename)
                
                # Vérifier le contenu
                data = sio.loadmat(str(dest_path))
                BD = data['BD']
                print(f"  - Shape: {BD.shape}")
                print(f"  - Nombre d'annotations: {len(BD)}")
                
                # Analyser les classes
                if len(BD) > 0:
                    unique_classes = np.unique(BD[:, 5])
                    print(f"  - Classes présentes: {unique_classes}")
                
            except Exception as e:
                print(f"✗ Erreur lors de la copie de {filename}: {e}")
        else:
            print(f"✗ Fichier source non trouvé: {source_path}")
    
    return copied_files

def verify_copied_files():
    """Vérifie que les fichiers copiés sont corrects"""
    print("\n=== VÉRIFICATION DES FICHIERS COPIÉS ===")
    
    docs_dir = Path("../docs")
    files_to_check = ["Apprentissage.mat", "Test.mat"]
    
    for filename in files_to_check:
        file_path = docs_dir / filename
        
        if file_path.exists():
            try:
                data = sio.loadmat(str(file_path))
                BD = data['BD']
                
                print(f"✓ {filename}:")
                print(f"  - Taille: {BD.shape}")
                print(f"  - Nombre d'annotations: {len(BD)}")
                
                if len(BD) > 0:
                    # Analyser les images
                    unique_images = np.unique(BD[:, 0])
                    print(f"  - Nombre d'images annotées: {len(unique_images)}")
                    print(f"  - Plage d'images: {int(np.min(unique_images))} à {int(np.max(unique_images))}")
                    
                    # Analyser les classes
                    unique_classes, counts = np.unique(BD[:, 5], return_counts=True)
                    print(f"  - Classes: {len(unique_classes)} différentes")
                    
                    # Vérifier le format des données
                    print(f"  - Format: [image_id, y1, y2, x1, x2, class_id]")
                    print(f"  - Exemple: {BD[0]}")
                
            except Exception as e:
                print(f"✗ Erreur lors de la vérification de {filename}: {e}")
        else:
            print(f"✗ {filename}: Non trouvé")

def create_synthetic_annotations():
    """Crée des annotations synthétiques pour tester le système"""
    print("\n=== CRÉATION D'ANNOTATIONS SYNTHÉTIQUES ===")
    
    # Paramètres
    total_images = 261
    num_classes = 14
    np.random.seed(42)  # Pour la reproductibilité
    
    # Splits d'images
    n = np.arange(1, total_images + 1)
    train_images = n[n % 3 == 0]  # Images multiples de 3
    test_images = n[n % 3 != 0]   # Images non multiples de 3
    
    print(f"Images d'entraînement: {len(train_images)}")
    print(f"Images de test: {len(test_images)}")
    
    def create_annotations_for_images(image_list, output_file):
        """Crée des annotations pour une liste d'images"""
        annotations = []
        
        for img_id in image_list:
            # Nombre aléatoire d'annotations par image (1 à 5)
            num_annotations = np.random.randint(1, 6)
            
            for _ in range(num_annotations):
                # Coordonnées aléatoires (format: image_id, y1, y2, x1, x2, class_id)
                # Taille d'image approximative: 640x480
                img_w, img_h = 640, 480
                
                # Taille de boîte aléatoire
                box_w = np.random.randint(30, 150)
                box_h = np.random.randint(30, 150)
                
                # Position aléatoire
                x1 = np.random.randint(0, max(1, img_w - box_w))
                y1 = np.random.randint(0, max(1, img_h - box_h))
                x2 = x1 + box_w
                y2 = y1 + box_h
                
                # Classe aléatoire
                class_id = np.random.randint(1, num_classes + 1)
                
                annotation = [img_id, y1, y2, x1, x2, class_id]
                annotations.append(annotation)
        
        # Convertir en array numpy
        BD = np.array(annotations)
        
        # Sauvegarder
        output_path = Path("../docs") / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        sio.savemat(str(output_path), {'BD': BD})
        print(f"Annotations sauvegardées: {output_path}")
        print(f"  - Nombre d'annotations: {len(annotations)}")
        
        return BD
    
    # Créer les annotations d'entraînement et de test
    train_annotations = create_annotations_for_images(train_images, "Apprentissage.mat")
    test_annotations = create_annotations_for_images(test_images, "Test.mat")
    
    # Analyser la distribution
    print("\n=== ANALYSE DE LA DISTRIBUTION ===")
    
    for name, annotations in [("Entraînement", train_annotations), ("Test", test_annotations)]:
        print(f"\n{name}:")
        unique_classes, counts = np.unique(annotations[:, 5], return_counts=True)
        total = len(annotations)
        
        for class_id, count in zip(unique_classes, counts):
            percentage = (count / total) * 100
            print(f"  Classe {int(class_id):2d}: {count:3d} instances ({percentage:5.1f}%)")

def convert_existing_data():
    """Tente de convertir les données existantes si disponibles"""
    print("\n=== CONVERSION DES DONNÉES EXISTANTES ===")
    
    docs_dir = Path("../docs")
    
    # Vérifier s'il y a des fichiers MATLAB existants
    mat_files = list(docs_dir.glob("*.mat"))
    
    if mat_files:
        print("Fichiers MAT trouvés:")
        for mat_file in mat_files:
            print(f"  - {mat_file.name}")
            try:
                data = sio.loadmat(str(mat_file))
                print(f"    Clés: {list(data.keys())}")
                
                if 'BD' in data:
                    BD = data['BD']
                    print(f"    Shape BD: {BD.shape}")
                    if BD.size > 0:
                        print(f"    Échantillon: {BD[0] if len(BD) > 0 else 'Vide'}")
                
            except Exception as e:
                print(f"    Erreur lors de la lecture: {e}")
    else:
        print("Aucun fichier MAT trouvé dans docs/")
    
    # Vérifier les fichiers dans progsPython
    prog_dir = docs_dir / "progsPython"
    if prog_dir.exists():
        # Rechercher des fichiers de données
        data_files = list(prog_dir.glob("*.mat")) + list(prog_dir.glob("*.xls*"))
        
        if data_files:
            print(f"\nFichiers de données dans progsPython:")
            for data_file in data_files:
                print(f"  - {data_file.name}")
                
                if data_file.suffix == '.mat':
                    try:
                        data = sio.loadmat(str(data_file))
                        print(f"    Clés MAT: {list(data.keys())}")
                    except Exception as e:
                        print(f"    Erreur MAT: {e}")
                
                elif data_file.suffix in ['.xls', '.xlsx']:
                    try:
                        # Tenter de lire avec pandas
                        df = pd.read_excel(str(data_file))
                        print(f"    Shape Excel: {df.shape}")
                        print(f"    Colonnes: {list(df.columns)[:5]}...")  # Premières colonnes
                    except Exception as e:
                        print(f"    Erreur Excel: {e}")

def create_test_structure():
    """Crée une structure de test pour vérifier le pipeline"""
    print("\n=== CRÉATION DE LA STRUCTURE DE TEST ===")
    
    # Créer les dossiers nécessaires
    directories = [
        "../models",
        "../results",
        "../results/visualizations", 
        "../results/evaluations",
        "../logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Dossier créé: {directory}")
    
    # Créer un fichier de configuration
    config = {
        "project_name": "Metro Sign Recognition",
        "data_path": "data/BD_METRO",
        "docs_path": "docs",
        "total_images": 261,
        "num_classes": 14,
        "class_names": {
            1: "Interdiction",
            2: "Obligation", 
            3: "Danger",
            4: "Direction",
            5: "Information",
            6: "Sortie",
            7: "Escalator",
            8: "Ascenseur",
            9: "Toilettes",
            10: "Téléphone",
            11: "Restaurant",
            12: "Boutique",
            13: "Parking",
            14: "Autre"
        }
    }
    
    import json
    config_path = Path("../config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"Configuration sauvegardée: {config_path}")

def visualize_data_distribution():
    """Visualise la distribution des données créées"""
    print("\n=== VISUALISATION DE LA DISTRIBUTION ===")
    
    docs_dir = Path("../docs")
    
    # Charger les annotations si elles existent
    files_to_check = ["Apprentissage.mat", "Test.mat"]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, filename in enumerate(files_to_check):
        file_path = docs_dir / filename
        
        if file_path.exists():
            try:
                data = sio.loadmat(str(file_path))
                BD = data['BD']
                
                # Distribution des classes
                unique_classes, counts = np.unique(BD[:, 5], return_counts=True)
                
                # Graphique
                axes[i].bar(unique_classes, counts)
                axes[i].set_title(f"Distribution des classes - {filename.replace('.mat', '')}")
                axes[i].set_xlabel("Classe")
                axes[i].set_ylabel("Nombre d'annotations")
                axes[i].grid(True, alpha=0.3)
                
                # Ajouter les valeurs sur les barres
                for class_id, count in zip(unique_classes, counts):
                    axes[i].text(class_id, count + 0.5, str(count), ha='center')
                
                print(f"Graphique créé pour {filename}")
                
            except Exception as e:
                print(f"Erreur lors de la lecture de {filename}: {e}")
                axes[i].text(0.5, 0.5, f"Erreur: {filename}", ha='center', va='center', 
                           transform=axes[i].transAxes)
        else:
            axes[i].text(0.5, 0.5, f"Fichier non trouvé:\n{filename}", ha='center', va='center',
                        transform=axes[i].transAxes)
    
    plt.tight_layout()
    
    # Sauvegarder le graphique
    output_path = Path("../results/data_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé: {output_path}")
    
    plt.show()

def main():
    """Fonction principale"""
    print("PRÉPARATION DES DONNÉES - PROJET METRO SIGNALISATION")
    print("="*60)
    
    # Étape 1: Vérifier les données disponibles
    check_data_availability()
    
    # Étape 2: Copier les fichiers d'annotations depuis progsPython
    copied_files = copy_annotation_files()
    
    # Étape 3: Vérifier les fichiers copiés
    if copied_files:
        verify_copied_files()
    
    # Étape 4: Vérifier si tous les fichiers nécessaires sont présents
    docs_dir = Path("../docs")
    apprentissage_exists = (docs_dir / "Apprentissage.mat").exists()
    test_exists = (docs_dir / "Test.mat").exists()
    
    print(f"\n=== ÉTAT DES FICHIERS D'ANNOTATIONS ===")
    print(f"  - Apprentissage.mat: {'✓' if apprentissage_exists else '✗'}")
    print(f"  - Test.mat: {'✓' if test_exists else '✗'}")
    
    if not apprentissage_exists or not test_exists:
        print(f"\nCertains fichiers d'annotations sont encore manquants.")
        response = input("Créer des annotations synthétiques pour les fichiers manquants? (o/n): ")
        if response.lower() in ['o', 'oui', 'y', 'yes']:
            create_synthetic_annotations()
        else:
            print("Annotations synthétiques non créées")
    else:
        print("\n✓ Tous les fichiers d'annotations sont présents")
    
    # Étape 5: Créer la structure de test
    create_test_structure()
    
    # Étape 6: Visualiser la distribution
    try:
        visualize_data_distribution()
    except Exception as e:
        print(f"Erreur lors de la visualisation: {e}")
    
    print("\n" + "="*60)
    print("PRÉPARATION TERMINÉE")
    print("="*60)
    print("\nVous pouvez maintenant exécuter:")
    print("  python main.py --mode full")
    print("  python main.py --mode gui")

if __name__ == "__main__":
    main() 