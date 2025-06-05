"""
Chargeur de données spécialisé pour le projet métro parisien
Split automatique: Train = IDs multiples de 3, Test = reste
"""

import os
import scipy.io as sio
import numpy as np
from typing import List, Tuple, Dict, Any


class DataLoader:
    def __init__(self, data_dir: str = "data"):
        """
        Initialise le chargeur de données avec split automatique
        
        Args:
            data_dir: Répertoire racine des données
        """
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "BD_METRO")
        self.docs_dir = "docs/progsPython"  # Chercher dans progsPython uniquement
        
        # Fichiers d'annotations (pour compatibilité si ils existent)
        self.train_mat = os.path.join(self.docs_dir, "Apprentissage.mat")
        self.test_mat = os.path.join(self.docs_dir, "Test.mat")
        
        # Cache pour les annotations
        self._all_annotations = None
        
    def _load_all_annotations(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        Charge toutes les annotations disponibles (depuis MAT ou autres sources)
        
        Returns:
            Dictionnaire {image_id: [annotations]}
        """
        if self._all_annotations is not None:
            return self._all_annotations
            
        annotations = {}
        
        # Essayer de charger depuis les fichiers MAT existants
        for mat_file in [self.train_mat, self.test_mat]:
            if os.path.exists(mat_file):
                try:
                    mat_data = sio.loadmat(mat_file)
                    # Trouver la variable qui contient les données
                    data_key = None
                    for key in mat_data.keys():
                        if not key.startswith('__'):
                            data_key = key
                            break
                    
                    if data_key:
                        bd_array = mat_data[data_key]
                        for row in bd_array:
                            if len(row) >= 6:
                                image_id = int(row[0])
                                annotation = {
                                    'xmin': int(row[3]),  # x1 est en position 3
                                    'ymin': int(row[1]),  # y1 est en position 1
                                    'xmax': int(row[4]),  # x2 est en position 4
                                    'ymax': int(row[2]),  # y2 est en position 2
                                    'line': int(row[5])
                                }
                                
                                if image_id not in annotations:
                                    annotations[image_id] = []
                                annotations[image_id].append(annotation)
                                
                except Exception as e:
                    print(f"⚠️  Erreur lecture {mat_file}: {e}")
        
        # Si aucune annotation trouvée, créer des annotations vides pour toutes les images
        if not annotations:
            all_images = self.get_all_images()
            for image_path in all_images:
                image_id = self._extract_image_id(os.path.basename(image_path))
                if image_id:
                    annotations[image_id] = []
        
        self._all_annotations = annotations
        return annotations
        
    def load_training_data(self) -> List[Tuple[str, List[Dict[str, Any]]]]:
        """
        Charge les données d'entraînement (images avec ID multiple de 3)
        
        Returns:
            Liste de tuples (chemin_image, annotations)
        """
        print("🔄 Chargement des données d'apprentissage (IDs multiples de 3)...")
        
        all_annotations = self._load_all_annotations()
        train_data = []
        
        for image_id, annotations in all_annotations.items():
            # Règle: entraînement = multiples de 3
            if image_id % 3 == 0:
                image_path = self._get_image_path(image_id)
                if os.path.exists(image_path):
                    train_data.append((image_path, annotations))
                else:
                    print(f"⚠️  Image non trouvée: {image_path}")
        
        train_data.sort(key=lambda x: self._extract_image_id(os.path.basename(x[0])) or 0)
        print(f"✅ Données d'apprentissage: {len(train_data)} images chargées")
        return train_data
    
    def load_test_data(self) -> List[Tuple[str, List[Dict[str, Any]]]]:
        """
        Charge les données de test (images avec ID NON multiple de 3)
        
        Returns:
            Liste de tuples (chemin_image, annotations)
        """
        print("🔄 Chargement des données de test (IDs non multiples de 3)...")
        
        all_annotations = self._load_all_annotations()
        test_data = []
        
        for image_id, annotations in all_annotations.items():
            # Règle: test = non multiples de 3
            if image_id % 3 != 0:
                image_path = self._get_image_path(image_id)
                if os.path.exists(image_path):
                    test_data.append((image_path, annotations))
                else:
                    print(f"⚠️  Image non trouvée: {image_path}")
        
        test_data.sort(key=lambda x: self._extract_image_id(os.path.basename(x[0])) or 0)
        print(f"✅ Données de test: {len(test_data)} images chargées")
        return test_data
    
    def _get_image_path(self, image_id: int) -> str:
        """
        Construit le chemin vers une image à partir de son ID
        
        Args:
            image_id: Numéro de l'image
            
        Returns:
            Chemin complet vers l'image
        """
        # Essayer différents formats de noms de fichiers
        possible_names = [
            f"IM ({image_id}).JPG",  # Format principal
            f"IM({image_id}).JPG",   # Sans espace
            f"metro{image_id:03d}.jpg",  # Format numéroté
            f"image{image_id}.jpg",  # Format simple
        ]
        
        for name in possible_names:
            path = os.path.join(self.image_dir, name)
            if os.path.exists(path):
                return path
        
        # Si aucun format ne marche, retourner le format principal
        return os.path.join(self.image_dir, f"IM ({image_id}).JPG")
    
    def get_all_images(self) -> List[str]:
        """
        Retourne la liste de toutes les images disponibles
        
        Returns:
            Liste des chemins d'images
        """
        if not os.path.exists(self.image_dir):
            return []
            
        images = []
        for file in os.listdir(self.image_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(self.image_dir, file))
        
        return sorted(images)
    
    def get_train_test_split(self):
        """
        Méthode de compatibilité pour metro2025_TEAM1.py
        Retourne les listes d'images d'entraînement et de test
        
        Returns:
            tuple: (liste images test, liste images train)
        """
        print("🔄 Génération des listes d'images pour compatibilité...")
        
        # Charger les données pour obtenir les listes d'images
        train_data = self.load_training_data()
        test_data = self.load_test_data()
        
        # Extraire uniquement les chemins d'images
        train_images = [image_path for image_path, _ in train_data]
        test_images = [image_path for image_path, _ in test_data]
        
        print(f"📊 Split: {len(test_images)} test, {len(train_images)} train")
        
        # Retourner dans l'ordre attendu par metro2025_TEAM1.py (test, train)
        return test_images, train_images
    
    def verify_split_compliance(self) -> Dict[str, Any]:
        """
        Vérifie que le split train/test respecte les règles du projet
        1/3 apprentissage (images 3, 6, 9, ...)
        2/3 test (images 1, 2, 4, 5, 7, 8, ...)
        
        Returns:
            Dictionnaire avec les statistiques de conformité
        """
        print("🔍 Vérification de la conformité du split train/test...")
        
        # Charger les données
        train_data = self.load_training_data()
        test_data = self.load_test_data()
        
        # Extraire les IDs d'images
        train_ids = set()
        test_ids = set()
        
        for image_path, _ in train_data:
            image_name = os.path.basename(image_path)
            image_id = self._extract_image_id(image_name)
            if image_id:
                train_ids.add(image_id)
        
        for image_path, _ in test_data:
            image_name = os.path.basename(image_path)
            image_id = self._extract_image_id(image_name)
            if image_id:
                test_ids.add(image_id)
        
        # Vérifier les règles
        expected_train = set(range(3, 262, 3))  # 3, 6, 9, ..., 261
        expected_test = set(range(1, 262)) - expected_train  # Le reste
        
        train_correct = train_ids == expected_train
        test_correct = test_ids == expected_test
        
        stats = {
            'train_images': len(train_ids),
            'test_images': len(test_ids),
            'total_images': len(train_ids) + len(test_ids),
            'train_compliance': train_correct,
            'test_compliance': test_correct,
            'expected_train': len(expected_train),
            'expected_test': len(expected_test),
            'train_ids': sorted(list(train_ids)),
            'test_ids': sorted(list(test_ids))
        }
        
        print(f"📊 Split actuel: {len(train_ids)} train, {len(test_ids)} test")
        print(f"📊 Split attendu: {len(expected_train)} train, {len(expected_test)} test")
        print(f"✅ Train conforme: {train_correct}")
        print(f"✅ Test conforme: {test_correct}")
        
        return stats
    
    def _extract_image_id(self, image_name: str) -> int:
        """
        Extrait l'ID numérique d'un nom d'image
        
        Args:
            image_name: Nom du fichier image
            
        Returns:
            ID numérique ou None si pas trouvé
        """
        import re
        
        # Patterns pour extraire l'ID
        patterns = [
            r'IM \((\d+)\)\.JPG',  # IM (123).JPG
            r'IM\((\d+)\)\.JPG',   # IM(123).JPG
            r'metro(\d+)\.jpg',    # metro123.jpg
            r'image(\d+)\.jpg',    # image123.jpg
        ]
        
        for pattern in patterns:
            match = re.search(pattern, image_name, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return None
    
    def export_annotations_summary(self, output_file: str = "annotations_summary.txt"):
        """
        Exporte un résumé des annotations pour vérification
        
        Args:
            output_file: Fichier de sortie
        """
        train_data = self.load_training_data()
        test_data = self.load_test_data()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=== RÉSUMÉ DES ANNOTATIONS (Fichiers MAT) ===\n\n")
            
            f.write(f"DONNÉES D'APPRENTISSAGE: {len(train_data)} images\n")
            f.write("-" * 50 + "\n")
            
            train_total_annotations = 0
            for image_path, annotations in train_data:
                image_name = os.path.basename(image_path)
                train_total_annotations += len(annotations)
                f.write(f"{image_name}: {len(annotations)} panneaux\n")
            
            f.write(f"\nTotal annotations apprentissage: {train_total_annotations}\n\n")
            
            f.write(f"DONNÉES DE TEST: {len(test_data)} images\n")
            f.write("-" * 50 + "\n")
            
            test_total_annotations = 0
            for image_path, annotations in test_data:
                image_name = os.path.basename(image_path)
                test_total_annotations += len(annotations)
                f.write(f"{image_name}: {len(annotations)} panneaux\n")
            
            f.write(f"\nTotal annotations test: {test_total_annotations}\n")
            f.write(f"TOTAL GÉNÉRAL: {train_total_annotations + test_total_annotations} annotations\n")
        
        print(f"📄 Résumé exporté dans {output_file}")


def main():
    """Test du DataLoader"""
    loader = DataLoader()
    
    # Vérifier la conformité du split
    stats = loader.verify_split_compliance()
    
    # Exporter le résumé
    loader.export_annotations_summary()
    
    print("\n" + "=" * 50)
    print("RÉSUMÉ FINAL:")
    print(f"Images d'apprentissage: {stats['train_images']}")
    print(f"Images de test: {stats['test_images']}")
    print(f"Total: {stats['total_images']}")


if __name__ == "__main__":
    main() 