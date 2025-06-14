"""
Module de chargement et gestion des données pour l'entraînement et le test du système.

Ce module fournit la classe DataLoader qui gère le chargement des images et annotations
depuis les fichiers MATLAB, organise les données selon le split train/test défini,
et assure la conformité avec la structure de données attendue par le système.

Auteur: LGrignola
"""

import os
import re
import scipy.io as sio
from typing import List, Tuple, Dict, Any
from src.constants import DATA_LOADER


class DataLoader:
    def __init__(self, data_dir: str = DATA_LOADER['default_data_dir']):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, DATA_LOADER['image_subdir'])
        self.docs_dir = DATA_LOADER['docs_subdir']
        self.train_mat = os.path.join(self.docs_dir, DATA_LOADER['train_mat_file'])
        self.test_mat = os.path.join(self.docs_dir, DATA_LOADER['test_mat_file'])
        self._annotations = None
        
    def _load_annotations(self) -> Dict[int, List[Dict[str, Any]]]:
        if self._annotations is not None:
            return self._annotations
            
        annotations = {}
        
        for mat_file in [self.train_mat, self.test_mat]:
            if os.path.exists(mat_file):
                try:
                    mat_data = sio.loadmat(mat_file)
                    data_key = next((k for k in mat_data.keys() if not k.startswith('__')), None)
                    
                    if data_key:
                        for row in mat_data[data_key]:
                            if len(row) >= 6:
                                image_id = int(row[0])
                                annotation = {
                                    'xmin': int(row[3]),
                                    'ymin': int(row[1]),
                                    'xmax': int(row[4]),
                                    'ymax': int(row[2]),
                                    'line': int(row[5])
                                }
                                annotations.setdefault(image_id, []).append(annotation)
                                
                except Exception:
                    pass
        
        if not annotations:
            for image_path in self._get_all_images():
                image_id = self._extract_id(os.path.basename(image_path))
                if image_id:
                    annotations[image_id] = []
        
        self._annotations = annotations
        return annotations
        
    def load_training_data(self) -> List[Tuple[str, List[Dict[str, Any]]]]:
        print("Chargement des données d'apprentissage (IDs multiples de 3)...")
        
        annotations = self._load_annotations()
        data = []
        
        for image_id, anns in annotations.items():
            if image_id % DATA_LOADER['train_id_divisor'] == 0:
                image_path = self._get_image_path(image_id)
                if os.path.exists(image_path):
                    data.append((image_path, anns))
                else:
                    print(f"Image non trouvée: {image_path}")
        
        data.sort(key=lambda x: self._extract_id(os.path.basename(x[0])) or 0)
        print(f"Données d'apprentissage: {len(data)} images chargées")
        return data
    
    def load_test_data(self) -> List[Tuple[str, List[Dict[str, Any]]]]:
        print("Chargement des données de test (IDs non multiples de 3)...")
        
        annotations = self._load_annotations()
        data = []
        
        for image_id, anns in annotations.items():
            if image_id % DATA_LOADER['train_id_divisor'] != 0:
                image_path = self._get_image_path(image_id)
                if os.path.exists(image_path):
                    data.append((image_path, anns))
                else:
                    print(f"Image non trouvée: {image_path}")
        
        data.sort(key=lambda x: self._extract_id(os.path.basename(x[0])) or 0)
        print(f"Données de test: {len(data)} images chargées")
        return data
    
    def _get_image_path(self, image_id: int) -> str:
        formats = [fmt.format(image_id) for fmt in DATA_LOADER['image_name_formats']]
        
        for name in formats:
            path = os.path.join(self.image_dir, name)
            if os.path.exists(path):
                return path
        
        return os.path.join(self.image_dir, DATA_LOADER['image_name_formats'][0].format(image_id))
    
    def _get_all_images(self) -> List[str]:
        if not os.path.exists(self.image_dir):
            return []
            
        return sorted([
            os.path.join(self.image_dir, f) 
            for f in os.listdir(self.image_dir) 
            if f.lower().endswith(DATA_LOADER['supported_extensions'])
        ])
    
    def verify_split_compliance(self) -> Dict[str, Any]:
        print("Vérification de la conformité du split train/test...")
        
        train_data = self.load_training_data()
        test_data = self.load_test_data()
        
        train_ids = set()
        test_ids = set()
        
        for image_path, _ in train_data:
            image_name = os.path.basename(image_path)
            image_id = self._extract_id(image_name)
            if image_id:
                train_ids.add(image_id)
        
        for image_path, _ in test_data:
            image_name = os.path.basename(image_path)
            image_id = self._extract_id(image_name)
            if image_id:
                test_ids.add(image_id)
        
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
        
        print(f"Train conforme: {train_correct}")
        print(f"Test conforme: {test_correct}")
        
        return stats
    
    def _extract_id(self, image_name: str) -> int:
        for pattern in DATA_LOADER['image_name_patterns']:
            match = re.search(pattern, image_name, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return None

def main():
    loader = DataLoader()
    
    stats = loader.verify_split_compliance()
    
    print("\n" + "=" * 50)
    print("Résumé final:")
    print(f"Images d'apprentissage: {stats['train_images']}")
    print(f"Images de test: {stats['test_images']}")
    print(f"Total: {stats['total_images']}")

if __name__ == "__main__":
    main() 