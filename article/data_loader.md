# DataLoader

## Vue d'ensemble
Le `DataLoader` gère le chargement et la répartition des données d'images du métro parisien pour l'apprentissage automatique. Utilise le fichier `constants.py` pour la configuration.

## Fonctionnalités principales

### Split automatique
- **Train** : Images avec ID multiple de 3 (3, 6, 9, ...)
- **Test** : Toutes les autres images (1, 2, 4, 5, 7, 8, ...)
- Diviseur configurable via `DATA_LOADER['train_id_divisor']`

### Chargement des annotations
- Lecture des fichiers MAT (`Apprentissage.mat`, `Test.mat`) 
- Structure : `{xmin, ymin, xmax, ymax, line}`
- Cache automatique en mémoire

### Gestion des images
- Support formats : JPG, JPEG, PNG
- Détection automatique des noms de fichiers

## Configuration via constants.py

Toutes les valeurs importantes sont centralisées dans `constants.py` :

```python
DATA_LOADER = {
    'default_data_dir': 'data',
    'image_subdir': 'BD_METRO', 
    'docs_subdir': 'docs/progsPython',
    'train_mat_file': 'Apprentissage.mat',
    'test_mat_file': 'Test.mat',
    'train_id_divisor': 3,
    'supported_extensions': ('.jpg', '.jpeg', '.png'),
    'image_name_patterns': [...],  # Regex pour extraction ID
    'image_name_formats': [...]    # Formats de noms de fichiers
}
```

## le data loader permet de :

```python
loader = DataLoader("data") #creer l'instance

train_data = loader.load_training_data()
test_data = loader.load_test_data()
```

## Structure des données
```python
annotation = {
    'xmin': int,  # Coordonnée x minimale
    'ymin': int,  # Coordonnée y minimale  
    'xmax': int,  # Coordonnée x maximale
    'ymax': int,  # Coordonnée y maximale
    'line': int   # Numéro de ligne de métro
}
```