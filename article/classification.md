# LineClassifier

## Vue d'ensemble
Le `LineClassifier` utilise l'apprentissage automatique avancé pour identifier les numéros de ligne de métro dans les ROIs segmentées. Combine analyse couleur, extraction de caractéristiques et classificateurs d'ensemble.

## Fonctionnalités principales

### Classification multi-modale
- **Analyse couleur** : Statistiques multi-espaces (BGR, HSV, Lab)
- **Analyse de forme** : Caractéristiques HOG + LBP + géométrie
- **Fusion d'ensemble** : RandomForest + SVM avec vote pondéré

### Extraction de caractéristiques

#### Caractéristiques couleur
- Zones d'analyse : centre (logo/chiffre) et anneau (fond)
- Espaces colorimétriques : BGR, HSV, Lab
- Statistiques : moyenne, écart-type, médiane, quartiles
- Histogrammes de couleur par zone

#### Caractéristiques de forme
- **HOG** : Histogrammes de gradients orientés
- **LBP** : Motifs binaires locaux (texture)
- **Géométrie** : Aire, périmètre, circularité, moments de Hu
- Prétraitement : égalisation, binarisation adaptative

### Clasificateurs d'ensemble
- **Couleur** : RandomForest (100 arbres)
- **Forme** : SVM RBF (C=10, probabilité)
- **Ensemble** : Vote pondéré (60% RF, 40% SVM)

## Configuration via constants.py

Paramètres dans `CLASS_PARAMS` :

```python
CLASS_PARAMS = {
    'roi_margin': 2,              # Marge autour des ROI
    'digit_size': (32, 32),       # Taille normalisée
    'hog_cells_per_block': (2, 2), # Paramètres HOG
    'hog_pixels_per_cell': (8, 8),
    'hog_orientations': 9
}
```

## API principale

```python
classifier = LineClassifier()

# Entraînement
training_data = [(roi_image, line_num), ...]
classifier.train_advanced_classifier(training_data)

# Classification
result = classifier.classify(roi_image)
# Returns: {
#   'line_num': int,                    # Ligne prédite
#   'confidence': float,                # Confiance globale
#   'color_prediction': (int, float),   # Résultat couleur
#   'digit_prediction': (int, float),   # Résultat forme
#   'ensemble_prediction': (int, float), # Résultat ensemble
#   'all_scores': {int: float}          # Scores par ligne
# }

# Sauvegarde/Chargement
classifier.save_model('model.pkl')
classifier.load_model('model.pkl')
```

## Pipeline de classification

```
ROI BGR → Prétraitement → Extraction features → Clasificateurs → Fusion → Résultat
```

### Détail du pipeline
1. **Extraction couleur** : Masques centre/anneau + stats multi-espaces
2. **Extraction forme** : Binarisation + HOG + LBP + géométrie  
3. **Classification** : 3 classificateurs en parallèle
4. **Fusion** : Vote pondéré avec confiances

## Algorithmes utilisés

### HOG (Histogram of Oriented Gradients)
- Détection de formes par gradients
- 9 orientations, cellules 8×8, blocs 2×2
- Robuste aux variations d'éclairage

### LBP (Local Binary Pattern) 
- Analyse de texture locale
- Rayon 3, 24 points, méthode uniforme
- Invariant aux rotations

### Ensemble Learning
- **RandomForest** : Résistant au surapprentissage
- **SVM RBF** : Séparation non-linéaire optimale
- **Vote pondéré** : Combine les forces de chaque approche

## Gestion des erreurs

### Classification de secours
- Si modèle non entraîné → classification couleur simple
- Distance Lab avec références couleur métro
- Confiance basée sur proximité colorimétrique

### Robustesse
- Gestion d'exceptions à tous niveaux
- Normalisation automatique des features
- Validation des données d'entrée

## Métriques de performance

Le classificateur suit :
- **Précision couleur** : Performance du classificateur couleur
- **Précision forme** : Performance du classificateur forme  
- **Précision ensemble** : Performance globale
- **Distribution classes** : Équilibrage des données

## Structure des résultats

```python
result = {
    'line_num': 3,                        # Ligne prédite finale
    'confidence': 0.89,                   # Confiance globale  
    'color_prediction': (3, 0.92),       # (ligne, confiance) couleur
    'digit_prediction': (3, 0.85),       # (ligne, confiance) forme
    'ensemble_prediction': (3, 0.91),    # (ligne, confiance) ensemble
    'all_scores': {1: 0.12, 2: 0.08, 3: 0.89, ...}  # Scores toutes lignes
}
```

## Dépendances
- `sklearn` : Classificateurs ML et preprocessing  
- `cv2` : Traitement d'image et morphologie
- `skimage` : Extraction HOG et LBP
- `numpy` : Calculs numériques
- `src.constants` : Configuration métro

## Notes importantes
- ⚠️ **Prend en entrée ROI BGR** (sorties du segmenteur)
- 🎯 **Modèle sauvegardable** (persistance avec pickle)
- 🧠 **Apprentissage supervisé** (nécessite données annotées)
- ⚡ **Classification temps réel** (optimisé pour performance) 