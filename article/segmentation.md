# MetroSegmenter

## Vue d'ensemble
Le `MetroSegmenter` détecte et segmente les panneaux de métro dans les images Lab en utilisant la segmentation par couleur. Utilise les couleurs et paramètres de `constants.py`.

## Fonctionnalités principales

### Segmentation par couleur
- Utilise l'espace **Lab** pour une meilleure séparation des couleurs
- Crée des plages de couleurs pour chaque ligne de métro (1-14)
- Tolérance configurable via `color_tolerance`

### Pipeline de détection
1. **Masques couleur** : Segmentation binaire par ligne
2. **Morphologie** : Nettoyage (fermeture + ouverture)
3. **Filtrage** : Aire et circularité
4. **NMS** : Suppression des doublons

### Filtres qualité
- **Aire** : `min_area` ↔ `max_area` 
- **Circularité** : `min_circularity` ↔ `max_circularity`
- **IoU** : Non-Max Suppression avec `nms_threshold`

## Configuration via constants.py

Les paramètres sont dans `SEG_PARAMS` :

```python
SEG_PARAMS = {
    'color_tolerance': 50,           # Tolérance couleur Lab
    'morph_kernel_close': (5, 5),   # Kernel fermeture
    'morph_kernel_open': (5, 5),    # Kernel ouverture  
    'min_area': 1500,               # Aire minimale
    'max_area': 50000,              # Aire maximale
    'min_circularity': 0.8,         # Circularité min
    'max_circularity': 1.5,         # Circularité max
    'nms_threshold': 0.2,           # Seuil IoU pour NMS
    'roi_margin': 2                 # Marge autour des ROI
}
```

Les couleurs sont dans `METRO_COLORS` (RGB → Lab automatiquement).

## API principale

```python
segmenter = MetroSegmenter()

# Segmentation complète
rois = segmenter.segment(lab_image)
# Returns: [
#   {
#     'bbox': (x1, y1, x2, y2),        # Coordonnées
#     'line_num_color': int,            # Ligne détectée
#     'confidence': 1.0                 # Confiance
#   }, ...
# ]

# Masque pour une ligne spécifique
mask = segmenter.create_color_mask(lab_image, line_num=1)

# Candidats avant NMS
candidates = segmenter.find_candidates(lab_image)

# Application NMS
filtered = segmenter.apply_nms(candidates)
```

## Flux de données
```
Image Lab → Masques couleur → Morphologie → Contours → Filtrage → NMS → ROIs
```

## Algorithmes utilisés

### Segmentation couleur Lab
- Conversion RGB → Lab pour chaque couleur de ligne
- Plages de tolérance dans l'espace Lab
- Masquage binaire avec `cv2.inRange()`

### Morphologie mathématique
- **Fermeture** : Remplit les trous (érosion + dilatation)
- **Ouverture** : Supprime le bruit (dilatation + érosion)
- Kernels elliptiques configurables

### Filtrage géométrique
- **Aire** : Élimine objets trop petits/grands
- **Circularité** : `4πA/P²` pour détecter formes rondes
- Paramètres optimisés automatiquement

### Non-Max Suppression
- Calcul IoU (Intersection over Union)
- Garde les candidats avec plus grande aire
- Évite les détections multiples

## Structure des résultats

```python
roi = {
    'bbox': (x1, y1, x2, y2),      # Coordonnées absolues
    'line_num_color': 3,            # Ligne 3 (vert olive)  
    'confidence': 1.0               # Confiance fixe
}
```

## Dépendances
- `cv2` : Morphologie et contours OpenCV
- `numpy` : Calculs numériques
- `constants` : Couleurs métro et paramètres

## Notes importantes
- ⚠️ **Prend en entrée image Lab** (sortie du preprocessing)
- ✅ **Coordonnées préservées** (pas de redimensionnement)
- 🎯 **Optimisé automatiquement** (paramètres SEG_PARAMS) 