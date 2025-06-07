# ImagePreprocessor

## Vue d'ensemble
Le `ImagePreprocessor` prépare les images **BGR** (OpenCV) pour la segmentation en appliquant des filtres et conversions d'espace colorimétrique. Utilise les paramètres de `SEG_PARAMS` dans `constants.py`.

## Fonctionnalités principales

### Validation d'entrée
- Vérifie que l'image est bien en couleur (3 canaux)
- Assume le format **BGR** d'OpenCV
- Protection contre les erreurs de format

### Filtrage bilatéral
- Réduit le bruit tout en préservant les contours
- Paramètres configurables : `bilateral_d`, `bilateral_sigma_color`, `bilateral_sigma_space`

### Conversion d'espace colorimétrique  
- Conversion **BGR** vers **Lab** (⚠️ Important : pas RGB!)
- L'espace Lab sépare mieux la luminance de la chrominance
- Utilise `cv2.COLOR_BGR2Lab` spécifiquement pour OpenCV

### Pipeline de prétraitement
- Traitement sans redimensionnement pour préserver les coordonnées exactes
- Retourne l'image traitée, l'originale ET la version filtrée

## Pourquoi BGR → Lab ?

D'après l'analyse colorimétrique :
- **OpenCV charge en BGR** : `[106.56, 117.99, 127.97]` 
- **Conversion BGR→Lab** : `L=126.3, a=129.9, b=135.7`
- **Conversion RGB→Lab** : `L=123.3, a=126.2, b=121.2` ⚠️ **Valeurs différentes !**

➡️ Il est **crucial** d'utiliser la bonne conversion selon la source des images.

## Configuration via constants.py

Les paramètres sont dans `SEG_PARAMS` :

```python
SEG_PARAMS = {
    'bilateral_d': 9,              # Diamètre du voisinage
    'bilateral_sigma_color': 30,   # Sigma couleur  
    'bilateral_sigma_space': 90,   # Sigma spatial
    # ... autres paramètres
}
```

## API principale

```python
preprocessor = ImagePreprocessor()

# Prétraitement complet
result = preprocessor.preprocess(image_bgr)
# Returns: {
#   'processed': image_lab,       # Image Lab filtrée
#   'bgr': image_bgr,            # Image BGR originale
#   'filtered_bgr': filtered,     # Image BGR filtrée
#   'scale_factor': 1.0          # Facteur d'échelle
# }

# Information sur le preprocessing
info = preprocessor.get_preprocessing_info()
# Returns: Format d'entrée/sortie, paramètres, etc.

# Filtrage seul
filtered = preprocessor.apply_bilateral_filter(image)

# Conversion seule  
lab_image = preprocessor.convert_to_lab(image)
```

## Flux de données
```
Image BGR (OpenCV) → Validation → Filtre bilatéral → Conversion BGR→Lab → Résultat
```

## Validation automatique

Le preprocessor vérifie automatiquement :
- ✅ Image en couleur (3 canaux)
- ✅ Format compatible avec BGR→Lab
- ❌ Lève une exception si format incorrect

## Dépendances
- `cv2` : Traitement d'image OpenCV
- `numpy` : Manipulation d'arrays
- `constants` : Paramètres de segmentation

## Notes importantes
- ⚠️ **Assume format BGR** (images chargées avec `cv2.imread()`)
- ⚠️ Utilise `COLOR_BGR2Lab` et non `COLOR_RGB2Lab`
- ✅ Préserve les coordonnées exactes (pas de redimensionnement) 