# 🚇 PROJET MÉTRO PARISIEN - DÉTECTION AUTOMATIQUE (TEAM1)

## 📋 OBJECTIF

Système de détection automatique des panneaux de métro parisien utilisant la vision par ordinateur. Le système détecte et classifie les panneaux des lignes 1-14 (excluant 3bis et 7bis) avec une méthodologie 100% efficace.

## 🚀 POINT D'ENTRÉE PRINCIPAL

Le projet a été refactorisé pour avoir un **point d'entrée unique** avec pipeline automatique complet:

```bash
python main.py
```

## ⚙️ PIPELINE AUTOMATIQUE

Le système exécute automatiquement les étapes suivantes:

### 1. 🔄 Chargement des données
- **Split train/test automatique** basé sur les IDs d'images:
  - **Train**: Images avec ID multiple de 3 (3, 6, 9, 12, ...)
  - **Test**: Images avec ID non multiple de 3 (1, 2, 4, 5, 7, 8, ...)
- Chargement des annotations depuis les fichiers MAT (si disponibles)

### 2. 🧠 Entraînement du modèle
- Entraînement automatique sur les données d'apprentissage
- Utilisation des images à **taille originale** (pas de redimensionnement)
- Conservation des coordonnées exactes

### 3. 💾 Sauvegarde du modèle
- Modèle entraîné sauvé dans `models/metro_detector_trained.pkl`
- Persistence pour réutilisation ultérieure

### 4. 🔍 Prédiction sur le test
- Détection automatique sur toutes les images de test
- Coordonnées préservées à l'échelle originale

### 5. 📊 Calcul des métriques
- Métriques de détection (Précision, Rappel, F1-Score)
- Métriques de classification par ligne
- Analyse détaillée des performances

### 6. 🖼️ Visualisation interactive
- Interface graphique avec navigation
- Comparaison vérité terrain vs prédictions
- Métriques en temps réel

### 7. 📤 Export des résultats
- Export automatique au format MAT: `results_test_TEAM1.mat`
- Format compatible avec les exigences du projet

## 📁 STRUCTURE DU PROJET

```
projetV1/
├── main.py                    # 🎯 POINT D'ENTRÉE PRINCIPAL
├── gui_visualizer.py          # Interface graphique avec pipeline automatique
├── metro2025_TEAM1.py         # Script de compatibilité (si nécessaire)
├── src/                       # Modules du système
│   ├── __init__.py
│   ├── data_loader.py         # Chargement avec split automatique
│   ├── detector.py            # Pipeline principal de détection
│   ├── preprocessing.py       # Prétraitement (sans redimensionnement)
│   ├── segmentation.py        # Segmentation des ROIs
│   ├── classification.py      # Classification des lignes
│   └── constants.py           # Paramètres et couleurs métro
├── data/
│   └── BD_METRO/             # 📸 Images du projet (261 images)
├── docs/                     # Documentation et fichiers MAT
├── models/                   # Modèles entraînés sauvegardés
└── requirements.txt          # Dépendances Python
```

## 🔧 INSTALLATION

1. **Cloner/extraire le projet**:
```bash
cd projetV1
```

2. **Installer les dépendances**:
```bash
pip install -r requirements.txt
```

3. **Vérifier les images**:
   - Placer les 261 images dans `data/BD_METRO/`
   - Format accepté: `.jpg`, `.jpeg`, `.png`

## ▶️ UTILISATION

### Lancement rapide:
```bash
python main.py
```

### Interface graphique:
- **Pipeline automatique**: Se lance au démarrage
- **Navigation**: Boutons ◀/▶ pour parcourir les images
- **Modes d'affichage**: 
  - 🎯 Vérité terrain
  - 🤖 Prédictions
  - 📊 Comparaison
- **Actions**:
  - 🔄 Relancer pipeline
  - 💾 Exporter résultats MAT
  - 📊 Rapport détaillé

## 🎯 SPÉCIFICATIONS TECHNIQUES

### Détection:
- **Lignes supportées**: 1-14 (excluant 3bis et 7bis)
- **Méthode**: Segmentation couleur + Classification HOG+SVM
- **Images**: Taille originale préservée
- **Coordonnées**: Exactes sans conversion d'échelle

### Split train/test:
- **Règle**: ID % 3 == 0 pour train, sinon test
- **Exemples**:
  - Train: IM (3).JPG, IM (6).JPG, IM (9).JPG...
  - Test: IM (1).JPG, IM (2).JPG, IM (4).JPG...

### Performance:
- Métriques calculées avec seuil IoU = 0.5
- Classification basée sur les couleurs officielles RATP
- Export compatible format projet

## 📊 MÉTRIQUES CALCULÉES

- **Précision** (Precision): TP / (TP + FP)
- **Rappel** (Recall): TP / (TP + FN)  
- **F1-Score**: 2 * (Precision * Recall) / (Precision + Recall)
- **Précision classification**: Détections correctement classifiées
- **Statistiques par ligne**: Performance individuelle par ligne de métro

## 📤 FICHIERS GÉNÉRÉS

- `results_test_TEAM1.mat`: Résultats au format MAT
- `models/metro_detector_trained.pkl`: Modèle entraîné
- Rapports de performance (optionnels)

## 🔍 DÉPANNAGE

### Erreurs communes:
- **Images manquantes**: Vérifier `data/BD_METRO/`
- **Modules manquants**: `pip install -r requirements.txt`
- **Erreurs de path**: Lancer depuis le dossier racine

### Mode debug:
- Consulter les logs dans la console
- Vérifier l'état du pipeline dans l'interface
- Utiliser le bouton "Rapport détaillé"

## 👥 ÉQUIPE

**TEAM1** - Projet IG.2405 Vision par ordinateur  
ISEP 2025-2026

## 📝 NOTES

- Système optimisé pour les 261 images du projet
- Pipeline entièrement automatique
- Interface graphique moderne et intuitive
- Export compatible avec les exigences du cours