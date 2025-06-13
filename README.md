# Système de Détection Automatique des Panneaux de Métro Parisien team12

Système complet de détection et classification automatique des panneaux de métro parisien utilisant YOLO pour la segmentation et un classificateur pour l'identification des lignes.

## Setup

### 1. Préparation des données
- Placez vos images de test dans le dossier `challenge/BD_CHALLENGE/`
- Placez votre fichier de vérités terrain (.mat) dans le dossier `challenge/`

### 2. Installation des dépendances
```bash
pip install -r requirements.txt
```

### 3. Exécution
```bash
python metroChallenge.py
```

## Utilisation

### Interface Graphique Complète
Pour faire gagner du temps au correcteur, nous avons développé une interface graphique complète qui permet de lancer le challenge via le bouton **"Challenge"**.

### Workflow 
1. Choisissez votre facteur de redimensionnement (ou gardez 1.0)
2. Le workflow automatique détecte chaque panneau de métro et les classifie
3. Les métriques sont automatiquement calculées par rapport aux vérités terrain
4. L'interface graphique permet de visualiser les résultats et chaque image du challenge avec ses détections

### Export des Résultats
- **Export .mat** : Exporte les prédictions au format MATLAB
- **Export Results** : Exporte les métriques de qualité

## Architecture

Nous avons implémenté ce système en POO pour avoir un code modulaire et propre suivant une clean architecture. Les noms de fichiers peuvent différer car notre but est de proposer au correcteur un système complet sans qu'il ait à lancer beaucoup de commandes.

### Structure du Code

#### `src/`
Contient tous les modules responsables de :
- La segmentation (YOLO)
- L'entraînement du classificateur
- La classification des ROIs
- Les prédictions
- Autres utilitaires

#### `ui/`
Englobe les modules de :
- L'interface graphique
- L'évaluation des métriques par rapport aux vérités terrain
- La gestion des différents workflows implémentés

### Documentation

Pour découvrir la méthodologie complète utilisée pour répondre à la demande de la société IMAGIK, consultez le document `teams12article.pdf`. Vous y trouverez aussi une analyse détaillée des résultats.

Chaque module, classe et fonction est documenté avec une en-tête détaillée décrivant son rôle, ses paramètres et son comportement attendu.

#### Roadmap d'exploration du projet

Si vous souhaitez explorer le projet, nous vous conseillons de suivre cette roadmap :

**1. Architecture principale (dossier `src/`)**
1. `preprocessing.py` - Préparation des données
2. `dataloader.py` - Chargement des datasets
3. `yolo_segmentation.py` - Détection YOLO
4. `classification.py` - Classification des lignes
5. `detector.py` - Module principal de détection

**2. Interface utilisateur (dossier `ui/`)**
- `gui_main.py` - Interface principale
- `evaluation.py` - Calcul des métriques
- `metrics_formatter.py` - Formatage des résultats
- `data_processor.py` - Traitement des données

**3. Données et modèles**
- `models/` - Modèles sauvegardés
- `data/` - Images d'entraînement et de validation
- `runs/` et `yolotrain/` - Utilisés pour le fine-tuning de TinyYOLO

### Workflows Disponibles

1. **Challenge** : Test sur dataset personnalisé avec métriques automatiques
2. **Image Unique** : Test sur une image seule
3. **Pipeline Complet** : Démonstration de l'entraînement et des prédictions sur le set de validation

## Métriques Calculées

- Précision, Rappel, F1-Score, Accuracy (Détection et Classification)
- Métriques par ligne de métro
- Moyennes Macro et Weighted
- Statistiques détaillées (TP, FP, FN)

Le système est conçu pour être utilisé facilement par le correcteur avec une interface intuitive et des exports automatiques des résultats.
