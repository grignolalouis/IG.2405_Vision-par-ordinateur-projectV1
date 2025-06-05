"""
Script de test pour le nouveau DataLoader avec fichiers MAT
"""

from src.data_loader import DataLoader
import os

def test_dataloader():
    print("🧪 Test du DataLoader avec fichiers MAT")
    print("=" * 50)
    
    # Initialiser le loader
    loader = DataLoader()
    
    # Vérifier l'existence des fichiers MAT
    print("📁 Vérification des fichiers:")
    print(f"Apprentissage.mat: {os.path.exists(loader.train_mat)}")
    print(f"Test.mat: {os.path.exists(loader.test_mat)}")
    print(f"Dossier BD_METRO: {os.path.exists(loader.image_dir)}")
    
    if os.path.exists(loader.image_dir):
        image_count = len([f for f in os.listdir(loader.image_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"Nombre d'images: {image_count}")
    
    print("\n" + "=" * 50)
    
    # Test de chargement
    try:
        print("🔄 Test de chargement des données...")
        
        # Charger les données d'apprentissage
        train_data = loader.load_training_data()
        print(f"✅ Données d'apprentissage: {len(train_data)} images")
        
        # Charger les données de test
        test_data = loader.load_test_data()
        print(f"✅ Données de test: {len(test_data)} images")
        
        # Compter les annotations
        train_annotations = sum(len(annotations) for _, annotations in train_data)
        test_annotations = sum(len(annotations) for _, annotations in test_data)
        
        print(f"📊 Annotations apprentissage: {train_annotations}")
        print(f"📊 Annotations test: {test_annotations}")
        print(f"📊 Total annotations: {train_annotations + test_annotations}")
        
        # Vérifier la conformité du split
        print("\n🔍 Vérification de la conformité du split...")
        stats = loader.verify_split_compliance()
        
        # Résumé final
        print("\n" + "=" * 50)
        print("📈 RÉSUMÉ FINAL:")
        print(f"Images d'apprentissage: {stats['train_images']}")
        print(f"Images de test: {stats['test_images']}")
        print(f"Total: {stats['total_images']}")
        print(f"Conformité train: {stats['train_compliance']}")
        print(f"Conformité test: {stats['test_compliance']}")
        
        # Afficher quelques exemples
        if train_data:
            print(f"\n📝 Exemple train - {os.path.basename(train_data[0][0])}: {len(train_data[0][1])} panneaux")
            # Afficher une annotation d'exemple
            if train_data[0][1]:
                ann = train_data[0][1][0]
                print(f"    Annotation: Ligne {ann['line']}, bbox=({ann['xmin']},{ann['ymin']},{ann['xmax']},{ann['ymax']})")
        
        if test_data:
            print(f"📝 Exemple test - {os.path.basename(test_data[0][0])}: {len(test_data[0][1])} panneaux")
            # Afficher une annotation d'exemple
            if test_data[0][1]:
                ann = test_data[0][1][0]
                print(f"    Annotation: Ligne {ann['line']}, bbox=({ann['xmin']},{ann['ymin']},{ann['xmax']},{ann['ymax']})")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataloader() 