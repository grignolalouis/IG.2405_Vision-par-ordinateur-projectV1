import pandas as pd

print("=== EXAMINATION DES FICHIERS EXCEL ===")

# Apprentissage
print("\n📚 APPRENTISSAGE.XLS:")
try:
    df_train = pd.read_excel('docs/Apprentissage.xls')
    print(f"Taille: {df_train.shape}")
    print(f"Colonnes: {list(df_train.columns)}")
    print("\nPremières lignes:")
    print(df_train.head())
    print("\nTypes de données:")
    print(df_train.dtypes)
except Exception as e:
    print(f"Erreur: {e}")

# Test
print("\n📚 TEST.XLS:")
try:
    df_test = pd.read_excel('docs/Test.xls')
    print(f"Taille: {df_test.shape}")
    print(f"Colonnes: {list(df_test.columns)}")
    print("\nPremières lignes:")
    print(df_test.head())
    print("\nTypes de données:")
    print(df_test.dtypes)
except Exception as e:
    print(f"Erreur: {e}") 