# -*- coding: utf-8 -*-
"""
Created on Sun May  4 11:49:42 2025

@author: fross
"""

import numpy as np
import scipy.io

def evaluation(BDREF_path, BDTEST_path, resize_factor):
    # Charger les bases de données .mat
    BDREF = scipy.io.loadmat(BDREF_path)['BD']
    BDTEST = scipy.io.loadmat(BDTEST_path)['BD']

    # Ajuster les dimensions en fonction de la résolution
    BDTEST[:, 1:5] = resize_factor * BDTEST[:, 1:5]

    # Calcul des centroïdes et diamètres moyens
    I = np.mean(BDREF[:, 1:3], axis=1)
    J = np.mean(BDREF[:, 3:5], axis=1)
    D = np.round((BDREF[:, 2] - BDREF[:, 1] + BDREF[:, 4] - BDREF[:, 3]) / 2)
    maxDecPer = 0.1

    confusionMatrix = np.zeros((14, 14), dtype=int)
    plusVector = np.zeros(14, dtype=int)
    minusVector = np.zeros(14, dtype=int)
    processed = np.zeros(BDREF.shape[0], dtype=bool)

    # Évaluation
    for k in range(BDTEST.shape[0]):
        n = BDTEST[k, 0]
        ind = np.where(BDREF[:, 0] == n)[0]

        i = np.mean(BDTEST[k, 1:3])
        j = np.mean(BDTEST[k, 3:5])
        d = np.sqrt((I[ind] - i) ** 2 + (J[ind] - j) ** 2)
        
        if len(d) > 0:
            mind = np.min(d)
            p = np.argmin(d)
            kref = ind[p]

            if mind <= maxDecPer * D[kref]:
                confusionMatrix[int(BDREF[kref, 5]) - 1, int(BDTEST[k, 5]) - 1] += 1
                processed[kref] = True
            else:
                plusVector[int(BDTEST[k, 5]) - 1] += 1

    # Symboles non trouvés
    for k in np.where(~processed)[0]:
        minusVector[int(BDREF[k, 5]) - 1] += 1

    # Affichage des résultats
    print("\n\n---------------\nConfusion matrix ....\n")
    for k in range(14):
        row = ' '.join(f"{val:3d}" for val in confusionMatrix[k])
        total = np.sum(confusionMatrix[k])
        print(f"{row}  : {total:3d} : + {plusVector[k]:3d} : - {minusVector[k]:3d} :")

    print("... ... ... ... ... ... ... ... ... ... ... ... ... ...")
    col_totals = [np.sum(confusionMatrix[:, k]) for k in range(14)]
    print(' '.join(f"{val:3d}" for val in col_totals))

    # Taux de reconnaissance
    print("\n\n---------------\nTaux de reconnaissance")
    reconnus = 0
    for k in range(14):
        total = np.sum(confusionMatrix[k]) + minusVector[k]
        taux_reconnu = 100 * confusionMatrix[k, k] / total if total > 0 else 0
        taux_plus = 100 * plusVector[k] / total if total > 0 else 0
        print(f"{k+1:2d} : {taux_reconnu:5.2f} %  - Ajouts : {taux_plus:5.2f} %")
        reconnus += confusionMatrix[k, k]

    total_all = np.sum(confusionMatrix) + np.sum(minusVector)
    print("---------------")
    print(f"Taux de reconnaissance global = {100 * reconnus / total_all:.2f} %")
    print(f"Taux de symboles en plus = {100 * np.sum(plusVector) / total_all:.2f} %")
    print("---------------")
    
    
# RUN
evaluation ('Test.mat', 'myResults.mat',1)
# evaluation ('Apprentissage.mat', 'myResults.mat',1)