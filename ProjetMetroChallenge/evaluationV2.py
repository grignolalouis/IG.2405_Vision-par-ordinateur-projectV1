# -*- coding: utf-8 -*-
"""
Created on Sun May  4 11:49:42 2025

@author: fross
"""

import numpy as np
import scipy.io

import os
import matplotlib.pyplot as plt
from  myMetroProcessing import processOneMetroImage, draw_rectangle
from PIL import Image
import  skimage as ski


def evaluation(BDREF_path, BDTEST_path, resize_factor):
    # Charger les bases de données .mat
    BDREF = scipy.io.loadmat(BDREF_path)['BD']
    BDTEST = scipy.io.loadmat(BDTEST_path)['BD']

    # Ajuster les dimensions en fonction de la résolution
    BDREF[:, 1:5] = resize_factor * BDREF[:, 1:5]

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
        
        # print(f'Symbol {k} in image ({BDTEST[k, 0]})')
        
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
                if BDREF[kref, 5] == BDTEST[k, 5]:
                    print(f'Symbol {k} in image ({BDTEST[k, 0]}):: = {BDREF[kref, 5]}')
                else:
                    print(f'Symbol {k} in image ({BDTEST[k, 0]}):: != {BDREF[kref, 5]} -> {BDTEST[k, 5]}')
            else:
                plusVector[int(BDTEST[k, 5]) - 1] += 1
                print(f'Symbol {k} in image ({BDTEST[k, 0]}):: + {BDTEST[k, 5]}')

    # Symboles non trouvés
    for k in np.where(~processed)[0]:
        minusVector[int(BDREF[k, 5]) - 1] += 1
        print(f'Symbol {k} in image {BDREF[k, 0]}:: - {BDREF[k, 5]}')

    
    print("\n\n---------------\nConfusion matrix ....\n")
    for k in range(14):
        row = ' '.join(f"{val:3d}" for val in confusionMatrix[k])
        total = np.sum(confusionMatrix[k])
        print(f"{row}  : {total:3d} : + {plusVector[k]:3d} : - {minusVector[k]:3d} :")

    print("... ... ... ... ... ... ... ... ... ... ... ... ... ...")
    col_totals = [np.sum(confusionMatrix[:, k]) for k in range(14)]
    print(' '.join(f"{val:3d}" for val in col_totals))


    # Affichage des résultats sur les performances globales
    # de détection des signes
    # TP : tout ce qui est dans la matrice de confusion
    # FN : tout ce qui est dans minusVector
    # FP : tout ce qui est dans plusVector
    
    TP = np.sum(confusionMatrix)
    FN = np.sum(minusVector)
    FP = np.sum(plusVector)
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    F1 = 2*recall*precision/(recall+precision)
    
    # Calcul d'un accuracy élargie, en considérant des classes rien et en-trop
    accuracy = np.trace(confusionMatrix) / (np.sum(confusionMatrix)+np.sum(plusVector)+np.sum(minusVector))
    
    print('')
    print('--------------------------------------------------------')
    print('SIGN DETECTION')
    print('--------------------------------------------------------')
    print(f'\t recall    = {recall:3.3f}')
    print(f'\t precision = {precision:3.3f}')
    print(f'\t F1-score  = {F1:3.3f}')
    
    print('--------------------------------------------------------')
    print('GLOBAL AND ENLARGED ACCURACY')
    print('--------------------------------------------------------')
    print(f'\nGlobal enlarged accuracy = {accuracy:3.3f}')
    print('')
    
    print('--------------------------------------------------------')
    print('CLASS EVALUATION REPORT')
    print('--------------------------------------------------------')
    print('Ligne\t\tPrecision\tRecall\tf1-score\tSupport')
    nbClasses =14
    Recall  = np.zeros(nbClasses)
    Precision  = np.zeros(nbClasses)
    F1Score  = np.zeros(nbClasses)
    support  = np.zeros(nbClasses,dtype=int)
    valide = np.zeros(nbClasses,dtype=int)
    
    mRecall     = 0
    mPrecision  = 0
    mF1         = 0
    
    wRecall     = 0
    wPrecision  = 0
    wF1         = 0
    
    nbValidClass = 0
    
    for k in range(nbClasses):
        support[k]  = int(np.sum(confusionMatrix[k,:]))
        if support[k] :
            nbValidClass+=1
            
            TP = confusionMatrix[k,k]
            FN = np.sum(confusionMatrix[k,:])-TP+minusVector[k]
            FP = np.sum(confusionMatrix[:,k])-TP+plusVector[k]
            
            Recall[k]   = TP/(TP+FN)
            Precision[k]= TP/(TP+FP)
            F1Score[k]  = 2*Recall[k] *Precision[k]/(Recall[k] +Precision[k])
            
            print(f'  {k+1:2d}\t\t  {Precision[k]:3.3f}\t\t {Recall[k]:3.3f}\t  {F1Score[k]:3.3f}\t\t {support[k]:3d}')
            
            # macro
            mRecall     += Recall[k]
            mPrecision  += Precision[k]
            mF1         += F1Score[k]
            
            # weighted
            wRecall     += Recall[k]*support[k]
            wPrecision  += Precision[k]*support[k]
            wF1         += F1Score[k]*support[k]
            
    nb = np.sum(support)        
    print('--------------------------------------------------------')   
    print(f'Macro \t\t  {mPrecision/nbValidClass:3.3f}\t\t {mRecall/nbValidClass:3.3f}\t  {mF1/nbValidClass:3.3f}\t\t {nbValidClass:2d} classes')
    print(f'Weighted \t  {wPrecision/nb:3.3f}\t\t {wRecall/nb:3.3f}\t  {wF1/nb:3.3f}\t\t {np.sum(support):3d} signs')
   
    
    # Summary
    # # Taux de reconnaissance
    # print("\n\n---------------\nTaux de reconnaissance")
    # reconnus = 0
    # for k in range(14):
    #     total = np.sum(confusionMatrix[k]) + minusVector[k]
    #     taux_reconnu = 100 * confusionMatrix[k, k] / total if total > 0 else 0
    #     taux_plus = 100 * plusVector[k] / total if total > 0 else 0
    #     print(f"{k+1:2d} : {taux_reconnu:5.2f} %  - Ajouts : {taux_plus:5.2f} %")
    #     reconnus += confusionMatrix[k, k]

    # total_all = np.sum(confusionMatrix) + np.sum(minusVector)
    # print("---------------")
    # print(f"Taux de reconnaissance global = {100 * reconnus / total_all:.2f} %")
    # print(f"Taux de symboles en plus = {100 * np.sum(plusVector) / total_all:.2f} %")
    # print("---------------")
    
    
    
def compareTestandRef(imageFilesList,challengeDirectory,BDREF_path, BDTEST_path, resize_factor):

    ftsize =30
    
    
    # Charger les bases de données .mat
    BDREF = scipy.io.loadmat(BDREF_path)['BD']
    BDTEST = scipy.io.loadmat(BDTEST_path)['BD']
    
    
    indices  = BDREF[:,0].squeeze()
    indices  = sorted(set(indices))
    
    for k in range(len(indices)):
        
        n_val = indices[k]
        
        # LOAD IMAGE ------------------------------
        nom = f'IM ({n_val}).JPG'
        print(f"---- [{n_val}] : {nom} -----")
        im_path = os.path.join(challengeDirectory, nom)
        im = np.array(Image.open(im_path).convert('RGB')) / 255.0
        
        
        # DISPLAY GROUND TRUTH ---------------------
        ind = np.argwhere(BDREF[:,0]== n_val).flatten()
        bd = BDREF[ind,:]
        
        plt.figure(figsize=(45,15))
        plt.subplot(1,2,1)
        plt.imshow(im)   
        plt.title(f'{nom} :  {bd[:,5]}',fontsize=ftsize)
        
        for k in range(len(ind)):
            draw_rectangle(bd[k,3], bd[k,4], bd[k,1], bd[k,2], 'g')
    
        # DISPLAY RECOGNITION  ---------------------
        if resize_factor != 1:
            im_resized = ski.transform.resize(im, (int(im.shape[0] * resize_factor), int(im.shape[1] * resize_factor)),
                            anti_aliasing=True, preserve_range=True).astype(im.dtype)
        else:
            im_resized = im
            
        ind = np.argwhere(BDTEST[:,0]== n_val).flatten()
        bd = BDTEST[ind,:]
         
        plt.subplot(1,2,2)
        plt.imshow(im_resized)   
        plt.title(f'{nom} : {bd[:,5]}',fontsize=ftsize)
        
       
        for k in range(len(ind)):
            draw_rectangle(bd[k,3], bd[k,4], bd[k,1], bd[k,2], 'g')
            
        plt.show()    