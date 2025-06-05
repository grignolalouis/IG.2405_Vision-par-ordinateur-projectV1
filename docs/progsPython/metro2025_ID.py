# -*- coding: utf-8 -*-
"""
Created on Sun May  4 11:17:49 2025

@author: fross
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import scipy.io as sio

def draw_rectangle(x1, x2, y1, y2, color):
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)

def metro2025(type_,viewImages = 1):
    plt.close('all')
    n = np.arange(1, 262)
    ok = True
    if type_ == 'Test':
        num_images = n[n % 3 != 0]
        BD = []
        
    elif type_ == 'Learn':
        num_images = n[n % 3 == 0]
        GT = sio.loadmat('Apprentissage.mat')['BD']
        BD = []
    else:
        print("Bad identifier (should be 'Learn' or 'Test')")
        return None, None

    
    resize_factor = 1

    for n_val in num_images:
        nom = f'IM ({n_val})'
        im_path = os.path.join('BD', f'{nom}.JPG')
        im = np.array(Image.open(im_path).convert('RGB')) / 255.0

        if viewImages:
            fig = plt.figure(figsize=(45,15))
            plt.subplot(1,2,1)
            plt.imshow(im)
    
        if type_ == 'Learn':
            ind = np.where(GT[:, 0] == n_val)[0]
            titre = 'GT: '
            for k in ind:
                bbox = np.round(resize_factor * GT[k, 1:5]).astype(int)
                if viewImages:
                    draw_rectangle(bbox[2], bbox[3], bbox[0], bbox[1], 'g')
                    titre += f'{int(GT[k, 5])}-'
            if viewImages:        
                plt.title(titre,fontsize=30)

        # Pseudo programme de reconnaissance, à remplacer par votre code.
        # Ne fait que copier des données de fichiers .mat
        # bd stocke les signes reconnus sans l'image
        # les concatène à BD
        # BD sera stocké dans un fichier .mat
        # Il sera porté en entrée de la fonction d'évaluation quantitative
        if type_ == 'Test':
            X = sio.loadmat('X.mat')['BD']  
        else:
            X = sio.loadmat('Y.mat')['BD']  
            
        ind = np.where(X[:, 0] == n_val)[0]
        bd  = X[ind,:]
       
            
        if viewImages:    
            plt.subplot(1,2,2)
            plt.imshow(im)
            titre = 'MyProg: '
            for k in range(bd.shape[0]):
                draw_rectangle( int(bd[k, 3]), int(bd[k, 4]), int(bd[k, 1]), int(bd[k, 2]), 'm')
                titre += f'{int(bd[k, 5])}-'
            plt.title(titre,fontsize=30)
            plt.show()
            plt.pause(0.1)

        BD.extend(bd.tolist())

    file_out = 'myResults.mat'
    sio.savemat(file_out, {'BD': np.array(BD)})

    return file_out, resize_factor

# metro2025('Learn',viewImages=1)  # Pour travailler sur la base d'appentissage
metro2025('Test',viewImages=0)  # Pour travailler sur la base de test