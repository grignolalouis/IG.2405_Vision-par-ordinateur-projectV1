# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 09:41:21 2025

@author: fross
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import skimage as ski
import scipy.io


def processOneMetroImage(nom,im,n,resizeFactor):
    
    if resizeFactor != 1:
        im_resized = ski.transform.resize(im, (int(im.shape[0] * resizeFactor), int(im.shape[1] * resizeFactor)),
                        anti_aliasing=True, preserve_range=True).astype(im.dtype)
    else:
        im_resized = im    
    
    # Put your program here to compute bd 
    # bd  Nx6 with N the number of recognized signs
    # one line for each sign. For example for the ith sign 
    # bd[i,:] = [n x1 x2 y1 y2 clase] with  n ~number of the image in the ()
    
    bd  =  np.array([[ n, 10, 20, 110, 120, 12  ],[ n, 110, 220, 210, 320, 5  ]]) # To replace by the result of your code

    # display
    plt.figure()
    plt.imshow(im_resized)
    for k in range(bd.shape[0]):
        draw_rectangle(bd[k,3], bd[k,4], bd[k,1], bd[k,2], 'g')
    plt.title(f'{nom} - Lines {bd[:,5]}')
    plt.show()
    
    return im_resized, bd




# Additional function =========================================================

def draw_rectangle(x1, x2, y1, y2, color):
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)