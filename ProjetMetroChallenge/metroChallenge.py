# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 09:08:43 2025

@author: frossant
"""
import numpy as np
import os
import re
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.io as sio
from evaluationV2 import evaluation,compareTestandRef


from  myMetroProcessing import processOneMetroImage, draw_rectangle

# Define directories ==========================================================

challengeDirectory  = "BD_CHALLENGE"
file_out            = 'teamsNN.mat'  # your results

# Define your resizing parameter===============================================
resize_factor = 1


# Read the challenge directory ================================================

image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
pattern = re.compile(r'\((\d+)\)')
numbered_images = []

for filename in os.listdir(challengeDirectory):
    if filename.lower().endswith(image_extensions):
        match = pattern.search(filename)
        if match:
            number = int(match.group(1))
            numbered_images.append((number, filename))

numbered_images.sort()
imageFilesList = [filename for filename in numbered_images]

    
# Process all images and save the results in a mat file =======================

num_images = len(imageFilesList)
BD = []

for n_val in range(num_images):
    
    # LOAD IMAGE ------------------------------
    nom = imageFilesList[n_val][1]
    print(f"----- {nom} -----")
    im_path = os.path.join(challengeDirectory, nom)
    im = np.array(Image.open(im_path).convert('RGB')) / 255.0
    
    
    # fig = plt.figure(figsize=(45,15))
    # plt.imshow(im)   
    # plt.title(f'{imageFilesList[n_val]}')
    
         
    # PROCESS IMAGE --------------------------- 
    im_resized,bd = processOneMetroImage(nom,im,imageFilesList[n_val][0],resize_factor)
    
    
    # ADD TO GLOBAL RECOGNITION RESULTS -------
    BD.extend(bd.tolist())

# Save recognition
sio.savemat(file_out, {'BD': np.array(BD)})


# # Check reference recognition =================================================
# # compareTestandRef(imageFilesList,challengeDirectory,'GTCHALLENGETEST.mat', file_out, resize_factor)
  
    
# Quantitatve evaluation ======================================================
evaluation('GTCHALLENGETEST.mat', file_out, resize_factor)  # it cannot work without the generated file of random recognition
                                                            
# test on the example teamsEX.mat
evaluation('GTCHALLENGETEST.mat', 'teamsEX.mat', 0.5) 
