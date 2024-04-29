import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import math

import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt

breedte_resultaat = 1480
hoogte_resultaat = 1000

# Transformeer de afbeelding zodat de positie van de gedetecteerde markers wordt toegewezen aan (10, 10) en (1480, 1000)
def transformeer_afbeelding(afbeeldingsmatrix, markerpunten):
    # Definieer de doelposities van de markers.
    doelposities = np.array([[0, 0], [0, hoogte_resultaat], [breedte_resultaat, 0], [breedte_resultaat, hoogte_resultaat]], dtype=np.float32)

    # Compute the perspective transformation matrix
    transformatiematrix = cv2.getPerspectiveTransform(markerpunten, doelposities)

    # Apply the perspective transformation to the image
    getransformeerde_afbeelding = cv2.warpPerspective(afbeeldingsmatrix, transformatiematrix, (breedte_resultaat, hoogte_resultaat))

    return getransformeerde_afbeelding

# Cut square images of size 240x240 with an offset of 20 pixels from the top and left edges of the transformed image
def knip_afbeeldingen_uit_matrix(afbeeldingsmatrix, formaat, offset, crop_marge=10):
    afbeeldingen = []
    for i in range(offset, afbeeldingsmatrix.shape[1] - offset, formaat):
        for j in range(offset, afbeeldingsmatrix.shape[0] - offset - 1, formaat):
            knip = afbeeldingsmatrix[j + crop_marge:j + formaat - crop_marge, i + crop_marge:i + formaat - crop_marge]
            # Turn the image into a binary image
            _, knip = cv2.threshold(knip, 127, 255, cv2.THRESH_BINARY)
            # Convert to 1d grayscale image
            knip = cv2.cvtColor(knip, cv2.COLOR_BGR2GRAY)
            afbeeldingen.append(knip)
    return afbeeldingen


def emoticons_inladen(bestandsnaam_afbeelding_raster, emotie):
    # Laadt de rastarafbeelding in.
    raster = Image.open(bestandsnaam_afbeelding_raster)
    # Zet de afbeelding om naar een numpy tensor.
    raster_array = np.array(raster)
    
    # Detecteer de markers in de afbeelding.
    acuro_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    markers, ids, rejectedImgPoints = aruco.detectMarkers(raster_array, acuro_dict)
    id_lijst = list(ids)
    
    # Check of allevier de markers zijn gevonden.
    if len(id_lijst) != 4:
        print("Niet alle markers zijn gevonden.")
        print("Probeer een nieuwe foto te maken van je raster waarop de markers duidelijk zichtbaar zijn en in focus zijn.")
        return [], []
    
    # Transformeer de afbeelding.
    markerpunten = np.array([markers[id_lijst.index(0)][0][2], markers[id_lijst.index(1)][0][1], markers[id_lijst.index(2)][0][3], markers[id_lijst.index(3)][0][0]], dtype=np.float32)
    getransformeerde_afbeelding = transformeer_afbeelding(raster_array, markerpunten)
    
    # Knip de afbeeldingen uit.
    afbeeldingen = knip_afbeeldingen_uit_matrix(getransformeerde_afbeelding, 240, 20)
    # Maak labels voor de afbeeldingen.
    labels = np.array([emotie] * len(afbeeldingen))
    
    return afbeeldingen, labels

def toon_afbeeldingen(afbeeldingen, labels, max_afbeeldingen=6):
    cols = 6
    empty_afbeelding = np.ones(afbeeldingen[0].shape, dtype=np.uint8)*255
    max_afbeeldingen = min(len(afbeeldingen), max_afbeeldingen)
    rows = math.ceil(max_afbeeldingen/cols)
    fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 5*rows), squeeze=False)
    for i, ax_row in enumerate(axs):
        for j, ax in enumerate(ax_row):
            if i*cols + j >= max_afbeeldingen:
                ax.imshow(empty_afbeelding, cmap='gray', vmin=0, vmax=255)
                ax.axis('off')
            else:
                ax.imshow(afbeeldingen[i*cols + j], cmap='gray', vmin=0, vmax=255)
                ax.axis('off')
                ax.set_title(labels[i*cols + j], fontsize=25)
                
    plt.show()
    
    
    
    
def laadt_bestanden_in_map_met_label(path, label):
    afbeeldingen = []
    labels = []
    for bestandsnaam in os.listdir(path):
        if bestandsnaam.endswith('.png') or bestandsnaam.endswith('.jpg'):
            afbeeldingen_emoticon, labels_emoticon = emoticons_inladen(os.path.join(path, bestandsnaam), label)
            afbeeldingen.extend(afbeeldingen_emoticon)
            labels.extend(labels_emoticon)
    return afbeeldingen, labels


def one_hot_encode_labels(labels, klasses):
    labels_one_hot = np.zeros((len(labels), len(klasses)))
    for i, label in enumerate(labels):
        labels_one_hot[i, klasses.index(label)] = 1
    return labels_one_hot

def create_confusion_matrix_for_one_hot_encoded_labels(true_labels, predicted_labels, class_names):
    # Create a confusion matrix
    confusion_matrix = np.zeros((len(class_names), len(class_names)))
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        confusion_matrix[np.argmax(true_label), np.argmax(predicted_label)] += 1
    return confusion_matrix

def visualize_confussion_matrix_in_heatmap(confusion_matrix, class_names):
    maxvalue = np.max(confusion_matrix)
    fig, ax = plt.subplots()
    cax = ax.matshow(confusion_matrix, cmap='Greens', vmin=0, vmax=maxvalue)

    for (i, j), val in np.ndenumerate(confusion_matrix):
        ax.text(j, i, f'{int(val)}', ha='center', va='center', fontsize=15, color='black')

    fig.colorbar(cax)
    ax.set_xticklabels([''] + class_names, fontsize=15)
    ax.set_yticklabels([''] + class_names, fontsize=15)
    ax.set_xlabel('Voorspelde labels', fontsize=15)
    ax.set_ylabel('Werkelijke labels', fontsize=15)
    plt.show()
    
    
    
    