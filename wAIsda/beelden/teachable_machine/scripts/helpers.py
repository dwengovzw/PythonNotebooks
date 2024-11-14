import os
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet import decode_predictions

def laadt_bestanden_in_map_met_label(path, label):
    afbeeldingen = []
    labels = []
    for bestandsnaam in os.listdir(path):
        if bestandsnaam.endswith('.png') or bestandsnaam.endswith('.jpg') or bestandsnaam.endswith('.jpeg'):
            afbeelding = Image.open(os.path.join(path, bestandsnaam))
            # Resize the smallest side of the image to image_size pixels.
            image_size = 224
            if afbeelding.width < afbeelding.height:
                afbeelding = afbeelding.resize((image_size, int(image_size * afbeelding.height / afbeelding.width)))
            else:
                afbeelding = afbeelding.resize((int(image_size * afbeelding.width / afbeelding.height), image_size))
            # Crop the center of the image.
            afbeelding = afbeelding.crop((afbeelding.width//2 - image_size//2, afbeelding.height//2 - image_size//2, afbeelding.width//2 + image_size//2, afbeelding.height//2 + image_size//2))
            # Convert the image to a numpy array.
            afbeelding = np.array(afbeelding)
            afbeeldingen.append(afbeelding)
            
    labels = np.array([label] * len(afbeeldingen))
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
                ax.imshow(empty_afbeelding)
                ax.axis('off')
            else:
                ax.imshow(afbeeldingen[i*cols + j])
                ax.axis('off')
                ax.set_title(labels[i*cols + j], fontsize=25)
                
    plt.show()
    
    
def one_hot_encode_labels(labels, klasses):
    labels_one_hot = np.zeros((len(labels), len(klasses)))
    for i, label in enumerate(labels):
        labels_one_hot[i, klasses.index(label)] = 1
    return labels_one_hot


def druk_imagenet_labels_af():
    # Initialize a dummy prediction array for 1000 classes (ImageNet classes)
    dummy_preds = np.array([[0]*1000])
    # Decode predictions without running the model (only for extracting labels)
    decoded_labels = decode_predictions(dummy_preds, top=1000)[0]

    # Extract and print labels
    imagenet_labels = [label for (imagenet_id, label, _) in decoded_labels]
    print(imagenet_labels)