import os
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet import decode_predictions
import tensorflow as tf

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Limit the GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

    # Set a memory limit (e.g., 2 GB)
    tf.config.experimental.set_virtual_device_configuration(
        device,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)]
    )
    
# Supress warnings
tf.get_logger().setLevel('ERROR')

# Suppress CUDA and other backend-related messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import warnings
warnings.filterwarnings('ignore')

def laadt_bestanden_in_map_met_label(path, label):
    afbeeldingen = []
    labels = []
    for bestandsnaam in os.listdir(path):
        if bestandsnaam.endswith('.png') or bestandsnaam.endswith('.jpg') or bestandsnaam.endswith('.jpeg'):
            afbeelding = Image.open(os.path.join(path, bestandsnaam))
            # Resize the smallest side of the image to image_size pixels.
            afbeelding = resize_afbeelding(afbeelding)
            afbeeldingen.append(afbeelding)
            
    labels = np.array([label] * len(afbeeldingen))
    return afbeeldingen, labels

def resize_afbeelding(afbeelding, image_size=224):
    image_size = 224
    if afbeelding.width < afbeelding.height:
        afbeelding = afbeelding.resize((image_size, int(image_size * afbeelding.height / afbeelding.width)))
    else:
        afbeelding = afbeelding.resize((int(image_size * afbeelding.width / afbeelding.height), image_size))
    # Crop the center of the image.
    afbeelding = afbeelding.crop((afbeelding.width//2 - image_size//2, afbeelding.height//2 - image_size//2, afbeelding.width//2 + image_size//2, afbeelding.height//2 + image_size//2))
    # Convert the image to a numpy array.
    afbeelding = np.array(afbeelding)
    return afbeelding



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
        
    
import urllib.parse
from IPython.display import HTML

def maak_jupyterhub_download_link(file_path, link_text="Download"):
    """
    Create a download link for a file in a JupyterHub environment.
    """
    # Extract the file name from the path
    file_name = file_path.split("/")[-1]
    
    # Encode the file path for URL safety
    encoded_file_path = urllib.parse.quote(file_path)
    
    # Construct the download URL
    # Adjust the base URL as needed to match your JupyterHub's configuration
    base_url = "/user-redirect/"  # Default user-redirect path in JupyterHub
    download_url = f"{base_url}files/{encoded_file_path}"
    
    # Return an HTML download link
    html = f"""
    <a href="{download_url}" target="_blank" download="{file_name}">
        {link_text}
    </a>
    """
    return HTML(html)


