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
    
    
import cv2
import matplotlib.pyplot as plt
import threading
import time
import ipywidgets as widgets
from IPython.display import display, clear_output

# Global variables to control the live feed and store the captured frame
running = True
captured_frame = None  
nn_model = None

# Function to display the live webcam feed
def live_feed():
    global captured_frame, running

    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while running:
        ret, frame = cap.read()
        if ret:
            # create PIL image 
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Resize the image
            pil_image = pil_image.resize((224, 224))
            # predict the class of the image
            prediction = nn_model.predict(np.expand_dims(np.array(pil_image), axis=0))
            # Get the predicted class
            mapped_labels_predicted = ["PMD" if np.argmax(label) == 0 else "Papier" for label in prediction]
            time.sleep(1)
            print(mapped_labels_predicted)
            
            captured_frame = frame
            # Convert frame to RGB for matplotlib
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            clear_output(wait=True)  # Clear previous output
            plt.imshow(frame_rgb)
            plt.axis('off')
            plt.show()
        else:
            print("Failed to grab frame")
            break
        
        time.sleep(1)  # Add a small delay for smoother updates

    cap.release()

# Function to capture an image when the button is clicked
def capture_image(button):
    global captured_frame
    if captured_frame is not None:
        # Save the captured frame
        cv2.imwrite("captured_image.jpg", captured_frame)
        print("Image captured and saved as 'captured_image.jpg'.")
        # display the captured image
        clear_output(wait=True)
        plt.imshow(cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    else:
        print("No frame available to capture.")
        
thread = None
        
def start_video_stream(model):
    global captured_frame, running, thread, nn_model
    nn_model = model
    # Start the live feed in a separate thread
    thread = threading.Thread(target=live_feed)
    thread.start()

    # Stop the live feed after 30 seconds (or when you manually interrupt)
    time.sleep(15)
    running = False
    thread.join()  # Wait for the live feed thread to finish
    print("Live feed stopped.")
    
 
