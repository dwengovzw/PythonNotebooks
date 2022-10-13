import cv2
import numpy as np
import matplotlib.pylab as plt
import os
import sys

#methode om een afbeelding weer te geven
def plt_imshow(title, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title(title)
    plt.grid(False)
    plt.show()

CONFIDENCE = 0.5
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# De files oproepen die het yolo netwerk nodig zal hebben
config_path = "yolov3.cfg"
weights_path = "kiks.ilabt.imec.be/files/yolov3.weights"

# alle klasselabels (objecten) inladen
labels = open("coco.names").read().strip().split("\n")
# een kleur genereren voor ieder object om later te gebruiken op de figuur
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
# inladen van Yolo
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)


#afbeelding inladen, waarop de objecten gedetecteerd zullen worden
#path_name = "images/hond-cake.jpg"
path_name = sys.argv[1]
image = cv2.imread(path_name)
file_name = os.path.basename(path_name)
filename, ext = file_name.split(".")


h, w = image.shape[:2]
# creeren van een 4D blob
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
r = blob[0, 0, :, :]
# de blob instellen als input van het netwerk
net.setInput(blob)
# alle namen van de lagen opvragen
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

layer_outputs = net.forward(ln)
font_scale = 1
thickness = 1
boxes, confidences, class_ids = [], [], []
# alle layeroutputs overlopen
for output in layer_outputs:
    # alle objectdetecties overlopen
    for detection in output:
        # het classid en de waarschijnelijkheid van de detectie abstraheren
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        #negeer zwakke voorspellingen
        #door ervoor te zorgen dat de gedetecteerde waarschijnlijkheid groter is dan de minimale waarschijnlijkheid
        if confidence > CONFIDENCE:
            #schaal de coördinaten van het selectiekader terug ten opzichte van de grootte van de afbeelding
            #rekening houdend met het feit dat YOLO de middelste (x, y)-coördinaten van het selectiekader teruggeeft
            #gevolgd door de breedte en hoogte van de kaders
            box = detection[:4] * np.array([w, h, w, h])
            (centerX, centerY, width, height) = box.astype("int")
            #gebruik de middelste (x, y)-coördinaten om de top af te leiden
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            # update de lijst van de box coordinaten, waarschijnlijkhgeden en classids
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)
# zorg ervoor dat er ten minste één detectie bestaat
if len(idxs) > 0:
    # overloop de indexes die we behouden
    for i in idxs.flatten():
        # extract the bounding box coordinates
        x, y = boxes[i][0], boxes[i][1]
        w, h = boxes[i][2], boxes[i][3]
        # teken een box en een label op de afbeelding
        color = [int(c) for c in colors[class_ids[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 5)
        text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
        # berekenen van de texthoogte en breedte
        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
        text_offset_x = x
        text_offset_y = y - 5
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
        overlay = image.copy()
        cv2.rectangle(image, box_coords[0], box_coords[1], color, -1)
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
        # label het gedetecteerde object en de waarschijnlijkheid
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            1, [0,0,0], 2)

plt_imshow(filename + "_yolo3." + ext, image)