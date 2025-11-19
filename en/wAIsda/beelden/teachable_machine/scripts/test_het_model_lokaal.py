import subprocess
import sys

# List of required packages
required_packages = ["opencv-python", "numpy", "tensorflow"]

# Install required packages
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import cv2
import numpy as np
import tensorflow as tf

# Load the model
uitgebreid_model = tf.keras.models.load_model("model.h5")

img_size = (224, 224)

# Initialize video capture
video = cv2.VideoCapture(0)

try:
    while True:
        # Read a frame from the webcam
        ret, frame = video.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # If your camera reverses the image

        # Preprocess the frame: resize and normalize
        img = cv2.resize(frame, img_size)
        img_array = np.expand_dims(img / 255.0, axis=0)  # Normalize and add batch dimension

        # Predict the class
        prediction = uitgebreid_model.predict(img_array)
        predicted_class = "PMD" if np.argmax(prediction) == 0 else "Papier"

        # Display the prediction on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Voorspelling: {predicted_class}, Verdeling: (PMD: {prediction[0][0]:.4f}, Papier: {prediction[0][1]:.4f})"
        cv2.putText(frame, text, (10, 30), font, 0.6, (255, 0, 0), 2)

        # Show the frame in a window
        cv2.imshow("Live Camera Feed", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    # Release resources
    video.release()
    cv2.destroyAllWindows()
