import ipywidgets as widgets
import io
import numpy as np
from PIL import Image

upload_widget = widgets.FileUpload(
    accept=".jpg",
    multiple=False  # We laden slechts één bestand op
)

def save_npy():
    if not upload_widget.value:
        print('No npy file selected yet.')
        return None

    first_key = next(iter(upload_widget.value))
    print(first_key.content)
    print(upload_widget.value)

    image = Image.open(io.BytesIO(first_key.content))
    eigen_afbeelding = np.array(image)
    np.save('./images/eigen_afbeelding.npy', eigen_afbeelding)
