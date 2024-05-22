import ipywidgets as widgets
import io
from PIL import Image
import numpy as np

upload_image_widget = widgets.FileUpload(
    accept=".jpg, .jpeg",
    multiple=False  # We laden slechts één bestand op
)

upload_npy_widget = widgets.FileUpload(
    accept=".npy",
    multiple=False  # We laden slechts één bestand op
)

def save_afbeelding():
    if not upload_image_widget.value:
        print('No image selected yet.')
        return None

    first_key = next(iter(upload_image_widget.value))

    image = Image.open(io.BytesIO(first_key['content']))
    image.save('./images/eigen_afbeelding.jpg')
    return image

def save_npy():
    if not upload_npy_widget.value:
        print('No npy file selected yet.')
        return None

    first_key = next(iter(upload_npy_widget.value))

    eigen_npy = io.BytesIO(upload_npy_widget.value[first_key]['content'])
    eigen_afbeelding = np.load(eigen_npy)
    np.save('./data/eigen_afbeelding.npy', eigen_afbeelding)
