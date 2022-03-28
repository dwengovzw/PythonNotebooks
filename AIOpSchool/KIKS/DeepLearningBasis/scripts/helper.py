import ipywidgets as widgets
import io
import numpy as np

upload_widget = widgets.FileUpload(
    accept=".npy",
    multiple=False  # We laden slechts één bestand op
)

def save_npy():
    if not upload_widget.value:
        print('No npy file selected yet.')
        return None

    first_key = next(iter(upload_widget.value))

    eigen_npy = io.BytesIO(upload_widget.value[first_key]['content'])
    eigen_afbeelding = np.load(eigen_npy)
    np.save('./images/eigen_afbeelding.npy', eigen_afbeelding)