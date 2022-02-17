from PIL import Image
from sklearn.cluster import MeanShift, estimate_bandwidth
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ipywidgets as widgets
import os
import GPUtil

GPUs = GPUtil.getGPUs()
available_gpu_ids = []
for gpu in GPUs:
    if gpu.memoryFree > 3000:
        available_gpu_ids.append(gpu.id)

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    for id in available_gpu_ids:
        if str(id) not in os.environ["CUDA_VISIBLE_DEVICES"].split(','):
            available_gpu_ids.remove(id)

if not available_gpu_ids:
    print('GPU currently not available, please try again later.')
    #return

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(available_gpu_ids[0])

import tensorflow as tf
import numpy as np
import warnings
import io

reference_model = 'detecting_stomata_model_VGG19FT.h5'
model_dir = 'data/'

upload_widget = widgets.FileUpload(accept='image/*', multiple=False)

image_crop = None

import time, sys
from IPython.display import clear_output

def update_progress(progress):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)

def choose_picture():
    if not upload_widget.value:
        print('No image selected yet.')
        return None

    first_key = next(iter(upload_widget.value))

    if not upload_widget.value[first_key]['metadata']['type'].startswith('image'):
        print('Please select an image.')
        return None

    image = Image.open(io.BytesIO(upload_widget.value[first_key]['content']))
    return image


def show_image():
    image = choose_picture()
    if image is None:
        return

    im_r = np.array(image)

    height, width, depth = im_r.shape
    rect_width = width
    rect_height = height
    if rect_width > 1600:
        rect_width = 1600
    if rect_height > 1200:
        rect_height = 1200

    fig = plt.figure(figsize=(20, 10))
    plt.imshow(im_r)
    rect = patches.Rectangle((0, 0), rect_width, rect_height, linewidth=-1, edgecolor='r', facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)
    plt.close()

    def choose_regio(x, y):
        global image_crop
        image_crop = im_r[y: y + rect_height, x: x + rect_width, :]
        rect.set_xy((x, y))
        display(fig)

    max_x = max(width - 1600, 0)
    max_y = max(height - 1200, 0)
    widgets.interact(choose_regio, x=widgets.IntSlider(min=0, max=max_x, continuous_update=False), y=widgets.IntSlider(min=0, max=max_y, continuous_update=False))


def detect_stomata_subproces(im_r, q):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.per_process_gpu_memory_fraction(gpu, 0.2)
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.compat.v1.Session(config=tf.config)
    K.set_session(sess)

    model_file = load_model(os.path.join(model_dir, reference_model))

    shift = 10
    offset = 60
    bandwidth = offset

    stomata_punten = {}
    for thr in range(5, 100, 5):
        stomata_punten[str(thr)] = []

    no_x_shifts = (np.shape(im_r)[0] - 2 * offset) // shift
    no_y_shifts = (np.shape(im_r)[1] - 2 * offset) // shift

    #confidence = np.zeros((np.shape(im_r)[0],np.shape(im_r)[1]))
    #print('start calculations')
    for x in np.arange(no_x_shifts + 1):
        #update_progress(x / (no_x_shifts + 2.0))
        for y in np.arange(no_y_shifts + 1):
            x_c = x * shift + offset
            y_c = y * shift + offset

            im_r_crop = im_r[x_c - offset:x_c + offset, y_c - offset:y_c + offset, :]
            im_r_crop = im_r_crop.astype('float32')
            im_r_crop /= 255

            y_model = model_file.predict(np.expand_dims(im_r_crop, axis=0))
            #print(y_model[0][1])

            for thr in range(5, 100, 5):
                if y_model[0][1] > thr / 100.:
                    stomata_punten[str(thr)].append([x_c, y_c])

            #for i in np.arange(2*offset):
            #    for j in np.arange(2*offset):
            #        if y_model[0][1] > confidence[x*shift+i][y*shift+j]:
            #            confidence[x*shift+i][y*shift+j] = y_model[0][1]

    for thr in range(5, 100, 5):
        if stomata_punten[str(thr)]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                #print('thr = '+str(thr))
                ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                ms.fit(stomata_punten[str(thr)])
                stomata_punten[str(thr)] = [[x[1], x[0]] for x in ms.cluster_centers_]  # Because cluster_centers is inverted
    #update_progress(1)
    #print(stomata_punten)
    #print('Im done calculations')
    q.put(stomata_punten)
    #print('Im done putting')
    #q.put(confidence)


def detect_stomata_in_image():
    global image_crop
    if image_crop is None:
        return
     
    #GPUs = GPUtil.getGPUs()
    #available_gpu_ids = []
    #for gpu in GPUs:
    #    if gpu.memoryFree > 5000:
    #        available_gpu_ids.append(gpu.id)

    #if 'CUDA_VISIBLE_DEVICES' in os.environ:
    #    for id in available_gpu_ids:
    #        if str(id) not in os.environ["CUDA_VISIBLE_DEVICES"].split(','):
    #            available_gpu_ids.remove(id)

    #if not available_gpu_ids:
    #    print('GPU currently not available, please try again later.')
    #    return
    #print('Available GPUs:')
    #print(available_gpu_ids)

    # We voeren dit uit in een appart proces omdat de gpu memory dan wordt vrijgegeven
    #gpus = tf.test.gpu_device_name()
    #with tf.device(gpus[available_gpu_ids[0]].name):
    q = Queue(maxsize=-1)
    p = Process(target=detect_stomata_subproces, args=(image_crop, q))
    #print('start')
    from time import sleep
    sleep(1)
    p.start()
    #print('join')
    #p.join()
    sleep(1)
    #print('get')
    stomata_punten = q.get()
    #print('join')
    sleep(1)
    p.join()
    #confidence = q.get()
    #print('plot')

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.imshow(image_crop)
    #ax.imshow(confidence, alpha=0.3, cmap='viridis')
    points_im, = ax.plot([], [], 'xr', alpha=0.75, markeredgewidth=3, markersize=12)
    plt.close()
    #print('plotted')

    def change_threshold(thr=0.5):
        x_points = [x[0] for x in stomata_punten[str(int(thr * 100))]]
        y_points = [x[1] for x in stomata_punten[str(int(thr * 100))]]
        points_im.set_xdata(x_points)
        points_im.set_ydata(y_points)
        display(fig)
        print('Number of detected stomata: ' + str(len(stomata_punten[str(int(thr * 100))])))

    widgets.interact(change_threshold, thr=widgets.FloatSlider(value=0.7, min=0.05, max=0.99, step=0.05, continuous_update=False))
