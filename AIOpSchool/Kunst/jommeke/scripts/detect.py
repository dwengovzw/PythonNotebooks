from __future__ import print_function
import torch

from scripts.unet.unet_model import UNet
from tqdm import tqdm
import os

import numpy as np

from collections import defaultdict
from PIL import Image
from skimage import color, io
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

import scripts.data as data
import scripts.utils as utils

max_dim = 700

no_cuda = False
use_cuda = torch.cuda.is_available() and not no_cuda
device = torch.device('cuda' if use_cuda else 'cpu')

def afbeeldingen_naar_dataset(filenames = ["./images/jommeke0.png"]):
    assigned_set = 'validation'
    # Create dict for dataset
    comics = defaultdict(set)
    sets = defaultdict(set)
    # Voeg afbeeldingen toe aan dataset
    for filename in filenames:
        sets[assigned_set].add(filename)
        
    sets = dict(sets)
    #data_set = data.ImageDataset(sets[assigned_set],
    #                             transform=transforms.Compose([transforms.CenterCrop([1080*1500/960,1500]),transforms.ToTensor(), utils.RgbToLab()]))
    data_set = data.ImageDataset(sets[assigned_set],
                                 transform=transforms.Compose(
                                     [transforms.Resize(500), transforms.ToTensor(),
                                      utils.RgbToLab()]))

    return data_set


def model_klaarmaken():
    print('running inference on ' + device.__str__())
    model = UNet(1, 3).to(device)
    model.load_state_dict(torch.load('scripts/models/netG-l1-discriminator-total-variation.torch', map_location=device))
    model.eval()
    return model
    
def afbeeldingen_naar_lijntekening(data_set):
    lijntekeningen = []
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=False)
    for original in tqdm(data_loader, total=len(data_loader)):
        line_art = utils.convert_lab_to_line_art(original, [30, 25, 25])
        line_art_rgb = color.gray2rgb(1 - line_art[0, 0])
        a = 255 * line_art_rgb
        a = a.astype(np.uint8)
        im = Image.fromarray(a)
        lijntekeningen.append(im)
    return lijntekeningen
    
    
def kleur_afbeeldingen_inkleuren(model, data_set):
    # maak map voor uitvoer als die nog niet zou bestaan.
    try:
        os.makedirs('output')
    except FileExistsError:
        pass
    
    ingekleurd = []
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=False)
    with torch.no_grad():
        for filename, original in tqdm(zip(data_set.filenames, data_loader), total=len(data_loader)):
            base_name = os.path.splitext(os.path.split(filename)[1])[0]
            if os.path.isfile('output/' + base_name + '-original.png'):
                print('Ignoring file', filename, 'since it already exists')
                continue

            line_art = utils.convert_lab_to_line_art(original, [30, 25, 25])
            #if args.remove_dither:
            #    line_art = utils.remove_dithering(line_art[0, 0])[np.newaxis, np.newaxis]
            line_art = torch.as_tensor(line_art, dtype=torch.float, device=device)
            colored = model(line_art).cpu() if model else None

            colored_rgb = color.lab2rgb(colored[0].numpy().transpose((1, 2, 0)) * [100, 128, 128]) if model is not None else None

            a = np.asarray(colored_rgb)
            a = 255 * a
            a = a.astype(np.uint8)
            im = Image.fromarray(a)
            ingekleurd.append(im)
    return ingekleurd
    
