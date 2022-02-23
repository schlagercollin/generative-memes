"""
infer_full

Main workhorse function for baseline inference: create_meme.
"""

import torch
from infer_caption import pred_vec_to_text
from PIL import ImageDraw
from PIL import ImageFont
import torchvision
from matplotlib.pyplot import imshow
import numpy as np
import cv2
from PIL import Image

import utils

to_pil = torchvision.transforms.ToPILImage(mode='RGB')

def generate_background(generator, device, noise=None, truncated_normal=False):
    if noise is None:
        if truncated_normal:
            noise = torch.nn.init.trunc_normal_(
                torch.zeros((1, 100)),
                a=-2,
                b=2,
            )
        else:
            noise = generator.sampler(1, device)[0]
    
    generator_out = generator.forward(noise)
    return generator_out

def generate_caption(generator_output, encoder, decoder, data_loader, device, dataset):

    features = encoder(generator_output)

    _, dummy_cap = next(data_loader)
    dummy_cap = dummy_cap[:1].to(device)
    caption_tensor = decoder(features, dummy_cap)
    text_caption = " ".join([word for word in pred_vec_to_text(caption_tensor, dataset)[0] if word != "<UNK>"])
    
    return text_caption



    