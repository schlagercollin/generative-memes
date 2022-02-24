"""
Torch Datasets for Baseline

MemeCaptionDataset ==> for image-conditioned meme caption generation
MemeTemplateDataset ==> for image generation
"""

import numpy as np
import os
import glob
import json

import torchvision.transforms.functional as TF
from torchvision.transforms.autoaugment import AutoAugmentPolicy 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchtext import vocab as Vocab
import torch

from PIL import Image

import settings

DATASET_PATH = 'scraper/dataset/'
IMG_SIZE = (settings.img_size, settings.img_size)

# build Pytorch generator dataset
class MemeCaptionDataset(Dataset):
    def __init__(self, max_seq_length=15, transform=None):
        """Dataset for the image-conditioned caption generation.

        Args:
            max_seq_length (int, optional): max sequence length. Defaults to 15.
            transform (torchvision.Transform, optional): transformations to apply to images.
            Defaults to None, which corresponds to AutoAugment and resize to settings.img_size.
        """
        
        if transform is None:
            # default transform
            transform = transforms.Compose([
                transforms.ConvertImageDtype(torch.uint8),
                transforms.AutoAugment(),
                transforms.Resize(IMG_SIZE),
                transforms.ConvertImageDtype(torch.float),
            ])
            
        self.transform = transform
        
        # cache dir for the template
        self.cache_dir = f"{DATASET_PATH}/template_image_cache"

        # load templates
        self.templates = [os.path.basename(template) for template in glob.glob(f"{DATASET_PATH}/memes/*.json")]

        # download the meme templates
        self.download_meme_templates()

        # save the different text outputs
        self.memes = []
        
        # load the templates to memory
        self.images = {}

        for template in self.templates:
            
            with Image.open(f"{self.cache_dir}/{template}.jpg") as template_image:
                img = TF.to_tensor(template_image)
                self.images[template] = img
            
            text_snippets = []
            for meme in json.load(open(f"{DATASET_PATH}/memes/{template}")):
                self.memes.append((template, " ".join(meme['boxes'])))  # (template, caption)

        def yield_strings():
            for template, caption in self.memes:
                yield caption.strip().split()

        self.vocab = Vocab.build_vocab_from_iterator(
            yield_strings(),
            min_freq=10
        )

        self.itos = self.vocab.get_itos()
        self.stoi = self.vocab.get_stoi()

        self.unk = "<UNK>"
        self.itos.append(self.unk)
        self.stoi[self.unk] = len(self.stoi)

        self.max_seq_length = max_seq_length

    def download_meme_templates(self):

        # ensure that the cache directory has been created
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # ensure that each template is cached within the cache directory
        for template in self.templates:
            if not os.path.exists(f"{self.cache_dir}/{template}.jpg"):
                # extract relevant URL
                url = json.load(open(f"{DATASET_PATH}/templates/{template}"))['template_url']

                # download the specific meme
                os.system(f"wget -O {self.cache_dir}/{template}.jpg {url}")

    def __getitem__(self, index):
        """
        Returns:
            img: image tensor
            caption_vector: numeric representation of text caption
        """
        
        template_name, caption = self.memes[index]
        
        img = self.images[template_name]
                    
        if self.transform:
            img = self.transform(img)

        caption_vector = np.zeros((self.max_seq_length))
        caption = caption.strip().split()

        for i, word in enumerate(caption):
            if i >= self.max_seq_length:
                break
            caption_vector[i] = self.stoi[word] if word in self.stoi else self.stoi[self.unk]

        return img, caption_vector.astype(np.long)
    
    def __len__(self):
        return len(self.memes)
    

class MemeTemplateDataset(Dataset):

    def __init__(self, transform=None, epoch_multiplier=100):
        """

        Args:
            transform (torchvision.Transform, optional): transformations to apply to images.
            Defaults to None, which corresponds to AutoAugment and resize to settings.img_size.
            epoch_multiplier (int, optional): a mini-hack that artifically increases the size
            of an epoch (to modulate how frequently `torchgan` does validation) without modifying
            it's code, since it doesn't appear to have that functionality out of the box.
            Defaults to 100, which means that the dataset epoch is repeated 100 times.
        """
        
        if transform is None:
            # default transform
            transform = transforms.Compose([
                transforms.ConvertImageDtype(torch.uint8),
                transforms.AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
                transforms.Resize(IMG_SIZE),
                transforms.ConvertImageDtype(torch.float),
            ])
            
        self.transform = transform
        
        # artificially increase the size of an epoch
        self.epoch_multiplier = epoch_multiplier
        
        # cache dir for the template
        self.cache_dir = f"{DATASET_PATH}/template_image_cache"

        # load templates
        self.templates = [os.path.basename(template) for template in glob.glob(f"{DATASET_PATH}/memes/*.json")]

        # download the meme templates
        self.download_meme_templates()
        
        # load the templates to memory
        self.images = []

        for template in self.templates:
            with Image.open(f"{self.cache_dir}/{template}.jpg") as template_image:
                img = TF.to_tensor(template_image)
                self.images.append(img)
                

    def download_meme_templates(self):

        # ensure that the cache directory has been created
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # ensure that each template is cached within the cache directory
        for template in self.templates:
            if not os.path.exists(f"{self.cache_dir}/{template}.jpg"):
                # extract relevant URL
                url = json.load(open(f"{DATASET_PATH}/templates/{template}"))['template_url']

                # download the specific meme
                os.system(f"wget -O {self.cache_dir}/{template}.jpg {url}")

    def __getitem__(self, index):
        """
        Returns:
            img: image tensor
        """
        
        # map dataset index to actual image index
        index = index % len(self.images)
        
        img = self.images[index]
                    
        if self.transform:
            img = self.transform(img)
            
        return img
    
    def __len__(self):
        return len(self.images) * self.epoch_multiplier
    

class SimpleMemeTemplateDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        
        self.image_dir = image_dir
        self.image_paths = [x for x in os.listdir(image_dir) if ".jpg" in x]
        
        if transform is None:
            # default transform
            self.transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.Resize(IMG_SIZE),
                transforms.ConvertImageDtype(torch.float),
            ])
        else:
            self.transform = transform
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        x = Image.open(self.image_dir + "/" + image_path)
        if self.transform is not None:
            x = self.transform(x)
        return x
    
    def __len__(self):
        return len(self.image_paths)


if __name__ == "__main__":
    caption_dataset = MemeCaptionDataset()
    print(caption_dataset.itos[-5:])
    print(caption_dataset.stoi[caption_dataset.unk])
    
    