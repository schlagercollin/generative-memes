"""
Torch Datasets for Baseline

MemeCaptionDataset ==> for image-conditioned meme caption generation
MemeTemplateDataset ==> for image generation
"""

import numpy as np
import os
import glob
import json
import re

import torchvision.transforms.functional as TF
from torchvision.transforms.autoaugment import AutoAugmentPolicy 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchtext import vocab as Vocab
import torch
import pickle as pkl
import os

from PIL import Image

import settings

DATASET_PATH = 'scraper/dataset/'
DATASET_CACHE_LOC = './dataset_cache/dataset.db'
IMG_SIZE = (settings.img_size, settings.img_size)

def get_meme_caption_dataset(
    max_seq_length=25,
    inclusion_threshold=5,
    max_unk_per_caption=2,
    transform=None
):
    """
    Returns a dataset of image-conditioned caption generation, with disk caching.
    """

    def gen_new_dataset():
        return MemeCaptionDataset(
            max_seq_length=max_seq_length,
            inclusion_threshold=inclusion_threshold,
            max_unk_per_caption=max_unk_per_caption,
            transform=transform
        )

    if os.path.exists(DATASET_CACHE_LOC):
        with open(DATASET_CACHE_LOC, 'rb') as f:
            dataset = pkl.load(f)

        if dataset.max_seq_length != max_seq_length or dataset.inclusion_threshold != inclusion_threshold or dataset.max_unk_per_caption != max_unk_per_caption:
            dataset = gen_new_dataset()

    else:
        dataset = gen_new_dataset()

    with open(DATASET_CACHE_LOC, 'wb') as f:
        pkl.dump(dataset, f)

    return dataset

# build Pytorch generator dataset
class MemeCaptionDataset(Dataset):
    def __init__(
        self,
        max_seq_length=25,
        inclusion_threshold=5,
        max_unk_per_caption=2,
        transform=None):
        """Dataset for the image-conditioned caption generation.

        Args:
            max_seq_length (int, optional): max sequence length. Defaults to 15.
            inclusion_threshold (int, optional): minimum number of occurences in dataset for word to be included in vocab. Defaults to 10.
            transform (torchvision.Transform, optional): transformations to apply to images.
            Defaults to None, which corresponds to AutoAugment and resize to settings.img_size.
        """

        self.max_seq_length = max_seq_length
        self.max_unk_per_caption = max_unk_per_caption
        self.inclusion_threshold = inclusion_threshold
        
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
            
            for meme in json.load(open(f"{DATASET_PATH}/memes/{template}")):
                self.memes.append((template, " ".join(meme['boxes'])))  # (template, caption)

        def yield_strings():
            for template, caption in self.memes:
                yield self.clean_caption(caption)

        self.vocab = Vocab.build_vocab_from_iterator(
            yield_strings(),
            min_freq=inclusion_threshold
        )

        self.itos = self.vocab.get_itos()
        self.stoi = self.vocab.get_stoi()

        self.unk = "<UNK>"
        self.itos.append(self.unk)
        self.stoi[self.unk] = len(self.stoi)

        self.start = "<START>"
        self.itos.append(self.start)
        self.stoi[self.start] = len(self.stoi)

        self.end = "<END>"
        self.itos.append(self.end)
        self.stoi[self.end] = len(self.stoi)

        # prune memes from the dataset that have too many <unk> instances

        pruned_memes = []
        num_pruned = 0
        for meme in self.memes:
            # tokenise using the vocab
            caption = meme[1]
            caption_vector = self.tokenize_meme_caption(caption)

            # check if the caption has too many <unk> instances
            if np.sum(caption_vector == self.stoi[self.unk]) <= self.max_unk_per_caption:
                pruned_memes.append(
                    (meme[0], caption_vector)
                )
            else:
                num_pruned += 1

        self.memes = pruned_memes
        print(f"Pruned {num_pruned} memes from the dataset due to excess <unk> occurences")
        print(f"We now have {len(self.memes)} memes remaining in the dataset")
        print(f"The vocabulary size is of size {len(self.itos)}")

        # for meme_to_print in range(10):
        #     print(" ".join([self.itos[idx] for idx in self.memes[meme_to_print][1]]))

    def tokenized_to_list_sentence(self, caption_token_vec):
        return [self.itos[idx] for idx in caption_token_vec]

    def clean_caption(self, caption):
        lower_caption = caption.lower()
        no_punc = re.sub(r'[^\w\s]','', lower_caption)
        # print(no_punc.strip().split())
        return no_punc.strip().split()

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

    def tokenize_meme_caption(self, caption):
        """Tokenize meme caption.

        Args:
            caption (str): caption to tokenize.

        Returns:
            caption_vector (np.array): tokenized caption.
        """

        caption_vector = np.zeros((self.max_seq_length))
        caption = self.clean_caption(caption)
        caption = [self.start] + caption + [self.end]

        for i, word in enumerate(caption):
            if i >= self.max_seq_length:
                break
            caption_vector[i] = self.stoi[word] if word in self.stoi else self.stoi[self.unk]

        return caption_vector.astype(np.int64)

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

        # this is returning a vector of sequence_length integers; we either need to put this through a one-hot encoding, or take glove embeddings

        # for now, let's just one-hot encode it...
        
        return img, caption
    
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
    tokenized = caption_dataset.tokenize_meme_caption("i am a meme")
    print(tokenized)
    print(" ".join([caption_dataset.itos[idx] for idx in tokenized]))

    # print(caption_dataset.memes)
    # print(caption_dataset.itos[-5:])
    # print(caption_dataset.stoi[caption_dataset.unk])
    
    