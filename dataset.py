import torch # :)
import numpy as np
import os
import glob
import json
from torch.utils.data import Dataset

DATASET_PATH = 'scraper/dataset/'

# build Pytorch generator dataset
class MemeDataset(Dataset):

    def __init__(self):
        self.current_template = 0
        self.current_meme = 0

        # load templates
        self.templates = [os.path.basename(template) for template in glob.glob(f"{DATASET_PATH}/memes/*.json")]

        # download the meme templates
        self.download_meme_templates()

        # save the different text outputs
        self.memes = {}

        for template in self.templates:
            text_snippets = []
            for meme in json.load(open(f"{DATASET_PATH}/memes/{template}")):
                text_snippets.append(" ".join(meme['boxes']))
            self.memes[template] = text_snippets

    def download_meme_templates(self):
        cache_dir = f"{DATASET_PATH}/template_image_cache"

        # ensure that the cache directory has been created
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # ensure that each template is cached within the cache directory
        for template in self.templates:
            if not os.path.exists(f"{cache_dir}/{template}.jpg"):
                # extract relevant URL
                url = json.load(open(f"{DATASET_PATH}/templates/{template}"))['template_url']

                # download the specific meme
                os.system(f"wget -O {cache_dir}/{template}.jpg {url}")

    def __getitem__(self, index):
        pass
        return template, caption


print("Hello, world!")
dataset = MemeDataset()