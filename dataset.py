import numpy as np
import os
import glob
import json

import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image

DATASET_PATH = 'scraper/dataset/'
IMG_SIZE = (32, 32)

# build Pytorch generator dataset
class MemeDataset(Dataset):

    def __init__(self):
        
        # cache dir for the template
        self.cache_dir = f"{DATASET_PATH}/template_image_cache"

        # load templates
        self.templates = [os.path.basename(template) for template in glob.glob(f"{DATASET_PATH}/memes/*.json")]

        # download the meme templates
        self.download_meme_templates()

        # save the different text outputs
        self.memes = []

        for template in self.templates:
            text_snippets = []
            for meme in json.load(open(f"{DATASET_PATH}/memes/{template}")):
                self.memes.append((template, " ".join(meme['boxes'])))  # (template, caption)

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
        
        template_name, caption = self.memes[index]
        
        with Image.open(f"{self.cache_dir}/{template_name}.jpg") as template_image:
            img = TF.to_tensor(template_image)
            
        img = TF.resize(img, IMG_SIZE)
            
        return img#, caption 
    
    def __len__(self):
        return 20000
        return len(self.memes)

def test_dataset_getitem(idx):
    t1, c1 = dataset.__getitem__(idx)
    img = transforms.ToPILImage()(t1).convert("RGB")
    print(t1.shape)
    print(c1)
    img.show()

def get_dataloader():

    # randomly crop and rotate the image
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # apply the transform to the dataset
    dataset = Dataset(
        MemeDataset(),
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    return dataloader
    

if __name__ == "__main__":
    
    dataset = MemeDataset()
    # test_dataset_getitem(0)
    # test_dataset_getitem(10000)
    
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    imgs, captions = next(iter(dataloader))
    
    