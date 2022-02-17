# Deep Generative Memes

Authors: `Collin Schlager`, `Harry Mellsop`, `Randy Quarles` 


Using GANs to create memes (and soon other image + image caption data!).

# Training Model Instructions

For our baseline model, we have two separate architectures that need to be trained: 
the image generation model and the text caption model.

These are trained by running:

```
python train_baseline_image_gen.py
python train_baseline_caption_gen.py
```

These training scripts will produce model checkpoint files that are then used in the inference
step (see next section).

# Inference (Generation) Instructions

Inference can be easily run in a Jupyter Notebook. Please see `infer.ipynb`.

The first step is to point the notebook variables to your saved model checkpoints.
Then you can sample memes and their captions via the last cell.

# Dataset

Dataset files are contained in the `scraper` directory. This directory contains a cloned repo of the [ImgFlip575K Memes Dataset](https://github.com/schesa/ImgFlip575K_Dataset) for convenience. The repo houses code for the ImgFlip scrapper as well as a pre-scraped version of the dataset.
