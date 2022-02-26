import torch
import torch.nn as nn
import sys
import numpy as np
import os

from dataset import get_meme_caption_dataset
from utils import get_preprocessing_normalisation_transform
from captionmodel import RefinedLanguageModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CKPT_PATH = "./caption-model-v2-ckpts"
BATCH_SIZE = 64
NUM_WORKERS = 8
NUM_EPOCHS = 2000
SAVE_EVERY = 10

def train():
    print("Initializing dataset...")

    dataset = get_meme_caption_dataset(
        transform=get_preprocessing_normalisation_transform(image_size=512)
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    vocab_size = len(dataset.itos)

    print("Initializing model...")
    model = RefinedLanguageModel(
        vocab_embed_size=512,
        decoder_hidden_size=1024,
        decoder_num_layers=1,
        vocab_size=vocab_size,
        encoder_embed_size=1024
    ).to(device)

    print("Initializing optimizer and loss function...")

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    losses = []

    print("Starting training...")

    for epoch in range(NUM_EPOCHS):
        for idx, (image_batch, labels_batch) in enumerate(data_loader):
            image_batch = image_batch.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            model.zero_grad()
            model.train()

            captions = labels_batch[:, :-1]
            captions_target = torch.nn.functional.one_hot(labels_batch[:, 1:], vocab_size).double().to(device)
            
            output = model(image_batch, captions)
            loss = criterion(output, captions_target)
            losses.append(loss.item())
            
            print_training_stats(
                current_epoch=epoch,
                total_num_epochs=NUM_EPOCHS,
                current_batch=idx,
                total_batches=len(data_loader),
                train_loss=loss.item()
            )

            loss.backward()
            optimizer.step()

        if epoch % SAVE_EVERY == 0:
            # ensure that we have the relevant directories created
            os.makedirs(CKPT_PATH, exist_ok=True)

            # save both the model parameters and the loss history
            torch.save(model.state_dict(), f"{CKPT_PATH}/epoch-{epoch}.ckpt")
            np.save(f"{CKPT_PATH}/losses", np.array(losses))

def print_training_stats(
    current_epoch,
    total_num_epochs,
    current_batch,
    total_batches,
    train_loss
):
    stats = f"Epoch {current_epoch}/{total_num_epochs} - Current Batch {current_batch}/{total_batches} - Loss: {train_loss} - Other random padding to stop things getting cut off..."
    print("\r" + stats, end="")
    sys.stdout.flush()

if __name__ == "__main__":
    train()