import torch
import torch.nn as nn
import sys
import numpy as np
import os

from dataset import get_meme_caption_dataset
from utils import get_preprocessing_normalisation_transform
from captionmodel import RefinedLanguageModel

from settings import refined_model_vocab_embed_size, refined_model_decoder_hidden_size, refined_model_decoder_num_layers, refined_model_encoder_embed_size, refined_model_batch_size, refined_model_num_epochs, refined_model_save_every, refined_model_num_workers

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CKPT_PATH = "./caption-model-v2-ckpts"

def train():
    print("Initializing dataset...")

    dataset = get_meme_caption_dataset(
        transform=get_preprocessing_normalisation_transform(image_size=512)
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=refined_model_batch_size,
        shuffle=True,
        num_workers=refined_model_num_workers
    )
    vocab_size = len(dataset.itos)

    print("Initializing model...")
    model = RefinedLanguageModel(
        vocab_embed_size=refined_model_vocab_embed_size,
        decoder_hidden_size=refined_model_decoder_hidden_size,
        decoder_num_layers=refined_model_decoder_num_layers,
        vocab_size=vocab_size,
        encoder_embed_size=refined_model_encoder_embed_size
    ).to(device)

    print("Initializing optimizer and loss function...")

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    losses = []

    print("Starting training...")

    for epoch in range(refined_model_num_epochs):
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
                total_num_epochs=refined_model_num_epochs,
                current_batch=idx,
                total_batches=len(data_loader),
                train_loss=loss.item()
            )

            loss.backward()
            optimizer.step()

        if epoch % refined_model_save_every == 0:
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