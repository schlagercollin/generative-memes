import torch
import torch.nn as nn
import sys
import numpy as np
from torchvision import transforms

from dataset import MemeCaptionDataset
from utils import get_preprocessing_normalisation_transform
from captionmodel import RefinedLanguageModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CKPT_PATH = "./caption-model-v2-ckpts"
BATCH_SIZE = 1
NUM_WORKERS = 8
NUM_EPOCHS = 2000
SAVE_EVERY = 10

def train():
    print("Initializing dataset...")
    dataset = MemeCaptionDataset(
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

    for epoch in range(NUM_EPOCHS):
        for idx, (image_batch, labels_batch) in enumerate(data_loader):
            image_batch = image_batch.to(device)
            labels_batch = labels_batch.to(device)

            # we are doing next-character prediction, so we want to give the model progressively longer
            # input sequences
            # for i in range(1, labels_batch.shape[1] - 1):
            optimizer.zero_grad()
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
                total_batches=len(data_loader) // BATCH_SIZE,
                train_loss=loss.item()
            )

            loss.backward()
            optimizer.step()

            break

        if epoch % SAVE_EVERY == 0:
            torch.save(model.state_dict(), f"{CKPT_PATH}/epoch-{epoch}.ckpt")
            np.save("losses", np.array(losses))

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