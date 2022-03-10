"""
train_baseline_caption_gen.py

Main training function for the baseline caption generator.

Run with `python train_baseline_caption_gen.py`
"""
import torch
from torch import nn
import sys
import os
from dataset import MemeCaptionDataset
import numpy as np
from captionmodel import EncoderCNN, DecoderRNN

from settings import caption_batch_size, workers, caption_num_epochs, caption_save_every

CKPT_PATH = "./caption-model-ckpts"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train():

    dataset = MemeCaptionDataset()
    vocab_size = len(dataset.itos)

    print("Vocab =", vocab_size)

    encoder = EncoderCNN().to(device)
    decoder = DecoderRNN(
        embed_size=1024, 
        hidden_size=1024, 
        vocab_size=vocab_size
    ).to(device)

    data_loader = iter(
        torch.utils.data.DataLoader(
            dataset, batch_size=caption_batch_size, shuffle=True, num_workers=workers
        )
    )

    val_data_loader = iter(
        torch.utils.data.DataLoader(
            dataset, batch_size=caption_batch_size, shuffle=True, num_workers=workers
        )
    )

    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params)

    criterion = nn.CrossEntropyLoss()

    total_step = 10

    losses = list()
    val_losses = list()

    for epoch in range(1, caption_num_epochs + 1):

        for i_step in range(1, total_step + 1):

            # zero the gradients
            decoder.zero_grad()
            encoder.zero_grad()

            # set decoder and encoder into train mode
            encoder.train()
            decoder.train()

            # Obtain the batch.
            images, captions = next(data_loader)

            # make the captions for targets and teacher forcer
            captions_target = captions[:, 1:].to(device)
            captions_train = captions[:, : captions.shape[1] - 1].to(device)

            # Move batch of images and captions to GPU if CUDA is available.
            images = images.to(device)

            # Pass the inputs through the CNN-RNN model.
            features = encoder(images)
            outputs = decoder(features, captions_train)

            # Calculate the batch loss
            loss = criterion(
                outputs.view(-1, vocab_size), captions_target.contiguous().view(-1)
            )

            # Backward pass
            loss.backward()

            # Update the parameters in the optimizer
            optimizer.step()

            # - - - Validate - - -
            # turn the evaluation mode on
            with torch.no_grad():

                # set the evaluation mode
                encoder.eval()
                decoder.eval()

                # get the validation images and captions
                val_images, val_captions = next(val_data_loader)

                # define the captions
                captions_target = val_captions[:, 1:].to(device)
                captions_train = val_captions[:, : val_captions.shape[1] - 1].to(device)

                # Move batch of images and captions to GPU if CUDA is available.
                val_images = val_images.to(device)

                # Pass the inputs through the CNN-RNN model.
                features = encoder(val_images)
                outputs = decoder(features, captions_train)

                # Calculate the batch loss.
                val_loss = criterion(
                    outputs.view(-1, vocab_size), captions_target.contiguous().view(-1)
                )

            # append the validation loss and training loss
            val_losses.append(val_loss.item())
            losses.append(loss.item())

            # save the losses
            np.save("losses", np.array(losses))
            np.save("val_losses", np.array(val_losses))

            # Get training statistics.
            stats = "Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Val Loss: %.4f" % (
                epoch,
                caption_num_epochs,
                i_step,
                total_step,
                loss.item(),
                val_loss.item(),
            )

            # Print training statistics (on same line).
            print("\r" + stats, end="")
            sys.stdout.flush()

        # Save the weights.
        if epoch % caption_save_every == 0:
            print("\nSaving the model")
            torch.save(
                decoder.state_dict(), os.path.join(CKPT_PATH, "decoder-%d.pth" % epoch)
            )
            torch.save(
                encoder.state_dict(), os.path.join(CKPT_PATH, "encoder-%d.pth" % epoch)
            )

    # Save the weights.
    if epoch % caption_save_every == 0:
        print("\nSaving the model")
        torch.save(decoder.state_dict(), os.path.join(CKPT_PATH, "final.pth"))
        torch.save(encoder.state_dict(), os.path.join(CKPT_PATH, "final.pth"))


if __name__ == "__main__":
    train()
