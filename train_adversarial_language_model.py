# FOr the generator, we have a model that takes caption, image and (effectively) outputs the next word.

# discriminator must take image, caption, next word, and output true/false

import torch
import sys

from settings import refined_model_vocab_embed_size, refined_model_decoder_hidden_size, refined_model_decoder_num_layers, refined_model_encoder_embed_size, refined_model_batch_size, refined_model_num_epochs, refined_model_save_every, refined_model_num_workers

from captionmodel import RefinedLanguageModel, LanguageModelDiscriminator
from dataset import get_meme_caption_dataset
from utils import get_preprocessing_normalisation_transform

from tqdm import tqdm

CKPT_PATH = "./caption-model-adversarial"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():
    """
    Train the model in an adversarial fashion
    """
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

    print("Finished loading dataset")

    generator = RefinedLanguageModel(
        vocab_embed_size=refined_model_vocab_embed_size,
        decoder_hidden_size=refined_model_decoder_hidden_size,
        decoder_num_layers=refined_model_decoder_num_layers,
        vocab_size=vocab_size,
        encoder_embed_size=refined_model_encoder_embed_size
    ).to(device)

    discriminator = LanguageModelDiscriminator(
        vocab_embed_size=refined_model_vocab_embed_size,
        decoder_hidden_size=refined_model_decoder_hidden_size,
        decoder_num_layers=refined_model_decoder_num_layers,
        vocab_size=vocab_size,
        encoder_embed_size=refined_model_encoder_embed_size
    ).to(device)

    print("Finished loading models")

    gen_opt = torch.optim.Adam(generator.parameters())
    disc_opt = torch.optim.Adam(discriminator.parameters())

    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(refined_model_num_epochs):

        gen_loss = '-'
        disc_loss = '-'
        with tqdm(data_loader, desc=f"Epoch {epoch}/{refined_model_num_epochs} | Gen Loss: {gen_loss} | Disc Loss: {disc_loss}") as pbar:
            for real_images, real_captions in pbar:
                real_images = real_images.to(device)
                real_captions = real_captions.to(device)

                generator.train()
                discriminator.train()

                ### Update the discriminator ###
                disc_opt.zero_grad()

                # randomly choose the 'length' of this batch (maximum is 25)
                batch_length = torch.randint(low=1, high=real_captions.shape[1], size=(1,)).to(device)
                real_next_word = real_captions[:, batch_length]
                real_captions = real_captions[:, :batch_length]

                # generate fake next word
                fake_next_word = generator(real_images, real_next_word)[:, -1]

                # convert real_next_word to one hot
                real_next_word_one_hot = torch.nn.functional.one_hot(real_next_word, num_classes=vocab_size).squeeze().float().to(device)                
                
                # print('Step 2: Discriminate')

                real_preds = discriminator(real_images, real_captions, real_next_word_one_hot)
                fake_preds = discriminator(real_images, real_captions, fake_next_word.detach())

                disc_real_loss = criterion(real_preds, torch.ones_like(real_preds))
                disc_fake_loss = criterion(fake_preds, torch.zeros_like(fake_preds))

                disc_loss = (disc_real_loss + disc_fake_loss) / 2

                disc_loss.backward(retain_graph=True)
                disc_opt.step()

                ### Update the generator ###
                gen_opt.zero_grad()

                fake_preds = discriminator(real_images, real_captions, fake_next_word)
                gen_loss = criterion(fake_preds, torch.ones_like(fake_preds))
                gen_loss.backward()
                gen_opt.step()

                pbar.set_description(f"Epoch {epoch}/{refined_model_num_epochs} | Gen Loss: {gen_loss} | Disc Loss: {disc_loss}")

        if epoch % refined_model_save_every == 0:
            torch.save(generator.state_dict(), f"{CKPT_PATH}/generator_epoch_{epoch}.ckpt")
            torch.save(discriminator.state_dict(), f"{CKPT_PATH}/discriminator_epoch_{epoch}.ckpt")


def print_training_stats(
    current_epoch,
    total_num_epochs,
    disc_loss,
    gen_loss,
):
    stats = f"Epoch {current_epoch}/{total_num_epochs} - Generator Loss: {gen_loss} - Disciminator Loss: {disc_loss} - Other random padding to stop things getting cut off..."
    print("\r" + stats, end="")
    sys.stdout.flush()

if __name__ == '__main__':
    train()