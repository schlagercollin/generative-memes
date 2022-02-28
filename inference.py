"""
inference.py
============

Scripts for model inference.
"""
import torch
import utils
import numpy as np
from tqdm import tqdm

from utils import get_preprocessing_normalisation_transform


def gt_vec_to_text(caption, dataset):
    """Converts word vector representation of ground-truth caption to text.

    Args:
        outputs: word vector
        dataset: corresponding dataset

    Returns:
        list of word captions
    """
    sentences = []
    for i in range(caption.shape[0]):
        sentence = []
        for j in range(caption.shape[1]):
            sentence.append(dataset.itos[caption[i, j]])
        sentences.append(sentence)

    return sentences


def pred_vec_to_text(outputs, dataset):
    """Converts word vector representation of prediction to text.

    Args:
        outputs: word vector
        dataset: corresponding dataset

    Returns:
        list of word captions
    """

    sentences = []
    max_elements = torch.argmax(outputs, axis=2)

    for prediction in range(max_elements.shape[0]):
        sentence = []
        for i in range(max_elements.shape[1]):
            sentence.append(dataset.itos[max_elements[prediction, i]])
        sentences.append(sentence)

    return sentences


def generate_background(generator, device, noise=None, truncated_normal=False):
    if noise is None:
        if truncated_normal:
            noise = torch.nn.init.trunc_normal_(
                torch.zeros((1, 100)),
                a=-2,
                b=2,
            )
        else:
            noise = generator.sampler(1, device)[0]
    
    generator_out = generator.forward(noise)
    return generator_out


def generate_caption(generator_output, encoder, decoder, data_loader, device, dataset):

    arr = np.asarray(utils.tensor_to_image(generator_output, ncol=1, padding=0), dtype=np.int32)
    arr = arr[np.newaxis, :, :, :]
    arr = torch.from_numpy(arr.transpose(0, 3, 1, 2)).type(torch.FloatTensor)

    features = encoder(arr)

    _, dummy_cap = next(data_loader)
    dummy_cap = dummy_cap[:1].to(device)
    caption_tensor = decoder(features, dummy_cap)
    text_caption = " ".join([word for word in pred_vec_to_text(caption_tensor, dataset)[0] if word != "<UNK>"])
    
    return text_caption


def generate_caption_v2_beam_search(
        generator_output,
        model,
        dataset,
        device,
        length_to_generate,
        beam_search_temperature=2.5,
        branch_factor=10
):
    assert generator_output.shape[0] == 1, "This function currently only supports a batch size of 1"

    # first we'll need to apply the appropriate transforms to the input tensor
    transform = get_preprocessing_normalisation_transform(image_size=512)
    generator_output = transform(generator_output)
    
    # begin with the start token, which should occur with certainty
    best_captions = [(torch.LongTensor([[dataset.stoi[dataset.start]]]).to(device), 1, False)]

    def generate_next_caption_from_base_caption(
        base_caption
    ):
        out = model(generator_output, base_caption)
        out = out[0, -1, :]
        next_word_idx = torch.argsort(out, descending=True)
        next_word_idx = next_word_idx[:100]
        next_word_probs = out[next_word_idx]
        next_word_probs = next_word_probs ** (1 / beam_search_temperature) / torch.sum(next_word_probs ** (1 / beam_search_temperature))
        index_to_choose = torch.multinomial(next_word_probs, 1)
        next_word_idx = next_word_idx[index_to_choose]
        caption = torch.cat([base_caption, torch.LongTensor([[next_word_idx]]).to(device)], dim=1)

        # option 1: weight the guess by MLE
        # probs = out[next_word_idx]

        #option 2: weight the guess by the Temperature modified probs
        probs = next_word_probs[index_to_choose]
        return caption, probs, dataset.stoi[dataset.end] == next_word_idx

    # feed the image and the start token to the model
    for i in range(length_to_generate):
        new_captions = []
        for caption, current_prob, is_done in tqdm(best_captions, desc=f"Generating captions of length {i}/{length_to_generate}"):
            if not is_done:
                # provide some suggestions on how to extend the caption
                for _ in range(branch_factor):
                    extended_caption, marginal_prob, done = generate_next_caption_from_base_caption(caption)
                    marginal_prob = current_prob + np.log(marginal_prob.detach().numpy())
                    new_captions.append((extended_caption, marginal_prob, done))
            else:
                # caption is done generating; just persist the caption
                new_captions.append((caption, current_prob, is_done))

        best_captions = sorted(new_captions, key=lambda x: x[1], reverse=True)[:branch_factor]

    # option 1: choose the predicted best caption
    overall_best_caption = sorted(best_captions, key=lambda x: x[1], reverse=True)[0][0]
    english_caption = []
    for idx in overall_best_caption.squeeze():
        if idx != dataset.stoi[dataset.start] and idx != dataset.stoi[dataset.end]:
            english_caption.append(dataset.itos[idx])

    all_captions = []
    for caption, _, _ in best_captions:
        all_captions.append(" ".join([dataset.itos[idx] for idx in caption.squeeze()]))

    return " ".join(english_caption), all_captions

def generate_caption_v2(
        generator_output,
        model,
        data_loader,
        dataset,
        device,
        length_to_generate,
        beam_search_temperature=2
):
    """Generates caption from generator output.

    Args:
        generator_output: generator output
        model: model
        data_loader: data loader
        dataset: dataset
        device: device
    """

    assert generator_output.shape[0] == 1, "This function currently only supports a batch size of 1"

    # first we'll need to apply the appropriate transforms to the input tensor
    transform = get_preprocessing_normalisation_transform(image_size=512)
    # print("Generator output shape", generator_output.shape)

    generator_output = transform(generator_output)
    # print("Generator output shape after rescaling", generator_output.shape)
    
    # begin with the start token
    caption = torch.LongTensor([[dataset.stoi[dataset.start]]]).to(device)
    # print("Caption shape", caption.shape)

    english_caption = []

    # feed the image and the start token to the model
    for i in tqdm(range(length_to_generate)):
        out = model(generator_output, caption)

        # print("Model output shape", out.shape)

        # extract the probability distribution over the next words
        out = out[0, -1, :]
        
        # extract the final prediction, by sampling from distribution over the 100 most probable words
        
        # indices of the 100 most probable words
        next_word_idx = torch.argsort(out, descending=True)
        next_word_idx = next_word_idx[:100]

        # probabilities of the indices in next_word_idx
        next_word_probs = out[next_word_idx]

        # rebalance the probabilities using a temperature change
        next_word_probs = next_word_probs ** (1 / beam_search_temperature) / torch.sum(next_word_probs ** (1 / beam_search_temperature))

        # now we sample from the distribution
        index_to_choose = torch.multinomial(next_word_probs, 1)
        print(index_to_choose)

        print(next_word_idx.shape)

        next_word_idx = next_word_idx[index_to_choose]

        # print(next_word_idx.shape)
        english_word = dataset.itos[next_word_idx]
        
        # print(f"Next word: {dataset.itos[next_word_idx]}")
        english_caption.append(english_word)

        if english_word == dataset.end:
            break

        caption = torch.cat([caption, torch.LongTensor([[next_word_idx]]).to(device)], dim=1)
        # print(caption.shape)

    return " ".join(english_caption)
       


def generate_meme(generator, encoder, decoder, data_loader, device, dataset, noise=None, truncated_normal=False):
    
    generator_out = generate_background(generator, device, noise, truncated_normal)
    
    caption = generate_caption(generator_out, encoder, decoder, data_loader, device, dataset)
    
    generator_img = utils.tensor_to_image(generator_out)

    meme = utils.Meme(caption, generator_img)
    meme_img = meme.draw()
    
    return meme_img

def generate_meme_v2(generator, model, data_loader, device, dataset, noise=None, truncated_normal=False, beam_search=True, custom_caption=None):
    
    generator_out = generate_background(generator, device, noise, truncated_normal)
    if custom_caption is not None:
        caption = custom_caption
    elif beam_search:
        caption, _ = generate_caption_v2_beam_search(
            generator_out,
            model,
            dataset,
            device,
            length_to_generate=10,
            beam_search_temperature=2,
            branch_factor=1
        )
    else:
        caption = generate_caption_v2(generator_out, model, data_loader, dataset, device, 10)
    # print(generator_out.mean())
    generator_img = utils.tensor_to_image(generator_out)

    meme = utils.Meme(caption, generator_img)
    meme_img = meme.draw()
    
    return meme_img

# delete all of this vvvvv
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from dataset import get_meme_caption_dataset
    from captionmodel import RefinedLanguageModel

    from settings import refined_model_vocab_embed_size, refined_model_decoder_hidden_size, refined_model_decoder_num_layers, refined_model_encoder_embed_size

    MODEL_PARAMS_TO_LOAD = 'v3-epoch-0.ckpt'
    MODEL_CKPT_PATH = f'./caption-model-v2-ckpts/{MODEL_PARAMS_TO_LOAD}'

    dataset = get_meme_caption_dataset()
    vocab_size = len(dataset.itos)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1
    )

    model = RefinedLanguageModel(
                vocab_embed_size=refined_model_vocab_embed_size,
                decoder_hidden_size=refined_model_decoder_hidden_size,
                decoder_num_layers=refined_model_decoder_num_layers,
                vocab_size=vocab_size,
                encoder_embed_size=refined_model_encoder_embed_size
            ).to(device)

    model.load_state_dict(
        torch.load(MODEL_CKPT_PATH, map_location=torch.device(device))
    )

    model.eval()

    out = torch.rand((1, 3, 64, 64))
    print(out.shape)

    # caption = generate_caption_v2(out, model, data_loader, dataset, device, 10)
    # print(caption)

    best_caption, all_captions = generate_caption_v2_beam_search(
        out,
        model,
        dataset,
        device,
        length_to_generate=10
    )

    print(all_captions)
    print(best_caption)