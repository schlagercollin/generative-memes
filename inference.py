"""
inference.py
============

Scripts for model inference.
"""
import torch
import utils
import numpy as np


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


def generate_meme(generator, encoder, decoder, data_loader, device, dataset, noise=None, truncated_normal=False):
    
    generator_out = generate_background(generator, device, noise, truncated_normal)
    caption = generate_caption(generator_out, encoder, decoder, data_loader, device, dataset)
    
    generator_img = utils.tensor_to_image(generator_out)

    meme = utils.Meme(caption, generator_img)
    meme_img = meme.draw()
    
    return meme_img