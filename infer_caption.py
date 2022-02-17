import torch
import os

from dataset import MemeCaptionDataset
from captionmodel import EncoderCNN, DecoderRNN

from settings import caption_batch_size, workers

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CKPT_PATH = "./caption-model-ckpts"

def infer():
    dataset = MemeCaptionDataset()
    vocab_size = len(dataset.itos)

    encoder = EncoderCNN().to(device)
    decoder = DecoderRNN(
        embed_size=1024, 
        hidden_size=1024, 
        vocab_size=vocab_size
    ).to(device)

    test_data_loader = iter(torch.utils.data.DataLoader(
        dataset,
        batch_size=caption_batch_size,
        shuffle=True, 
        num_workers=workers
    ))

    # print("Loaded dataset; now loading model")

    encoder.load_state_dict(
        torch.load(os.path.join(CKPT_PATH, 'encoder-%d.pth' % 440), map_location=torch.device(device))
    )

    decoder.load_state_dict(
        torch.load(os.path.join(CKPT_PATH, 'decoder-%d.pth' % 440), map_location=torch.device(device))
    )
    encoder.eval()
    decoder.eval()

    # print("Successfully loaded model.")

    for i_step, (img, caption) in enumerate(test_data_loader):
        img = img.to(device)
        caption = caption.to(device)

        features = encoder(img)
        outputs = decoder(features, caption)

        predictions = pred_vec_to_text(outputs, dataset)

        ground_truth_sentences = gt_vec_to_text(caption, dataset)

        for i in range(outputs.shape[0]):
            print("Predicted: ", predictions[i])
            print("Ground truth: ", ground_truth_sentences[i])
            print('\n\n\n')

        break

def gt_vec_to_text(caption, dataset):
    sentences = []
    for i in range(caption.shape[0]):
        sentence = []
        for j in range(caption.shape[1]):
            sentence.append(dataset.itos[caption[i, j]])
        sentences.append(sentence)

    return sentences

def pred_vec_to_text(outputs, dataset):
    sentences = []
    # print(outputs.shape)
    max_elements = torch.argmax(outputs, axis=2)
    # print(max_elements.shape)
    # print(max_elements)

    for prediction in range(max_elements.shape[0]):
        sentence = []
        for i in range(max_elements.shape[1]):
            sentence.append(dataset.itos[max_elements[prediction, i]])
        sentences.append(sentence)

    return sentences

if __name__ == "__main__":
    infer()