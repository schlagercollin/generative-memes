"""
Baseline Captioning Model

EncoderCNN
DecoderRNN
"""
from torchvision import models
from torch import nn
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Our most modern language model

class RefinedLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_embed_size=512,
        decoder_hidden_size=1024,
        decoder_num_layers=1,
        vocab_size=234,
        encoder_embed_size=1024
    ):
        super(RefinedLanguageModel, self).__init__()

        # encoder modules
        self.encoder_cnn = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        self.encoder_cnn.fc = nn.Identity()             # essentially strip the last layer
        # self.encoder_cnn.requires_grad = False

        self.encoder_to_decoder = nn.Linear(2048, encoder_embed_size)
        self.embed = nn.Embedding(vocab_size, vocab_embed_size)

        # decoder modules
        self.decoder_lstm = nn.LSTM(
            input_size=vocab_embed_size + encoder_embed_size,
            hidden_size=decoder_hidden_size,
            num_layers=decoder_num_layers,
            batch_first=True,
        )
        self.decoder_to_vocab = nn.Linear(decoder_hidden_size, vocab_size)
        self.output_activation = nn.Softmax(dim=-1)

    def forward(self, images, captions):
        # we should run a check here and throw a warning if not properly normalised; see instructions:
        # https://pytorch.org/hub/pytorch_vision_inception_v3/

        # extract image features from image batch
        # with torch.no_grad():
        embeddings = self.encoder_cnn(images).logits
        # print(embeddings.shape)

        embeddings = self.encoder_to_decoder(embeddings)

        # concatenate image features and caption embeddings at each time step
        embeddings = embeddings[:, None, :]
        captions_embed = self.embed(captions)
        embeddings = embeddings.repeat(1, captions_embed.size(1), 1)
        lstm_input = torch.cat([captions_embed, embeddings], dim=2)

        # pass this through LSTM
        lstm_output, _ = self.decoder_lstm(lstm_input)

        vocab_output = self.decoder_to_vocab(lstm_output)
        
        return self.output_activation(vocab_output)

# Our older language models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size=1024):
        super(EncoderCNN, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        self.densenet.classifier = nn.Linear(in_features=1024, out_features=1024)
        self.embed = nn.Linear(in_features=1024, out_features=embed_size)
        self.dropout = nn.Dropout(p=0.5)
        self.prelu = nn.PReLU()

    def forward(self, images):
        densenet_outputs = self.dropout(self.prelu(self.densenet(images)))
        return self.embed(densenet_outputs)


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.lstm_cell = nn.LSTMCell(input_size=embed_size, hidden_size=hidden_size)

        self.fc_out = nn.Linear(
            in_features=self.hidden_size, out_features=self.vocab_size
        )

        self.embed = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.embed_size
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, captions):

        batch_size = features.size(0)

        hidden_state = torch.zeros((batch_size, self.hidden_size)).to(device)
        cell_state = torch.zeros((batch_size, self.hidden_size)).to(device)

        outputs = torch.empty((batch_size, captions.size(1), self.vocab_size)).to(
            device
        )

        captions_embed = self.embed(captions)

        for t in range(captions.size(1)):

            if t == 0:
                hidden_state, cell_state = self.lstm_cell(
                    features, (hidden_state, cell_state)
                )

            else:
                hidden_state, cell_state = self.lstm_cell(
                    captions_embed[:, t, :], (hidden_state, cell_state)
                )

            out = self.fc_out(hidden_state)

            outputs[:, t, :] = out

        return outputs

if __name__ == "__main__":
    test_refined = RefinedLanguageModel()
    # ds = MemeCaptionDataset()
    # image, caption = ds[0][0], ds[0][1]
    # print(image.shape)
    # print(caption.shape)

    batch_size = 2
    test_images = torch.rand((batch_size, 3, 512, 512))
    test_captions = torch.randint(low=0, high=233, size=(batch_size, 25))
    out = test_refined(test_images, test_captions)

    print(out.shape)
    print(out.sum(dim=1))