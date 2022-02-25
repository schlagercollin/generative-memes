"""
Baseline Captioning Model

EncoderCNN
DecoderRNN
"""
from torchvision import models
from torch import nn
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RefinedLanguageModel(nn.Module):
    def __init__(
        self,
        decoder_embed_size,
        decoder_hidden_size,
        vocab_size,
        encoder_embed_size=1024
    ):
        super(RefinedLanguageModel, self).__init__()

        # save params
        self.decoder_embed_size = decoder_embed_size
        self.decoder_hidden_size = decoder_hidden_size
        self.vocab_size = vocab_size
        self.encoder_embed_size = encoder_embed_size

        # encoder modules
        self.densenet = models.densenet121(pretrained=True)
        self.densenet.classifier = nn.Linear(
            in_features=1024,
            out_features=1024
        )
        self.embed = nn.Linear(
            in_features=1024,
            out_features=encoder_embed_size
        )
        self.dropout = nn.Dropout(p=0.5)
        self.prelu = nn.PReLU()

        # decoder modules
        self.lstm_cell = nn.LSTMCell(
            input_size=self.decoder_embed_size,
            hidden_size=self.decoder_hidden_size
        )

        # keep going from here

    def forward(self, images):
        # overall strategy here;
        # - use the densenet, or ideally can we use InceptionNet pretrained on ILSVRC-2012-CLS
        # - strip last layer
        # - this should give us 2048 output dimension vector, map this down to be our hidden dimension size using FC layer
        # - LSTM cell repeat however many times
        # - approach based on simplified: https://arxiv.org/pdf/1806.04510.pdf
        densenet_outputs = self.prelu(self.densenet(images))
        post_dropout = self.dropout(densenet_outputs)
        embedded = self.embed(post_dropout)


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