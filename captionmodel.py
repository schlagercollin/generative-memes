"""
Baseline Captioning Model

EncoderCNN
DecoderRNN
"""
from torchvision import models
from torch import nn
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
