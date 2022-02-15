import torch
from torch import nn
from torchvision import models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size = 1024):
        super(EncoderCNN, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        self.densenet.classifier = nn.Linear(in_features=1024, out_features=1024)
        self.embed = nn.Linear(in_features=1024, out_features=embed_size)
        self.dropout = nn.Dropout(p=0.5)
        self.prelu = nn.PReLU()
        
    def forward(self, images):
        
        densenet_outputs = self.dropout(self.prelu(self.densenet(images)))
        embeddings = self.embed(densenet_outputs)
        
        return embeddings