import torch
from torch import nn
from torchvision import models
import sys
import os
from dataset import MemeCaptionDataset
import numpy as np

from settings import caption_batch_size, workers, caption_num_epochs, caption_save_every

CKPT_PATH = "./caption-model-ckpts"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        return self.embed(densenet_outputs)


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        self.lstm_cell = nn.LSTMCell(input_size=embed_size, hidden_size=hidden_size)
    
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
    
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
    
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, features, captions):
  
        batch_size = features.size(0)
        
        hidden_state = torch.zeros((batch_size, self.hidden_size)).to(device)
        cell_state = torch.zeros((batch_size, self.hidden_size)).to(device)
    
        outputs = torch.empty((batch_size, captions.size(1), self.vocab_size)).to(device)

        captions_embed = self.embed(captions)
        
        for t in range(captions.size(1)):

            if t == 0:
                hidden_state, cell_state = self.lstm_cell(features, (hidden_state, cell_state))
                
            else:
                hidden_state, cell_state = self.lstm_cell(captions_embed[:, t, :], (hidden_state, cell_state))
            
            out = self.fc_out(hidden_state)
            
            outputs[:, t, :] = out
    
        return outputs


def train():

    dataset = MemeCaptionDataset()
    vocab_size = len(dataset.itos)

    encoder = EncoderCNN().to(device)
    decoder = DecoderRNN(
        embed_size=1024, 
        hidden_size=1024, 
        vocab_size=vocab_size
    ).to(device)

    data_loader = iter(torch.utils.data.DataLoader(
        dataset,
        batch_size=caption_batch_size,
        shuffle=True, 
        num_workers=workers
    ))

    val_data_loader = iter(torch.utils.data.DataLoader(
        dataset,
        batch_size=caption_batch_size,
        shuffle=True, 
        num_workers=workers
    ))

    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params)

    criterion = nn.CrossEntropyLoss()

    total_step = 10

    losses = list()
    val_losses = list()

    for epoch in range(1, caption_num_epochs + 1):
        
        for i_step in range(1, total_step+1):
            
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
            captions_train = captions[:, :captions.shape[1]-1].to(device)

            # Move batch of images and captions to GPU if CUDA is available.
            images = images.to(device)
            
            # Pass the inputs through the CNN-RNN model.
            features = encoder(images)
            outputs = decoder(features, captions_train)
            
            # Calculate the batch loss
            loss = criterion(outputs.view(-1, vocab_size), captions_target.contiguous().view(-1))
            
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
                captions_train = val_captions[:, :val_captions.shape[1]-1].to(device)

                # Move batch of images and captions to GPU if CUDA is available.
                val_images = val_images.to(device)

                # Pass the inputs through the CNN-RNN model.
                features = encoder(val_images)
                outputs = decoder(features, captions_train)

                # Calculate the batch loss.
                val_loss = criterion(outputs.view(-1, vocab_size), captions_target.contiguous().view(-1))
            
            # append the validation loss and training loss
            val_losses.append(val_loss.item())
            losses.append(loss.item())
            
            # save the losses
            np.save('losses', np.array(losses))
            np.save('val_losses', np.array(val_losses))
            
            # Get training statistics.
            stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Val Loss: %.4f' % (epoch, caption_num_epochs, i_step, total_step, loss.item(), val_loss.item())
            
            # Print training statistics (on same line).
            print('\r' + stats, end="")
            sys.stdout.flush()
                
        # Save the weights.
        if epoch % caption_save_every == 0:
            print("\nSaving the model")
            torch.save(decoder.state_dict(), os.path.join(CKPT_PATH, 'decoder-%d.pth' % epoch))
            torch.save(encoder.state_dict(), os.path.join(CKPT_PATH, 'encoder-%d.pth' % epoch))

    # Save the weights.
    if epoch % caption_save_every == 0:
        print("\nSaving the model")
        torch.save(decoder.state_dict(), os.path.join(CKPT_PATH, 'final.pth'))
        torch.save(encoder.state_dict(), os.path.join(CKPT_PATH, 'final.pth'))

if __name__ == "__main__":
    train()