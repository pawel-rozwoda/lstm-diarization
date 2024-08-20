import torch
import torch.nn as nn
import pandas as pd
from ge2e import similarity_per_speaker

import sys
sys.path.append('../')
from aux import get_embeddings


class LSTM_Diarization(nn.Module):
    def __init__(self,* , input_dim, hidden_dim, num_layers, init_w, init_b, train=False):
        super(LSTM_Diarization,self).__init__() 

        self.dtype = torch.double
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers 
        self.lstm_layer = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear1 = nn.Linear(768, 512)
        self.linear2 = nn.Linear(512, 256)
        self.window_size=25
        self.shift = 10
        self.w = torch.nn.Parameter(torch.tensor(init_w))
        self.b = torch.nn.Parameter(torch.tensor(init_b))

        self.linear2.requires_grad = True
        self.linear1.requires_grad = True
        self.lstm_layer.requires_grad=True
        self.w.requires_grad = False
        self.b.requires_grad = False

        self.train = train
        

    def forward(self, batch): 
        """ sliding window on feats """
        spk_count = batch.shape[0]
        _b = [batch[:, i:i+self.window_size] for i in range(0, batch.shape[1] - self.window_size, self.shift)]
        batch = torch.stack(_b, dim=1)

        """ sticking 1 and 2 dims """
        batch = batch.reshape((batch.shape[0], -1, 40))

        batch = batch.reshape((-1, self.window_size, 40))

        out, (_, _) = self.lstm_layer(batch) 

        """ last outputs """
        out = out[:, -1, :]
        """ last outputs """ 

        out = self.linear1(out) 
        out = self.linear2(out)
        out = out.reshape(spk_count, -1, 256)

        """ 
            embeddings 
            norm, stack, average
        """

        embeddings = get_embeddings(out) 
        """ end embeddings """

        if self.train:
            similarities = (similarity_per_speaker(embeddings) * self.w) + self.b
            return similarities

        else:
            return embeddings
