import torch
from torch import nn


class Network(nn.Module):
    def __init__(self, tokens, n_hidden=256, n_layers=2,
                 drop_prob=0.5, lr=0.01):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.lr = lr
        # create character dictionairies
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: i for i, ch in self.int2char.items()}
        self.lstm = nn.LSTM(len(self.chars),
                            n_hidden,
                            n_layers,
                            dropout=drop_prob,
                            batch_first=True)
        self.dropout = nn.Dropout(self.drop_prob)
        self.fc = nn.Linear(n_hidden, len(self.chars))

    def forward(self, x, hidden):
        x, (h, c) = self.lstm(x, hidden)
        x = self.dropout(x)
        x = x.view(-1, self.n_hidden)
        x = self.fc(x)
        return x, (h, c)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if torch.cuda.is_available():
            hidden = (weight.new(self.n_layers,
                                 batch_size,
                                 self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers,
                                 batch_size,
                                 self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers,
                                 batch_size,
                                 self.n_hidden).zero_(),
                      weight.new(self.n_layers,
                                 batch_size,
                                 self.n_hidden).zero_())
        return hidden
