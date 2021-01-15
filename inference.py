import numpy as np
import torch
from torch.nn import functional as F
from helper import one_hot_encode
from model import Network

with open('data/x_files_dataset.txt', 'r') as f:
    text = f.read()


# We need turn our data into numerical tokens
# Neural networks can only learn from numerical data
chars = tuple(set(text))
# obtaining all of the unique characters being used in the text
chars = tuple(set(text))

with open('weights/rnn_70_epochs.net', 'rb') as f:
    checkpoint = torch.load(f, map_location='cpu')

net = Network(checkpoint['tokens'],
              n_hidden=checkpoint['n_hidden'],
              n_layers=checkpoint['n_layers'])

net.load_state_dict(checkpoint['state_dict'])


def predict(net, char, h=None, top_k=None):
    x = np.array([[net.char2int[char]]])
    x = one_hot_encode(x, len(net.chars))
    inputs = torch.from_numpy(x)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    h = tuple([each.data for each in h])
    out, h = net(inputs, h)
    p = F.softmax(out, dim=1).data
    if torch.cuda.is_available():
        p = p.cpu()
    if top_k is None:
        top_ch = np.arange(len(net.chars))
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()
    p = p.numpy().squeeze()
    char = np.random.choice(top_ch, p=p/p.sum())
    return net.int2char[char], h


def sample(net=net, size=2000, prime='Mulder', top_k=2):
    if torch.cuda.is_available():
        net.cuda()
    else:
        net.cpu()
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)
    chars.append(char)
    for i in range(size):
        char, h = predict(net, char[-1], h, top_k=top_k)
        chars.append(char)
    return ''.join(chars)
