import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from helper import one_hot_encode, get_batches
from model import Network


def train(net, data, epochs=10, batch_size=10, seq_length=50, lr=0.001):
    clip = 5
    val_frac = 0.1
    print_every = 10
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    val_idx = int(len(data) * (1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]
    if torch.cuda.is_available():
        net = net.cuda()
    counter = 0
    n_chars = len(net.chars)
    print(data)
    for e in range(epochs):
        h = net.init_hidden(batch_size)
        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1
            x = one_hot_encode(x, n_chars)
            inputs = torch.from_numpy(x)
            targets = torch.from_numpy(y)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.long().cuda()
            h = tuple([each.data for each in h])
            net.zero_grad()
            output, h = net(inputs, h)
            loss = criterion(output, targets.view(batch_size*seq_length))
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            if counter % print_every == 0:
                val_h = net.init_hidden(batch_size)
                val_losses = []
                for x, y in get_batches(val_data, batch_size, seq_length):
                    x = one_hot_encode(x, n_chars)
                    x = torch.from_numpy(x)
                    y = torch.from_numpy(y)
                    val_h = tuple([each.data for each in val_h])
                    inputs, targets = x, y
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        targets = targets.long().cuda()
                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output,
                                         targets.view(batch_size*seq_length))
                    val_losses.append(val_loss.item())
                    print('epoch: {}/{}'.format(e+1, epochs),
                          'steps: {}'.format(counter),
                          'Loss {:.4f}'.format(loss.item()),
                          'val_loss {:.4f}'.format(np.mean(val_losses)))


with open('data/x_files_dataset.txt', 'r') as f:
    text = f.read()


# We need turn our data into numerical tokens
# Neural networks can only learn from numerical data
chars = tuple(set(text))
# obtaining all of the unique characters being used in the text
chars = tuple(set(text))

# coverting the chars into a dictionary with the index
# being the key, and the unique chars being the value
int2char = dict(enumerate(chars))
# Creating a dictionary where the we map the unique characters as key
# the values being the digits
char2int = {ch: i for i, ch in int2char.items()}

# Looping through the text, and pulling the interger values
# char2int dictionary then coverting to array

encoded = np.array([char2int[ch] for ch in text])
net = Network(chars,
              n_hidden=1024,
              n_layers=2,
              drop_prob=0.5,
              lr=0.001)

batch_size = 128
seq_length = 250
n_epochs = 70
train(net, encoded,
      epochs=n_epochs,
      batch_size=batch_size,
      seq_length=seq_length)


model_name = "rnn_70_epochs.net"
checkpoint = {"n_hidden": net.n_hidden,
              "n_layers": net.n_layers,
              "state_dict": net.state_dict(),
              "tokens": net.chars}

with open("weights/"+model_name, mode="wb") as f:
    torch.save(checkpoint, f)


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


def sample(net=net, size=2000, prime='Mulder', top_k=3):
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
