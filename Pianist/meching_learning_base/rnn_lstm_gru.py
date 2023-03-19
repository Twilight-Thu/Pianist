import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(1)

batch_size_train = 100
batch_size_test = 10000

data_process = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.137,), (0.3081,)
        )
    ]
)

train_set = torchvision.datasets.MNIST(
    './dataset_mnist',
    train=True,
    download=True,
    transform=data_process
)

test_set = torchvision.datasets.MNIST(
    './dataset_mnist',
    train=False,
    download=True,
    transform=data_process
)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=True)

time_step = 28  # the length of the sequence
input_size = 28  # the size of input
hidden_size = 64  # the feature size of hidden layer
num_layers = 1  # the number of reccurent layer

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(  # use the original rnn
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Ture: batch_size is the first dimension of input and output, eg (batch_size, time_step, input_size)
        )
        self.out = nn.Linear(hidden_size, 10)

    def forward(self, X):
        # x = (batchsize, timestep, input_size)
        # r_out : the last reccurent layer's ouput feature h_t on every t, shape as (batchsize, timestep, outputsize)
        # h_state : include every last hidden state of this batch as (number_layers, batch, hidden_size)
        r_out, h_state = self.rnn(X, None)  # initial the hidden state as None

        out = self.out(r_out[:, time_step - 1, :])
        assert (r_out[:, time_step - 1, :] == h_state[num_layers - 1]).prod()

        return out

# rnn = RNN()
# rnn.cuda()
# print(rnn)
#
# def get_correct(preds, labels):
#     return preds.argmax(dim=1).eq(labels).sum().item()
#
# epochs = 10
# lr = 0.002
#
# loss_func = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
#
# for epoch in range(epochs):
#
#     total_loss = 0.0
#     total_train_correct = 0
#
#     for step, batch in enumerate(train_loader):
#         images, labels = batch
#         images = images.cuda()
#         labels = labels.cuda()
#         images = images.view(batch_size_train, time_step, input_size)  # reshape the input shape
#
#         preds = rnn(images)
#         loss = loss_func(preds, labels)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#         total_train_correct += get_correct(preds, labels)
#
#     print(f"epoch[{epoch}] \t loss: {total_loss} \t training accuracy: "
#           f"{total_train_correct / len(train_set) * 100}%")

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.out = nn.Linear(hidden_size, 10)

    def forward(self, X):
        # X [batch_size, time_step, input_size]
        # r_out [batch_size, time_step, output_size]
        # h_n  [number of layers, batch_size, hidden_size]
        # h_c  [number of layers, batch_size, hidden_size]
        r_out, (h_n, h_c) = self.lstm(X, None)

        out = self.out(r_out[:, time_step - 1, :])
        assert (r_out[:, time_step - 1, :] == h_n[num_layers - 1]).prod()

        return out

# lstm = LSTM()
# lstm.cuda()
# print(lstm)
#
# def get_correct(preds, labels):
#     return preds.argmax(dim=1).eq(labels).sum().item()
#
# epochs = 10
# lr = 0.002
#
# loss_func = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)
#
# for epoch in range(epochs):
#
#     total_loss = 0.0
#     total_train_correct = 0
#
#     for step, batch in enumerate(train_loader):
#         images, labels = batch
#         images = images.cuda()
#         labels = labels.cuda()
#         images = images.view(batch_size_train, time_step, input_size)  # reshape the input shape
#
#         preds = lstm(images)
#         loss = loss_func(preds, labels)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#         total_train_correct += get_correct(preds, labels)
#
#     print(f"epoch[{epoch}] \t loss: {total_loss} \t training accuracy: "
#           f"{total_train_correct / len(train_set) * 100}%")

class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.out = nn.Linear(hidden_size, 10)

    def forward(self, X):
        # X: [batch, time_step, input_size]
        # r_out: [batch, time_step, output_size]
        # h_n: [number_of_layers, batch_size, hidden_state]
        # h_c: [number_of_layers, batch_size, hidden_state]
        r_out, h_n = self.gru(X, None)

        out = self.out(r_out[:, time_step - 1, :])
        assert (r_out[:, time_step - 1, :] == h_n[num_layers - 1]).prod()

        return out

gru = GRU()
gru.cuda()
print(gru)

def get_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

epochs = 10
lr = 0.002

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(gru.parameters(), lr=lr)

for epoch in range(epochs):

    total_loss = 0.0
    total_train_correct = 0

    for step, batch in enumerate(train_loader):
        images, labels = batch
        images = images.cuda()
        labels = labels.cuda()
        images = images.view(batch_size_train, time_step, input_size)  # reshape the input shape

        preds = gru(images)
        loss = loss_func(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_train_correct += get_correct(preds, labels)

    print(f"epoch[{epoch}] \t loss: {total_loss} \t training accuracy: "
          f"{total_train_correct / len(train_set) * 100}%")


