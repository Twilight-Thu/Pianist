import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


batch_size_train = 128
batch_size_test = 128

data_process = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ]
)

# load the train set
train_set = torchvision.datasets.MNIST(
    './dataset_mnist',
    train=True,
    download=False,
    transform=data_process
)

# load the test set
test_set = torchvision.datasets.MNIST(
    './dataset_mnist',
    train=False,
    download=True,
    transform=data_process
)
# print(train_set)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=False)

# sample = next(iter(train_set))
# image, label = sample
# plt.imshow(image.squeeze(), cmap="gray")
# plt.show()

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # input = [28, 28, 1] output = [24, 24, 6]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        # [12, 12, 6]
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # [8, 8, 12]
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        # [4, 4, 12]
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=192, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.fc3 = nn.Linear(in_features=60, out_features=10)

    def forward(self, X):
        X = self.conv1(X)
        X = F.relu(X)
        X = self.pool1(X)

        X = self.conv2(X)
        X = F.relu(X)
        X = self.pool2(X)

        X = X.reshape((X.shape[0], -1))
        X = self.fc1(X)
        X = self.fc2(X)
        X = self.fc3(X)
        return X

# net = Network()
# print(net)

# for name in net.state_dict():
#     print(name, '\t\t', net.state_dict()[name].shape)

# torch.set_grad_enabled(False)
network = Network()
# images, labels = batch
# print(images.shape)
# print(labels.shape)
#
# output = network(images)
# print(output.shape)
#
# output_prob = F.softmax(output, dim=1)
# scores, predict_class = torch.max(output_prob, dim=1)

# print("scores:\n", scores)
# print("predicted labels:\n", predict_class)
#
# print("actual labels:\n", labels)
# correct_predictions = (predict_class == labels).sum().item()
# print("correct_predictions: ", correct_predictions)
# print("accuracy: ", correct_predictions / batch_size_train)
# torch.set_grad_enabled(True)

loss_func = nn.CrossEntropyLoss()

def get_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

optimizer = optim.SGD(network.parameters(), lr=0.01)

#batch = next(iter(train_loader))
# images, labels = batch
#
# preds = network(images)
# loss = loss_func(preds, labels)
# print(loss)
# loss.backward()
# print("correct numbers:", get_correct(preds, labels))
# optimizer.step()

# send the same examples again and observe the change
# preds = network(images)
# loss = loss_func(preds, labels)
# print(loss)
# print("correct numbers:", get_correct(preds, labels))

epochs = 5
#batch = next(iter(train_loader))
for epoch in range(epochs):

    total_loss = 0.0
    total_train_correct = 0

    for batch in train_loader: # get a batch of train loader
        images, labels = batch
        preds = network(images)
        loss = loss_func(preds, labels)

        # set every batch's gradient to zero
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_train_correct += get_correct(preds, labels)

    print("epoch:", epoch,
        "correct times", total_train_correct,
        f"training accuracy:", "%.3f" % (total_train_correct / len(train_set) * 100), "%",
        "total loss", "%.3f" % total_loss)

with torch.no_grad():
    total_corr_number = 0
    for batch in test_loader:
        images, labels = batch
        preds = network(images)

        total_corr_number += get_correct(preds, labels)
    print(f"accuracy:", "%.3f" % (total_corr_number / len(test_set) * 100), "%")














