import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

batch_size_small = 10
batch_size_large = 100

data_process = torchvision.transforms.Compose(
[
    torchvision.transforms.ToTensor()
]
)

# train set
train_set = torchvision.datasets.CIFAR10(
    './dataset_cifar10',
    train=True,
    download=True,
    transform=data_process
)

# test set
test_set = torchvision.datasets.CIFAR10(
    './dataset_cifar10',
    train=False,
    download=True,
    transform=data_process
)

train_loader_small = torch.utils.data.DataLoader(train_set, batch_size=batch_size_large,
                                            shuffle=True)
test_loader_small = torch.utils.data.DataLoader(test_set, batch_size=batch_size_large,
                                           shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # input [32, 32, 3]      output [32, 32, 32]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, padding=1, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        # output [16, 16, 32]
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, padding=1, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        # output [32, 32, 32]
        self.bn3 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, padding=1, kernel_size=3)

        self.fc1 = nn.Linear(in_features=32*4*4, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=10)

    def forward(self, X):

        X = self.conv1(X)
        X = self.bn1(X)
        X = F.relu(X)
        X = F.max_pool2d(X, kernel_size=2, stride=2)

        X = self.conv2(X)
        X = self.bn2(X)
        X = F.relu(X)
        X = F.max_pool2d(X, kernel_size=2, stride=2)

        X = self.conv3(X)
        X = self.bn3(X)
        X = F.relu(X)
        X = F.max_pool2d(X, kernel_size=2, stride=2)

        X = X.reshape(batch_size_large, 32*4*4)

        X = self.fc1(X)
        X = F.relu(X)

        X = self.fc2(X)

        return X

network = CNN()
print(network)
network.cuda()

loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=0.1)

def get_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

epochs = 15

for epoch in range(epochs):
    total_loss = 0.0
    total_trian_correct = 0

    for batch in train_loader_small:
        images, labels = batch

        images = images.cuda()
        labels = labels.cuda()

        preds = network(images)
        loss = loss_func(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_trian_correct += get_correct(preds, labels)

    print("epoch:", epoch,
          "correct times:", total_trian_correct,
          f"training accuracy:", "%.3f" % (total_trian_correct/len(train_set) * 100),
          "%", "total_loss:", "%.3f" % total_loss)

torch.save(network.cpu(), 'cnn5.pt')

