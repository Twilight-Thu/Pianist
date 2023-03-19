import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

batch_size = 250

data_process = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        )
    ]
)

train_set = torchvision.datasets.CIFAR10(
    './dataset_cifar10',
    train=True,
    download=True,
    transform=data_process
)

test_set = torchvision.datasets.CIFAR10(
    './dataset_cifar10',
    train=False,
    download=True,
    transform=data_process
)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

class Basic_CNN(nn.Module):
    def __init__(self):
        super(Basic_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 12, 5)

        self.fc1 = nn.Linear(12 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 10)

    def forward(self, X):

        # conv1
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, kernel_size=2, stride=2)

        # conv2
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, kernel_size=2, stride=2)

        X = X.reshape(batch_size, 12 * 5 * 5)

        # fc1
        X = F.relu(self.fc1(X))

        # fc2
        X = F.relu(self.fc2(X))

        # fc3
        X = self.fc3(X)

        return X

# network = Basic_CNN()
# print(network)
# network.cuda()
#
# optimizer = optim.SGD(network.parameters(), lr=0.1)
# loss_func = nn.CrossEntropyLoss()
#
# def get_correct(preds, labels):
#     return preds.argmax(dim=1).eq(labels).sum().item()
#
# epochs = 10
#
# for epoch in range(epochs):
#     total_loss = 0.0
#     total_train_correct = 0
#
#     for batch in train_loader:
#         images, labels = batch
#
#         images = images.cuda()
#         labels = labels.cuda()
#
#         preds = network(images)
#         loss = loss_func(preds, labels)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#         total_train_correct += get_correct(preds, labels)
#
#     print("epoch:", epoch,
#           "correct:", total_train_correct,
#           f"training accuracy", "%.3f" % (total_train_correct/len(train_set) * 100),
#           "%", f"total loss", "%.3f" % total_loss)

# test_loss = 0.0
# total_test_correct = 0
# for batch in test_loader:
#     images, labels = batch
#
#     images = images.cuda()
#     labels = labels.cuda()
#
#     preds = network(images)
#     loss = loss_func(preds, labels)
#
#     test_loss += loss.item()
#     total_test_correct += get_correct(preds, labels)
#
# print(f"test accuracy:", "%.3f" % (total_test_correct/len(test_set) * 100), "%")

class Basic_block(nn.Module):
    channel_expansion = 1   # output_channels / input_channels = channel_expansion

    def __init__(self, blk_in_channels, blk_mid_channels, stride=1):
        super(Basic_block, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=blk_in_channels,
            out_channels=blk_mid_channels,
            kernel_size=3,
            padding=1,
            stride=stride  # stride can be any value
        )

        self.bn1 = nn.BatchNorm2d(blk_mid_channels)

        self.conv2 = nn.Conv2d(
            in_channels=blk_mid_channels,
            out_channels=blk_mid_channels * self.channel_expansion,
            kernel_size=3,
            padding=1,
            stride=1,  # stride can't be updated
        )

        self.bn2 = nn.BatchNorm2d(blk_mid_channels * self.channel_expansion)

        # shortcut connection
        # if the shape of the X is same as conv2/bn2's output,
        # we can add it directly
        # else we should transform the X on shortcut connection by conv/bn
        if stride != 1 or blk_in_channels != self.channel_expansion * blk_mid_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=blk_in_channels,
                    out_channels=blk_mid_channels * self.channel_expansion,
                    kernel_size=1,
                    padding=0,
                    stride=stride
                ),
                nn.BatchNorm2d(blk_mid_channels * self.channel_expansion)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, X):
        # conv1
        out = self.conv1(X)
        out = self.bn1(out)
        out = F.relu(out)

        # conv2
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        # shortcut connection
        out += self.shortcut(X)
        out = F.relu(out)

        return out

class BottleneckBlock(nn.Module):
    channel_expansion = 4  # channel_expansion = output_channels / input_channels

    def __init__(self, blk_in_channels, blk_mid_channels, stride=1):
        super(BottleneckBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=blk_in_channels,
            out_channels=blk_mid_channels,
            kernel_size=1,
            padding=0,
            stride=1
        )

        self.bn1 = nn.BatchNorm2d(blk_mid_channels)

        self.conv2 = nn.Conv2d(
            in_channels=blk_mid_channels,
            out_channels=blk_mid_channels,
            kernel_size=3,
            padding=1,
            stride=stride
        )

        self.bn2 = nn.BatchNorm2d(blk_mid_channels)

        self.conv3 = nn.Conv2d(
            in_channels=blk_mid_channels,
            out_channels=blk_mid_channels * self.channel_expansion,
            kernel_size=1,
            padding=0,
            stride=1
        )

        self.bn3 = nn.BatchNorm2d(blk_mid_channels * self.channel_expansion)

        if stride != 1 or blk_in_channels != blk_mid_channels * self.channel_expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                  in_channels=blk_in_channels,
                  out_channels=blk_mid_channels * self.channel_expansion,
                  kernel_size=1,
                  padding=0,
                  stride=stride
                ),
                nn.BatchNorm2d(blk_mid_channels * self.channel_expansion)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, X):
        # out1
        out = self.conv1(X)
        out = self.bn1(out)
        out = F.relu(out)

        # out2
        out = self.conv2(X)
        out = self.bn2(out)
        out = F.relu(out)

        # out3 + shortcut
        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(X)
        out = F.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()

        self.residual_layers = 4    # every residual layers include many blocks
        self.blk1_in_channels = 32  # 此处应填64，但由于训练时间长故减半
        self.blk_mid_channels = [32, 64, 128, 256]  # [64, 128, 256, 512]
        self.blk_channels = [self.blk1_in_channels] + self.blk_mid_channels
        self.blk_stride = [1, 2, 2, 2]  # 每个residual的stride

        self.blk_channel_expansion = block.channel_expansion

        # The first conv layer
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=self.blk_channels[0],
            kernel_size=3,
            padding=1,
            stride=1
        )
        self.bn1 = nn.BatchNorm2d(self.blk_channels[0])

        # Residual layers
        self.layers = nn.Sequential()
        for i in range(self.residual_layers):
            blk_in_channels = self.blk_channels[i] if i==0 else self.blk_channels[i]*block.channel_expansion
            blk_mid_channels = self.blk_channels[i + 1]
            self.layers.add_module(
                f"residule_layer{i}",
                self._make_layer(
                    block=block,
                    blk_in_channels=blk_in_channels,
                    blk_mid_channels=blk_mid_channels,
                    num_blocks=num_blocks[i],
                    stride=self.blk_stride[i]
                )
            )

        # The fully connected layer
        self.linear = nn.Linear(
            in_features=self.blk_channels[self.residual_layers] * block.channel_expansion,
            out_features=num_classes
        )

    def _make_layer(self, block, blk_in_channels, blk_mid_channels, num_blocks, stride):
        block_list = []
        stride_list = [stride] + [1] * (num_blocks - 1)

        for block_idx in range(num_blocks):
            if block_idx != 0:
                blk_in_channels = blk_mid_channels * block.channel_expansion
            block_list.append(
                block(
                    blk_in_channels=blk_in_channels,
                    blk_mid_channels=blk_mid_channels,
                    stride=stride_list[block_idx]
                )
            )
        return nn.Sequential(*block_list)

    def forward(self, X):
        out = self.conv1(X)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.layers(out)

        out = F.avg_pool2d(out, 4)

        out = out.reshape((out.shape[0], -1))
        out = self.linear(out)

        return out

num_classes = 10
def ResNet18():
    return ResNet(block=Basic_block, num_blocks=[2, 2, 2, 2], num_classes=num_classes)

def ResNet34():
    return ResNet(block=Basic_block, num_blocks=[3, 4, 6, 3], num_classes=num_classes)

def ResNet50():
    return ResNet(block=BottleneckBlock, num_blocks=[3, 4, 6, 3], num_classes=num_classes)

def ResNet101():
    return ResNet(block=BottleneckBlock, num_blocks=[3, 4, 23, 3], num_classes=num_classes)

def ResNet152():
    return ResNet(block=BottleneckBlock, num_blocks=[3, 8, 36, 3], num_classes=num_classes)

# 先测试一下网络结构和输出结果的形状是不是跟预想的一样
# def test_output_shape():
#     net = ResNet18()
#     x = torch.randn(batch_size, 3, 32, 32)
#     y = net(x)
#     print(net)
#     print("")
#     print(y.shape)
# test_output_shape()

net = ResNet18()
net.cuda()

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

def get_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

total_epochs = 5

# for epoch in range(total_epochs):
#     total_loss = 0.0
#     total_train_correct = 0
#
#     for batch in train_loader:
#
#         images, labels = batch
#         images = images.cuda()
#         labels = labels.cuda()
#
#         preds = net(images)
#         loss = loss_func(preds, labels)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss
#         total_train_correct += get_correct(preds, labels)
#
#     print("epoch:", epoch,
#           "correct times:", total_train_correct,
#           "training accuracy:", "%.3f" % (total_train_correct / len(train_set) * 100),
#           "total loss:", "%.3f" % total_loss)
#
# torch.save(net.cpu(), "resnet18.pt")

net2 = ResNet18()
net2 = torch.load("resnet18.pt")
net2.cuda()

total_test_correct = 0

with torch.no_grad():
    for batch in test_loader:
        images, labels = batch
        images = images.cuda()
        labels = labels.cuda()

        preds = net2(images)
        total_test_correct += get_correct(preds, labels)
    print("test accuracy:", "%.3f" % (total_test_correct / len(test_set) * 100))