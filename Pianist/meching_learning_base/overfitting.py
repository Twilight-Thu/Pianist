import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim


torch.manual_seed(0)

n_samples = 20
# generate the train set
x = torch.unsqueeze(torch.linspace(-1, 1, n_samples), dim=1)
y = x + 0.3 * torch.normal(torch.zeros(n_samples, 1), torch.ones(n_samples, 1))

# generate the test set
test_x = torch.unsqueeze(torch.linspace(-1, 1, n_samples), dim=1)
test_y = test_x + 0.3 * torch.normal(torch.zeros(n_samples, 1), torch.ones(n_samples, 1))

# plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.5, label='train')
# plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.5, label='test')
# plt.legend(loc='upper left')
# plt.ylim((-2.5, 2.5))
# plt.show()

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(1, 300)
        self.fc2 = nn.Linear(300, 300)
        self.drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, X):
        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)
        X = F.relu(X)
        X = self.drop(X)
        X = self.fc3(X)

        return X

net = Network()
# print(net)

optimizer = optim.Adam(net.parameters(), lr=0.01)
loss_func = nn.MSELoss()
test_loss1 = []

plt.ion()

epochs = 500

for epoch in range(epochs):
    pred1 = net(x)
    loss1 = loss_func(pred1, y)

    optimizer.zero_grad()
    loss1.backward()
    optimizer.step()

    net.eval()
    # test
    test_pred1 = net(test_x)
    test_loss1.append(loss_func(test_pred1, test_y))

    if epoch % 20 == 0:
        print("epoch[", epoch, "]:  test loss ", test_loss1[-1])
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.3, label='train')
        plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.3, label='test')

        plt.plot(test_x.data.numpy(), test_pred1.data.numpy(), 'r-', lw=3, label='network1')
        plt.text(0, -1.2, 'network1 loss=%.4f' % loss_func(test_pred1, test_y).data.numpy(), fontdict={
            'size':20, 'color':'red'
        })
        plt.legend(loc='upper left')
        plt.ylim((-2.5, 2.5))
        plt.pause(0.1)

    net.train()

plt.ioff()
plt.show()

plot_loss_x = np.arange(0, 500)
plt.ylim((0, 1.45))
plt.plot(plot_loss_x, np.array(test_loss1))

