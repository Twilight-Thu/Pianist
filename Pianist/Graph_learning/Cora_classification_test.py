import time

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

# print(f"Dataset: {dataset}")
# print("======================")
# print(f"Number of graphs: {len(dataset)}")
# print(f"Number of features: {dataset.num_features}")
# print(f"Number of classes: {dataset.num_classes}")

data = dataset[0]
# print()
# print(data)
# print("==================================")
# print(f"Numbers of nodes: {data.num_nodes}")
# print(f"Numbers of edges: {data.num_edges}")
# print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
# print(f"Training label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}")
# print(f"Has isolated nodes: {data.has_isolated_nodes()}")
# print(f"Has self loops: {data.has_self_loops()}")

class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.fc1 = Linear(dataset.num_features, hidden_channels)
        self.fc2 = Linear(hidden_channels, dataset.num_classes)

    def forward(self, X):
        X = self.fc1(X)
        X = F.relu(X)
        X = F.dropout(X, p=0.5, training=self.training)
        X = self.fc2(X)
        return X

contrastive_model = MLP(hidden_channels=16)
# print(contrastive_model)
loss_func_1 = torch.nn.CrossEntropyLoss()
optimizer_1 = torch.optim.Adam(contrastive_model.parameters(), lr=0.01,
                               weight_decay=5e-4)

# def mlp_train():
#     contrastive_model.train()
#     optimizer_1.zero_grad()
#     train_out = contrastive_model(data.x)
#     loss = loss_func_1(train_out[data.train_mask], data.y[data.train_mask])
#     loss.backward()
#     optimizer_1.step()
#     return loss
#
# def mlp_test():
#     contrastive_model.eval()
#     eval_out = contrastive_model(data.x)
#     preds = eval_out.argmax(dim=1)
#     eval_correct = (preds[data.test_mask] == data.y[data.test_mask]).sum()
#     eval_acc = int(eval_correct) / int(data.test_mask.sum())
#     return eval_acc
#
# epochs = 200
# for epoch in range(epochs):
#     loss = mlp_train()
#     print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
#
# test_acc = mlp_test()
# print(f"Test accuracy: {test_acc:.4f}")

class GCN(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(dataset.num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, dataset.num_classes)

    def forward(self, X, edge_index):
        X = self.conv1(X, edge_index)
        X = F.relu(X)
        X = F.dropout(X, p=0.5, training=self.training)
        X = self.conv2(X, edge_index)

        return X

hidden_size = 16
GCN_model = GCN(hidden_size)
loss_func_2 = torch.nn.CrossEntropyLoss()
optimizer_2 = torch.optim.Adam(GCN_model.parameters(),lr=0.01,
                               weight_decay=5e-4)
def GCN_train(data):
    GCN_model.train()
    optimizer_2.zero_grad()
    X = GCN_model(data.x, data.edge_index)
    loss = loss_func_2(X[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer_2.step()
    return loss, X

def GCN_test():
    GCN_model.eval()
    X = GCN_model(data.x, data.edge_index)
    preds = X.argmax(dim=1)
    GCN_train_correct = (preds[data.train_mask] == data.y[data.train_mask]).sum()
    GCN_train_acc = int(GCN_train_correct) / int(data.train_mask.sum())
    return GCN_train_acc, X

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

# GCN_model.eval()
# X = GCN_model(data.x, data.edge_index)
# visualize(X, data.y)
epochs = 200
for epoch in range(epochs):
    loss, X = GCN_train(data)
    print(f"epoch: {epoch:03d}, loss: {loss:.4f}")
#
test_acc, X = GCN_test()
visualize(X, data.y)
print(f"Test accuracy: {test_acc:.4f}")

