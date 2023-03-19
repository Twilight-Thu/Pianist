import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import time

def visualize_graph(G, color):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()

def visualize_embeding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f"Epoch: {epoch}, Loss: {loss.item():.4f}", fontsize=16)
    plt.show()

dataset = KarateClub()
# print(f"Dataset: {dataset}")
# print("======================")
# print(f"Number of graphs: {len(dataset)}")
# print(f"Number of features: {dataset.num_features}")
# print(f"Number of classes: {dataset.num_classes}")

data = dataset[0]
# print(data)

edge_index = data.edge_index
# print(edge_index.t())

G = to_networkx(data, to_undirected=True)
# visualize_graph(G, color=data.y)

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)

        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, X, edge_index):
        h = self.conv1(X, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()

        out = self.classifier(h)

        return out, h

model = GCN()
print(model)

_, h = model(data.x, data.edge_index)
print(f"Embeding shape: {list(h.shape)}")

# visualize_embeding(h, color=data.y)

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100

def train(data):
    for epoch in range(epochs):
        out, h = model(data.x, data.edge_index)
        loss = loss_func(out[data.train_mask], data.y[data.train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            visualize_embeding(h, color=data.y, epoch=epoch, loss=loss)
            time.sleep(0.3)

train(data)
