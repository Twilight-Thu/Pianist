import warnings
warnings.filterwarnings("ignore")
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset

x = torch.tensor([[2, 1], [5, 6], [3, 7], [12, 0]], dtype=torch.float)
y = torch.tensor([0, 1, 0, 1], dtype=torch.float)

edge_index = torch.tensor([[0, 1, 2, 0, 3],  # source
                          [1, 0, 1, 3, 2]], dtype=torch.long)  # end

data = Data(x=x, y=y, edge_index=edge_index)
# print(data)

df = pd.read_csv('yoochoose-clicks.dat', header=None)
df.columns = ['session_id', 'timestamp', 'item_id', 'category']

buy_df = pd.read_csv('yoochoose-buys.dat', header=None)
buy_df.columns = ['session_id', 'timestamp', 'item_id', 'price', 'quantity']

item_encoder = LabelEncoder()
df['item_id'] = item_encoder.fit_transform(df.item_id)
# print(df.head())

sampled_session_id = np.random.choice(df.session_id.unique(), 100000, replace=False)
df = df.loc[df.session_id.isin(sampled_session_id)]
# print(df.nunique())

df['label'] = df.session_id.isin(buy_df.session_id)
# print(df.head())

df_test = df[:100]
total_group = df_test.groupby('session_id')
print(tqdm(total_group))
# for session_id, group in tqdm(total_group):
#     print("session_id: ", session_id)
#     session_item_id = LabelEncoder().fit_transform(group.item_id)
#     print("session_item_id: ", session_item_id)
#     group = group.reset_index(drop=True)
#     group['session_item_id'] = session_item_id
#     print("group: ", group)
#     node_features = group.loc[group.session_id==session_id,["session_item_id",
#                                                             "item_id"]].sort_values(
#         "session_item_id"
#     ).item_id.drop_duplicates().values
#     node_features = torch.LongTensor(node_features).unsqueeze(1)
#     print("node_features: ", node_features)
#     target_nodes = group.session_item_id.values[1:]
#     source_nodes = group.session_item_id.values[:-1]
#     print("target_nodes: ", target_nodes)
#     print("source_nodes: ", source_nodes)
#     edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
#     x = node_features
#     y = torch.FloatTensor(group.label.values[0])
#     data = Data(x=x, edge_index=edge_index, y=y)
#     print("data: ", data)



