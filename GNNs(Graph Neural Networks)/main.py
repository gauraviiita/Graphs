# This file contains the implementaion of the graph neural networks using geometric pytorch library. 
# import the libraries
from filecmp import cmp
import nntplib
from turtle import forward
import torch 
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# graph data

from torch_geometric.datasets import KarateClub
#import dataset from pytorch gveometric
dataset = KarateClub()
print(dataset)
print(['-----------------'])
print(f'Number of graphs:{len(dataset)}')
print(f'Number of features:{dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

print(f'Graph:{dataset[0]}')


data = dataset[0]
print(f'x = {data.x.shape}')
print(data.x)
#here the node feature matrix x is actually an identity matrix. 

print(f'edge_index={data.edge_index.shape}')
print(data.edge_index)

from torch_geometric.utils import to_dense_adj

A = to_dense_adj(data.edge_index)[0].numpy().astype(int)
print(f'A={A.shape}')
print(A)

# print the labels
print(f'y={data.y.shape}')
print(data.y)

#print the train mask
print(f'train_mask = {data.train_mask.shape}')
print(data.train_mask)

print(f'Edges are directed: {data.is_directed()}')
print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
print(f'Graph has loops:{data.has_self_loops()}')

'''
to_networkx use to convert your data instance into a nextworkx.Graph to visualize the graph.
'''

from torch_geometric.utils import to_networkx

G = to_networkx(data, to_undirected=True)
plt.figure(figsize=(12, 12))
plt.axis('off')
nx.draw_networkx(G,
    pos=nx.spring_layout(G, seed=0),
    with_labels=True,
    node_size = 800,
    node_color=data.y,
    cmap='hsv',
    vmin=2,
    vmax=3,
    width=0.8,
    edge_color='grey',
    font_size=14
)
#plt.show()

print('second part started ------------------------------')
# a numpy array with integers instead of floats
X = data.x.numpy().astype(int)

print(f'X={X.shape}')
print(X)

# now we need W, the learnable weight matrix of our graph convolutional layer.

W = np.identity(X.shape[0], dtype=int)
print(f'W={W.shape}')
print(W)

'''
we can apply the weights in W to every feature vector by calculating WX, So we look the neighbor 
nodes by using the adjacency matrix.
'''

print(f'A={A.shape}')
print(A)

'''
A is symmetric bcz edges in this graph are bidirectional. This would not be the case if connections were unidirectional.
This is why we use A^TX and not AX in general in order to obtain the feature vectors of the neighboring nodes.
This operation does not only select the relevant vectors, it also sums them in the process.

We just miss the central node itself. which can be fixed by adding a connection to itself(self loops) in the 
adjacency matrix A' = A + I
'''

A_tilde = A + np.identity(A.shape[0], dtype=int)
print(f'\nA_tilde={A_tilde.shape}')
print(A_tilde)

'''
A'^TX selects the feature vectors of the neighboring nodes (and the node itself) and WX applies
the weights in W to every feature vector in the graph. We can combine both experssions to apply
these weights to the feature vectors of the neighboring nodes (and the node itself) with A'^T X W^T.

H = A'^T X W^T
'''

H = A_tilde.T@X@W.T

print(f'H=A_tilde.T@X@W.T{H.shape}')
print(H)

'''
Now we would like to normalize H by the number of neighbors as seen previously. We can use the 
degree matrix D that counts the number of neighbors for each node. In our case We want the matrix D', based
on A' instead of A.
'''

D = np.zeros(A.shape, dtype=int)
np.fill_diagonal(D,A.sum(axis=0))
print(f'D={D.shape}')
print(D)

D_tilde = np.zeros(D.shape, dtype=int)
np.fill_diagonal(D_tilde, A_tilde.sum(axis=0))

print(f'\nD_tilde={D_tilde.shape}')
print(D_tilde)

'''
We could use D to eather
* D'-1A' to normalize every row in A'.
* A'D'-1 to normalize evey column in A'.

    In our case, A' is symmetric so the results would be the equivalent. We can thus translate the normalized
    operation as follows:
    H = D'-1 A'T X WT
'''

D_inv = np.linalg.inv(D_tilde)
print(f'D_inv={D_inv.shape}')
print(D_inv)

H = D_inv@A_tilde.T@X@W.T
print(f'\nH = D_inv@A.T@X@W.T {H.shape}')
print(H)

"""
Graph neural network

one of the simplest GNN is called GCNConv in PyG for Graph Convolutional Network.The main idea 
is that feature vectors fraom nodes with a lot of neighbors will spread very easily, unlike ones from
more isolated nodes. 
"""

D_inv12 = np.linalg.inv(D_tilde)
np.fill_diagonal(D_inv12, 1/(D_tilde.diagonal()**0.5))

#new H
H = D_inv12@A_tilde.T@D_inv12@X@W.T

print(f'\nH = D_inv12@A.T@D_inv12@X@W.T {H.shape}')
print(H)

'''
however, edges are not always bidirectional so nodes can have different numbers of neighbors. This is why the 
result of this operation is different from the one we designed. It is clever trick to take into
account another difference between the graph and images.

X = H(X) -> ReLU(H(X)) - > Z(ReLU(H(X)))


'''

#start train of graph neural networks using geometric graph

import torch 
from torch.nn import Linear
from torch_geometric.nn import GCNConv

class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn = GCNConv(dataset.num_features, 3)
        self.out = Linear(3, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.gcn(x, edge_index)
        embedding = torch.relu(h)
        z = self.out(embedding)
        return h, embedding, z

model = GNN()
print(model)

# print number of parameters model has.
print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

# calculate accuracy

def accuracy(pred_y, y):
    return (pred_y==y).sum()/len(y)

# data for animations
embeddings = []
losses = []
accuracies = []
outputs = []

#training loop
for epoch in range(201):
    #clear gradients
    optimizer.zero_grad()

    #forward pass
    h, embedding, z = model(data.x, data.edge_index)

    #calculate loss
    loss = criterion(z, data.y)

    # calculate accuracy
    acc = accuracy(z.argmax(dim=1), data.y)

    # compute gradients
    loss.backward()

    # tune parameters
    optimizer.step()

    # store dat afor animations
    embeddings.append(embedding)
    losses.append(loss)
    accuracies.append(acc)
    outputs.append(z.argmax(dim=1))

    # print metrics evey 10 epochs
    if epoch % 10 == 0:
        print(f'Epoch{epoch:>3} | Loss:{loss:.2f} | Acc:{acc*100:.2f}%')


'''
Visualization by animation 
'''

#%%capture 
from IPython.display import HTML
from matplotlib import animation
plt.rcParams["animation.bitrate"] = 3000

def animate(i):
    G = to_networkx(data, to_undirected=True)
    nx.draw_networkx(G,
        pos=nx.spring_layout(G, seed=0),
        with_labels=True,
        node_size=800,
        node_color=outputs[i],
        cmap="hsv",
        vmin=2,
        vmax=3,
        width=0.8,
        edge_color='grey',
        font_size=14
    )
    plt.title(f'Epoch{i} | Loss:{losses[i]:.2f}| Acc:{accuracies[i]*100:.2f}%', fontsize=18, pad=20)

fig = plt.figure(figsize=(12, 12))
plt.axis("off")

anim = animation.FuncAnimation(fig, animate,  
        np.arange(0, 200, 10), interval=500, repeat=True)
#html = HTML(anim.to_html5_video())

#display(html)

print(f'final embeddings = {embedding.shape}')
print(embedding)


embed = embeddings[0].detach().cpu().numpy()

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.patch.set_alpha(0)
plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
ax.scatter(embed[:, 0], embed[:, 1], embed[:, 2],
           s=200, c=data.y, cmap="hsv", vmin=-2, vmax=3)

#plt.show()

#%capture

def animate(i):
    embed = embeddings[i].detach().cpu().numpy()
    ax.clear()
    ax.scatter(embed[:, 0], embed[:, 1], embed[:, 2],
           s=200, c=data.y, cmap="hsv", vmin=-2, vmax=3)
    plt.title(f'Epoch {i} | Loss: {losses[i]:.2f} | Acc: {accuracies[i]*100:.2f}%',
              fontsize=18, pad=40)

fig = plt.figure(figsize=(12, 12))
plt.axis('off')
ax = fig.add_subplot(projection='3d')
plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)

anim = animation.FuncAnimation(fig, animate, \
              np.arange(0, 200, 10), interval=800, repeat=True)
#html = HTML(anim.to_html5_video())