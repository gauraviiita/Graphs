from turtle import color, forward
import torch 
import numpy as np 
np.random.seed(0)

import networkx as nx
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size':24})

# Graph data
'''
There are three classic graph datasets. They represent networks of research papers, where each connection is a citation.

Cora: It consists of 2708 machine learning papers that belong to one of 7 categories.
Node features represent the presence (1) or absence (0) of 1433 words in a paper.

CiteSeer: It is a bigger but similar dataset of 3312 scientific papers to classify into one of 6 categories. 
Node features represent the presence (1) or absence (0) of 3703 words in a paper

PubMed: It is an even bigger dataset with 19717 scientific publications about diabetes from PubMed database, 
classfied into 3 categories. Node features are TF-IDF weighted word vector from a dictionary of 500 unique words.

'''

from torch_geometric.datasets import Planetoid

# import dataset from pytorch geometric
dataset = Planetoid(root='.',name='CiteSeer')

print(f'Number of graphs: {len(dataset)}')
print(f'Number of nodes: {dataset[0].x.shape[0]}')
print(f'Number of features:{dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
print(f'Has isolated nodes:{dataset[0].has_isolated_nodes()}')

data = dataset[0]

from torch_geometric.utils import degree
from collections import Counter

# Get list of degrees for each node
degrees = degree(dataset[0].edge_index[0]).numpy()

# count the number of nodes for each degree
numbers = Counter(degrees)

#Bar plot 
fig, ax = plt.subplots(figsize=(18, 6))
ax.set_xlabel('Node degree')
ax.set_ylabel('Number of nodes')
plt.bar(numbers.keys(),
        numbers.values(),
        color='#0A047A')

#plt.show()

# self attention
'''
self attention in GNNs relies on a simple idea, nodes should not all have the same importance.
We talk about self attention(and not just attention) because inputs are compared to each other.
This mechanism assigns a weighting factor (attention score) to each connection. Let's call alpha_ij the attention
score between the nodes i and j.

How can we calculate the attention scores? We could write a static formula, but there's a smarter
solution: we can learn their values with a neural network. There are three steps in this process.
1 Linear Transformation
2 Activation function
3 Softmax normalization

1 Linear transformation
We want to calculate the importance of each connection, so we need pairs of hidden vectors. An easy way to create
these pairs is to concatenate vectors from both nodes.

Only then can we apply a new linear transformation with a weight matrix Watt

2 Activation function
We are building a neural network, so the second step is to add an acitvation function. In this case, the
authors of the paper chose the LeakyReLU function

e_ij = LeakyReLU(alpha_ij)

3 softmax normalization - The output of our neural network is not normalized, which is a problem since we want to
compare these scores. To be able to say if node 2 is more important to node 1 than node 3 (alpha_12 > alpha_13),
we need to share the same scale.
A common way to do it with neural networks function. Here, we apply it to every neighboring node:


# Multi-head attention

This is only slightly surprising since we've been talking about self-attention a lot but, in reality
transformers are GNNs in disguise.This is why we can reuse some ideas from natural language processing.

There are two methods in multihead attention

Average- we sum the different h^k_i and normalize the result by number of attention heads n.

Concatenation: we concatenate the different h^k_i

In practice, we use the concatenation scheme when it's a hidden layer, and the average scheme when it's 
the last layer of the network.
'''

# Implementation

import torch.nn.functional as F 
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, GATv2Conv

class GCN(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1 = GCNConv(dim_in, dim_h)
        self.gcn2 = GCNConv(dim_h, dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)

    def forward(self, x, edge_index):
        h = F.dropout(x, p=0.5, training=self.training)
        h = self.gcn1(h, edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.gcn2(h, edge_index)
        return h, F.log_softmax(h, dim=1)

class GAT(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, heads=8):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                        lr=0.005,
                                        weight_decay=5e-4)
    
    def forward(self, x, edge_index):
        h = F.dropout(x, p=0.6, training=self.training)
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.gat2(h, edge_index)
        return h, F.log_softmax(h, dim=1)

def accuracy(pred_y, y):
    return ((pred_y == y).sum()/len(y)).item()

def train(model, data):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = model.optimizer
    epochs = 200

    model.train()
    for epoch in range(epochs+1):
        optimizer.zero_grad()
        _, out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # validation
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
        val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])

        if (epoch % 10 == 0):
            print(f'Epoch {epoch:>3} | train loss: {loss:.3f} | train Acc:'
            f'{acc*100:>6.2f}% | Val Loss: {val_loss:.2f}|'
            f'Val Acc: {val_acc*100:.2f}%')
    return model

@torch.no_grad()
def test(model, data):
    model.eval()
    _, out = model(data.x, data.edge_index)
    acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    return acc

gcn = GCN(dataset.num_features, 16, dataset.num_classes)
print(gcn)
train(gcn, data)
acc = test(gcn, data)
print(f'\nGCN test accuracy: {acc*100:.2f}%\n')

gat = GAT(dataset.num_features, 8, dataset.num_classes)
print(gat)
train(gat, data)
acc = test(gat, data)
print(f'\n GAT test accuracy: {acc*100:.2f}%\n')


'''
This experiment is not super rigorous: we'd need to repeat it n times and take the average accuracy with a 
standard deviation as the final result.
We can see in thsi example that the GAT outperform the GCn in terms of accuracy. 
'''

untrained_gat = GAT(dataset.num_features, 8, dataset.num_classes)

h, _ = untrained_gat(data.x, data.edge_index)

tsne = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(h.detach())

plt.figure(figsize=(10, 10))
plt.axis('off')
plt.scatter(tsne[:,0], tsne[:,1], s=50, c=data.y)
#plt.show()

'''

Now use trained networks'''
h,_ = gat(data.x, data.edge_index)

tsne = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(h.detach())

plt.figure(figsize=(10, 10))
plt.axis('off')
plt.scatter(tsne[:,0], tsne[:,1],s=50, c=data.y)
#plt.show()

# lets calculate the model's accuracy for each degree.

from torch_geometric.utils import degree
# get model's classifications
_, out = gat(data.x, data.edge_index)

#calculate the degree of each node
degrees = degree(data.edge_index[0]).numpy()

accuracies = []
sizes = []

# accuracy for degrees between 0 and 5
for i in range(0,6):
    mask = np.where(degree==i)[0]
    accuracies.append(accuracy(out.argmax(dim=1)[mask], data.y[mask]))
    sizes.append(len(mask))

# accuracy for degrees > 5
mask = np.where(degrees > 5)[0]
accuracies.append(accuracy(out.argmax(dim=1)[mask], data.y[mask]))
sizes.append(len(mask))

# bar plot
fig, ax = plt.subplots(figsize=(18, 9))
ax.set_xlabel('Node degree')
ax.set_ylabel('Accuracy score')
ax.set_facecolor('#EFEEEA')
plt.bar(['0', '1', '2', '3', '4', '5', '>5'],
            accuracies, color='#0A047A')

for i in range(0, 7):
    plt.text(i, accuracies[i]//2, sizes[i], ha='center', color='white')

plt.show()