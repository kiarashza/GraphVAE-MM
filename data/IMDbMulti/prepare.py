import torch_geometric.utils.convert
from torch_geometric import datasets
from  networkx import convert_matrix
dataset = datasets.GEDDataset( root=, name=IMDBMulti)
import pickle

adjacency = []

for graph in dataset
    adjacency.append(convert_matrix.to_scipy_sparse_matrix(torch_geometric.utils.convert.to_networkx(graph)))

adjacency

filehandler = open("IMDbbinary", 'w')
pickle.dump( adjacency, open( "/media/kiarash/New Volume/gits/IMDBMulti.p", "wb" ) )