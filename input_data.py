import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import zipfile
import math
from  Synthatic_graph_generator import Synthetic_data

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset):
    """the method will return adjacncy matrix, node features, nodes label, edges label and circules.
      None in case the data set does not come with the information"""
    if dataset =="IMDB":
        return IMDB()
    if dataset =="NELL":
        return NELL()
    elif dataset =="DBLP":
        return DBLP()

    elif dataset =="ACM":
        return ACM()
        
    elif dataset== "AMiner":
        return AMiner()

    elif dataset=="facebook_egoes":
        return facebook_egoes__dataset()
    elif dataset=="facebook_pages":
        return facebook_pages()
        
#Synthetic Datasets
    elif dataset== "grid":
        return Synthetic_data(dataset)

    elif dataset == "community":
        return Synthetic_data(dataset)

    elif dataset == "ego":
        return Synthetic_data(dataset)

    elif dataset == "lobster":
        return Synthetic_data(dataset)

    elif dataset == "karate":
        return build_karate_club_graph()

    

    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features, None, None, None

def build_karate_club_graph():
    # All 78 edges are stored in two numpy arrays. One for source endpoints
    # while the other for destination endpoints.
    src = np.array([1, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 9, 10, 10,
        10, 11, 12, 12, 13, 13, 13, 13, 16, 16, 17, 17, 19, 19, 21, 21,
        25, 25, 27, 27, 27, 28, 29, 29, 30, 30, 31, 31, 31, 31, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33,
        33, 33, 33, 33, 33, 33, 33, 33, 33, 33])
    dst = np.array([0, 0, 1, 0, 1, 2, 0, 0, 0, 4, 5, 0, 1, 2, 3, 0, 2, 2, 0, 4,
        5, 0, 0, 3, 0, 1, 2, 3, 5, 6, 0, 1, 0, 1, 0, 1, 23, 24, 2, 23,
        24, 2, 23, 26, 1, 8, 0, 24, 25, 28, 2, 8, 14, 15, 18, 20, 22, 23,
        29, 30, 31, 8, 9, 13, 14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30,
        31, 32])
    # Edges are directional in DGL; Make them bi-directional.
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    # Construct a DGLGraph
    adj = sp.csr_matrix((np.ones(u.shape,dtype="float32"),(u,v)))
    feature = sp.csr_matrix(np.ones(adj.shape,dtype="int16"))
    return adj, feature, None, None, None

def AMiner():
    ob = []
    with open("data/AMiner/paper_author_matrix.pickle", 'rb') as f:
            ob.append(pkl.load(f))

    adj = ob[0].tocsr()
    with open("data/AMiner/feature_matrix.pickle", 'rb') as f:
            ob.append(pkl.load(f))

    feature = ob[1].tocsr()
    to_ = math.floor(adj.shape[0]*.4)
    return adj, feature, None, None, None


def facebook_egoes__dataset():
    ob = []
    with open("data/facebook_matrix.pickle", 'rb') as f:
            ob.append(pkl.load(f))

    adj = ob[0].tocsr()
    with open("data/facebook_feature_matrix.pickle", 'rb') as f:
            ob.append(pkl.load(f))
    feature = ob[1].tocsr()

    with open("data/facebook_circle_dict.pickle", 'rb') as f:
            circules = pkl.load(f)

    return adj, feature, None , None, circules
#

def IMDB():
    obj = []

    adj_file_name = "data/IMDB/edges.pkl"


    with open(adj_file_name, 'rb') as f:
            obj.append(pkl.load(f))

    # merging diffrent edge type into a single adj matrix
    adj = sp.csr_matrix(obj[0][0].shape)
    for matrix in obj[0]:
        adj +=matrix

    matrix = obj[0]
    edge_labels = matrix[0] + matrix[1]
    edge_labels += (matrix[2] + matrix[3])*2

    node_label= []
    in_1 = matrix[0].indices.min()
    in_2 = matrix[0].indices.max()+1
    in_3 = matrix[2].indices.max()+1
    node_label.extend([0 for i in range(in_1)])
    node_label.extend([1 for i in range(in_1,in_2)])
    node_label.extend([2 for i in range(in_2, in_3)])


    obj = []
    with open("data/IMDB/node_features.pkl", 'rb') as f:
        obj.append(pkl.load(f))
    feature = sp.csr_matrix(obj[0])


    return adj, feature, node_label, edge_labels, None


def DBLP():
    obj = []

    adj_file_name = "data/DBLP/edges.pkl"


    with open(adj_file_name, 'rb') as f:
            obj.append(pkl.load(f))

    # merging diffrent edge type into a single adj matrix
    adj = sp.csr_matrix(obj[0][0].shape)
    for matrix in obj[0]:
        adj +=matrix

    matrix = obj[0]
    edge_labels = matrix[0] + matrix[1]
    edge_labels += (matrix[2] + matrix[3])*2

    node_label= []
    # in_1 = matrix[0].indices.min()
    # in_2 = matrix[0].indices.max()+1
    # in_3 = matrix[2].indices.max()+1
    in_1 = matrix[0].nonzero()[0].min()
    in_2 = matrix[0].nonzero()[0].max()+1
    in_3 = matrix[3].nonzero()[0].max()+1
    matrix[0].nonzero()
    node_label.extend([0 for i in range(in_1)])
    node_label.extend([1 for i in range(in_1,in_2)])
    node_label.extend([2 for i in range(in_2, in_3)])


    obj = []
    with open("data/DBLP/node_features.pkl", 'rb') as f:
        obj.append(pkl.load(f))
    feature = sp.csr_matrix(obj[0])


    return adj, feature, node_label, edge_labels, None

def ACM():
    obj = []


    with zipfile.ZipFile('data/ACM/ACM/ACM.zip', 'r') as zip_ref:
        zip_ref.extractall('data/ACM/')

    adj_file_name = "data/ACM/ACM/edges.pkl"

    with open(adj_file_name, 'rb') as f:
            obj.append(pkl.load(f))

    adj = sp.csr_matrix(obj[0][0].shape)
    for matrix in obj[0]:
        adj +=matrix
    matrix = obj[0]
    edge_labels = matrix[0] + matrix[1]
    edge_labels += (matrix[2] + matrix[3])*2

    node_label= []
    in_1 = matrix[0].indices.min()
    in_2 = matrix[0].indices.max()+1
    in_3 = matrix[2].indices.max()+1
    node_label.extend([0 for i in range(in_1)])
    node_label.extend([1 for i in range(in_1,in_2)])
    node_label.extend([2 for i in range(in_2, in_3)])


    obj = []
    with open("data/ACM/ACM/node_features.pkl", 'rb') as f:
        obj.append(pkl.load(f))
    feature = sp.csr_matrix(obj[0])


    index = -1

    return adj, feature, node_label, edge_labels, None
# def ACM():
#     obj = []
#
#
#     with zipfile.ZipFile('data/ACM/ACM/ACM.zip', 'r') as zip_ref:
#         zip_ref.extractall('data/ACM/')
#
#     adj_file_name = "data/ACM/ACM/edges.pkl"
#
#     with open(adj_file_name, 'rb') as f:
#             obj.append(pkl.load(f))
#
#     adj = sp.csr_matrix(obj[0][0].shape)
#     for matrix in obj[0]:
#         adj +=matrix
#
#     matrix = obj[0]
#     edge_labels = matrix[0] + matrix[1]
#     edge_labels += (matrix[2] + matrix[3])*2
#
#     node_label= []
#     in_1 = matrix[0].indices.min()
#     in_2 = matrix[0].indices.max()+1
#     in_3 = matrix[2].indices.max()+1
#     node_label.extend([0 for i in range(in_1)])
#     node_label.extend([1 for i in range(in_1,in_2)])
#     node_label.extend([2 for i in range(in_2, in_3)])
#
#
#     obj = []
#     with open("data/ACM/ACM/node_features.pkl", 'rb') as f:
#         obj.append(pkl.load(f))
#     feature = sp.csr_matrix(obj[0])
#
#
#     index = -1
#
#     return adj[:index,:index], feature[:index], node_label[:index], edge_labels[:index,:index], None

def facebook_pages():


    adj_file_name = "data/facebook_pages/edges.pickle"

    with open(adj_file_name, 'rb') as f:
        adj=pkl.load(f)



    adj_file_name = "data/facebook_pages/labels.pickle"
    with open(adj_file_name, 'rb') as f:
        node_label = pkl.load(f)
    node_label = [j for (i,j) in node_label]

    obj = []
    with open("data/facebook_pages/node_features.pickle", 'rb') as f:
        obj.append(pkl.load(f))
    feature = sp.csr_matrix(obj[0])

    return adj, feature, node_label, None, None

def NELL():
    A = []
    data_path = "data/NELL/"
    with open(data_path+'X.pkl', 'rb') as f:
        feature = pkl.load(f)
    with open(data_path+'test_A.pkl', 'rb') as f:
        A.extend(pkl.load(f))

    with open(data_path+'train_A.pkl', 'rb') as f:
        A.extend(pkl.load(f))
    with open(data_path+'val_A.pkl', 'rb') as f:
        A.extend(pkl.load(f))
    adj =  A[0]
    for a in A:
        adj = adj+a
    adj += adj.transpose()
    adj[adj>0] = 1
    index = np.where(adj.sum(0) > 0)


    return adj[index[1]][:,index[1]], feature[index[1],: ], None, None, None

if __name__ == '__main__':
    # NELL()
    # AMiner()
    ACM()

    IMDB()
    DBLP()

    facebook_pages()
    facebook_egoes__dataset()




