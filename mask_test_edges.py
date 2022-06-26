import numpy as np
import pylab as p
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix,average_precision_score
np.random.seed(0)

def roc_auc_estimator_onGraphList(pos_edges, negative_edges, reconstructed_adj, origianl_agjacency):
    prediction = []
    true_label = []
    for i,_ in enumerate(reconstructed_adj):
        for edge in pos_edges[i]:
            prediction.append(reconstructed_adj[i][edge[0],edge[1]])
            true_label.append(origianl_agjacency[i][edge[0], edge[1]])

        for edge in negative_edges[i]:
            prediction.append(reconstructed_adj[i][edge[0], edge[1]])
            true_label.append(origianl_agjacency[i][edge[0], edge[1]])

    pred = [1 if x>.5 else 0 for x in prediction]
    auc = roc_auc_score(y_score= prediction, y_true= true_label)
    acc = accuracy_score(y_pred= pred, y_true= true_label, normalize= True)
    ap=average_precision_score(y_score= prediction, y_true= true_label)
    cof_mtx = confusion_matrix(y_true=true_label, y_pred=pred)
    return auc , acc,ap, cof_mtx

def roc_auc_estimator(pos_edges, negative_edges, reconstructed_adj, origianl_agjacency):
    prediction = []
    true_label = []
    if type(pos_edges) == list or type(pos_edges) ==np.ndarray:
        for edge in pos_edges:
            prediction.append(reconstructed_adj[edge[0],edge[1]])
            true_label.append(origianl_agjacency[edge[0], edge[1]])

        for edge in negative_edges:
            prediction.append(reconstructed_adj[edge[0], edge[1]])
            true_label.append(origianl_agjacency[edge[0], edge[1]])
    else:
        prediction = list(reconstructed_adj.reshape(-1))
        true_label = list(np.array(origianl_agjacency.todense()).reshape(-1))
    pred = np.array(prediction)
    pred[pred>.5] = 1
    pred[pred < .5] = 0
    pred = pred.astype(int)
    # pred = [1 if x>.5 else 0 for x in prediction]

    auc = roc_auc_score(y_score= prediction, y_true= true_label)
    acc = accuracy_score(y_pred= pred, y_true= true_label, normalize= True)
    ap=average_precision_score(y_score= prediction, y_true= true_label)
    cof_mtx = confusion_matrix(y_true=true_label, y_pred=pred)
    return auc , acc,ap, cof_mtx

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    # assert np.diag(adj.todense()).sum() == 0
    assert adj.diagonal().sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    index = list(range(train_edges.shape[0]))
    np.random.shuffle(index)
    train_edges_true = train_edges[index[0:num_val]]

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue

        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_i, idx_j], np.array(test_edges_false)):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    train_edges_false = []
    while len(train_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_i, idx_j], np.array(val_edges_false)):
            continue
        if ismember([idx_i, idx_j], np.array(test_edges_false)):
            continue
        if train_edges_false:
            if ismember([idx_j, idx_i], np.array(train_edges_false)):
                continue
        train_edges_false.append([idx_i, idx_j])
    # print(test_edges_false)
    # print(val_edges_false)
    # print(test_edges)
    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    ignore_edges_inx = [list(np.array(val_edges_false)[:,0]),list(np.array(val_edges_false)[:,1])]
    ignore_edges_inx[0].extend(val_edges[:,0])
    ignore_edges_inx[1].extend(val_edges[:,1])
    import copy

    val_edge_idx = copy.deepcopy(ignore_edges_inx)
    ignore_edges_inx[0].extend(test_edges[:, 0])
    ignore_edges_inx[1].extend(test_edges[:, 1])
    ignore_edges_inx[0].extend(np.array(test_edges_false)[:, 0])
    ignore_edges_inx[1].extend(np.array(test_edges_false)[:, 1])

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, list(train_edges_true), train_edges_false,ignore_edges_inx, val_edge_idx
