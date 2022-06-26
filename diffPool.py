
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import block_diag
import dgl as dgl

import dgl.function as fn


class EntropyLoss(nn.Module):
    # Return Scalar
    def forward(self, adj, anext, s_l):
        entropy = (torch.distributions.Categorical(
            probs=s_l).entropy()).sum(-1).mean(-1)
        assert not torch.isnan(entropy)
        return entropy


class LinkPredLoss(nn.Module):

    def forward(self, adj, anext, s_l):
        link_pred_loss = (
            adj - s_l.matmul(s_l.transpose(-1, -2))).norm(dim=(1, 2))
        link_pred_loss = link_pred_loss / (adj.size(1) * adj.size(2))
        return link_pred_loss.mean()

class Bundler(nn.Module):
    """
    Bundler, which will be the node_apply function in DGL paradigm
    """

    def __init__(self, in_feats, out_feats, activation, dropout, bias=True):
        super(Bundler, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_feats * 2, out_feats, bias)
        self.activation = activation

        nn.init.xavier_uniform_(self.linear.weight,
                                gain=nn.init.calculate_gain('relu'))

    def concat(self, h, aggre_result):
        bundle = torch.cat((h, aggre_result), 1)
        bundle = self.linear(bundle)
        return bundle

    def forward(self, node):
        h = node.data['h']
        c = node.data['c']
        bundle = self.concat(h, c)
        bundle = F.normalize(bundle, p=2, dim=1)
        if self.activation:
            bundle = self.activation(bundle)
        return {"h": bundle}

class Aggregator(nn.Module):
    """
    Base Aggregator class. Adapting
    from PR# 403
    This class is not supposed to be called
    """

    def __init__(self):
        super(Aggregator, self).__init__()

    def forward(self, node):
        neighbour = node.mailbox['m']
        c = self.aggre(neighbour)
        return {"c": c}

    def aggre(self, neighbour):
        # N x F
        raise NotImplementedError


class MeanAggregator(Aggregator):
    '''
    Mean Aggregator for graphsage
    '''

    def __init__(self):
        super(MeanAggregator, self).__init__()

    def aggre(self, neighbour):
        mean_neighbour = torch.mean(neighbour, dim=1)
        return mean_neighbour


class MaxPoolAggregator(Aggregator):
    '''
    Maxpooling aggregator for graphsage
    '''

    def __init__(self, in_feats, out_feats, activation, bias):
        super(MaxPoolAggregator, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)
        self.activation = activation
        # Xavier initialization of weight
        nn.init.xavier_uniform_(self.linear.weight,
                                gain=nn.init.calculate_gain('relu'))

    def aggre(self, neighbour):
        neighbour = self.linear(neighbour)
        if self.activation:
            neighbour = self.activation(neighbour)
        maxpool_neighbour = torch.max(neighbour, dim=1)[0]
        return maxpool_neighbour


class LSTMAggregator(Aggregator):
    '''
    LSTM aggregator for graphsage
    '''

    def __init__(self, in_feats, hidden_feats):
        super(LSTMAggregator, self).__init__()
        self.lstm = nn.LSTM(in_feats, hidden_feats, batch_first=True)
        self.hidden_dim = hidden_feats
        self.hidden = self.init_hidden()

        nn.init.xavier_uniform_(self.lstm.weight,
                                gain=nn.init.calculate_gain('relu'))

    def init_hidden(self):
        """
        Defaulted to initialite all zero
        """
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def aggre(self, neighbours):
        '''
        aggregation function
        '''
        # N X F
        rand_order = torch.randperm(neighbours.size()[1])
        neighbours = neighbours[:, rand_order, :]

        (lstm_out, self.hidden) = self.lstm(neighbours.view(neighbours.size()[0],
                                                            neighbours.size()[
            1],
            -1))
        return lstm_out[:, -1, :]

    def forward(self, node):
        neighbour = node.mailbox['m']
        c = self.aggre(neighbour)
        return {"c": c}

class GraphSageLayer(nn.Module):
    """
    GraphSage layer in Inductive learning paper by hamilton
    Here, graphsage layer is a reduced function in DGL framework
    """

    def __init__(self, in_feats, out_feats, activation, dropout,
                 aggregator_type, bn=True, bias=True):
        super(GraphSageLayer, self).__init__()
        self.use_bn = bn
        self.bundler = Bundler(in_feats, out_feats, activation, dropout,
                               bias=bias)
        self.dropout = nn.Dropout(p=dropout)

        if aggregator_type == "maxpool":
            self.aggregator = MaxPoolAggregator(in_feats, in_feats,
                                                activation, bias)
        elif aggregator_type == "lstm":
            self.aggregator = LSTMAggregator(in_feats, in_feats)
        else:
            self.aggregator = MeanAggregator()

    def forward(self, g, h):
        h = self.dropout(h)
        g.ndata['h'] = h
        if self.use_bn and not hasattr(self, 'bn'):
            device = h.device
            self.bn = torch.nn.LayerNorm(h.size()[-1], elementwise_affine=False)
            # self.bn = nn.BatchNorm1d(h.size()[1]).to(device)
        g.update_all(fn.copy_src(src='h', out='m'), self.aggregator,
                     self.bundler)
        if self.use_bn:
            h = self.bn(h)
        h = g.ndata.pop('h')
        return h

class GraphGCN(nn.Module):
    """
    Grahpsage network that concatenate several graphsage layer
    """

    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation,
                 dropout, aggregator_type):
        super(GraphGCN, self).__init__()
        self.layers = nn.ModuleList()
        self.NORM_layers = nn.ModuleList()

        # input layer
        self.layers.append(dgl.nn.pytorch.conv.GraphConv(in_feats, n_hidden, activation = activation, bias=True,weight=True))
        self.NORM_layers.append(torch.nn.LayerNorm(n_hidden,elementwise_affine=False))
        # hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(dgl.nn.pytorch.conv.GraphConv(n_hidden, n_hidden, activation = activation,
                                                             bias=True,weight=True))
            self.NORM_layers.append(torch.nn.LayerNorm(n_hidden,elementwise_affine=False))
        # output layer
        self.layers.append(dgl.nn.pytorch.conv.GraphConv(n_hidden, n_classes, activation =activation,
                                                         bias=True,weight=True))
        self.NORM_layers.append(torch.nn.LayerNorm(n_classes,elementwise_affine=False))

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            h= self.NORM_layers[i](h)
        return h


class GraphSage(nn.Module):
    """
    Grahpsage network that concatenate several graphsage layer
    """

    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation,
                 dropout, aggregator_type):
        super(GraphSage, self).__init__()
        self.layers = nn.ModuleList()
        self.NORM_layers = nn.ModuleList()

        # input layer
        self.layers.append(GraphSageLayer(in_feats, n_hidden, activation, dropout,
                                          aggregator_type))
        self.NORM_layers.append(torch.nn.LayerNorm(n_hidden,elementwise_affine=False))

        # hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(GraphSageLayer(n_hidden, n_hidden, activation,
                                              dropout, aggregator_type))
            self.NORM_layers.append(torch.nn.LayerNorm(n_hidden,elementwise_affine=False))
        # output layer
        self.layers.append(GraphSageLayer(n_hidden, n_classes, None,
                                          dropout, aggregator_type))
        self.NORM_layers.append(torch.nn.LayerNorm(n_classes,elementwise_affine=False))

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            h= self.NORM_layers[i](h)
        return h


class DiffPoolBatchedGraphLayer(nn.Module):

    def __init__(self, input_dim, assign_dim, output_feat_dim,
                 activation, dropout, aggregator_type, link_pred):
        super(DiffPoolBatchedGraphLayer, self).__init__()
        self.embedding_dim = input_dim
        self.assign_dim = assign_dim
        self.hidden_dim = output_feat_dim
        self.link_pred = link_pred
        self.feat_gc = GraphSageLayer(
            input_dim,
            output_feat_dim,
            activation,
            dropout,
            aggregator_type)
        self.pool_gc = GraphSageLayer(
            input_dim,
            assign_dim,
            activation,
            dropout,
            aggregator_type)
        self.reg_loss = nn.ModuleList([])
        self.loss_log = {}
        self.reg_loss.append(EntropyLoss())

    def forward(self, g, h):
        feat = self.feat_gc(g, h)  # size = (sum_N, F_out), sum_N is num of nodes in this batch
        device = feat.device
        assign_tensor = self.pool_gc(g, h)  # size = (sum_N, N_a), N_a is num of nodes in pooled graph.
        assign_tensor = F.softmax(assign_tensor, dim=1)
        assign_tensor = torch.split(assign_tensor, g.batch_num_nodes().tolist())
        assign_tensor = torch.block_diag(*assign_tensor)  # size = (sum_N, batch_size * N_a)

        h = torch.matmul(torch.t(assign_tensor), feat)
        adj = g.adjacency_matrix(transpose=True, ctx=device)
        adj_new = torch.sparse.mm(adj, assign_tensor)
        adj_new = torch.mm(torch.t(assign_tensor), adj_new)

        if self.link_pred:
            current_lp_loss = torch.norm(adj.to_dense() -
                                         torch.mm(assign_tensor, torch.t(assign_tensor))) / np.power(g.number_of_nodes(), 2)
            self.loss_log['LinkPredLoss'] = current_lp_loss

        for loss_layer in self.reg_loss:
            loss_name = str(type(loss_layer).__name__)
            self.loss_log[loss_name] = loss_layer(adj, adj_new, assign_tensor)

        return adj_new, h