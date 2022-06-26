"""This file include a set of functions which calculate a graph statistics such as the number of
triangle in the given graph"""

# def triangles_count(adj):
#     '''
#     :param adj: the sparse adjacency matrix
#     :return: the number of triangle in the graph adj
#     '''
#     tri = adj.multiply(adj).adj(adj)
#     return sum(tri.diagonal())/6
#
#
# def charechteristic_path_lenght(adj):
#     '''
#     For any connected graph G,  this function return average distance between pairs of all vertices
#     which is referred to as the graph's "characteristic path length."
#     :param adj: matrix adjacency matrix; a numpy matrix.
#     :return: characteristic path length
#     '''
#     g = nx.from_numpy_matrix(adj)
#     return g.average_shortest_path_length
#
#
# def graph_components(adj):
#     """
#     this function returns a sorted list which contain the size of each coponent in the graph.
#     :param adj: matrix adjacency matrix; a numpy matrix.
#     :return:  a sorted list which contain the size of each coponent in the graph
#     """
#     G =nx.from_numpy_matrix(adj)
#     comp = sorted(nx.connected_components(G), key=len, reverse=True)
#     return [len(x) for x in comp]

import igraph
import networkx as nx
import numpy as np
#asakhuja Changing import statement - Starts
import powerlaw
# from scipy.stats import powerlaw
#asakhuja Changing import statement - Ends
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
# import matplotlib.pyplot as plt
# plt.switch_backend('agg')




def statistics_degrees(A_in):
    """
    Compute min, max, mean degree
    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    d_max. d_min, d_mean
    """

    degrees = A_in.sum(axis=0)
    return np.max(degrees), np.min(degrees), np.mean(degrees)


def statistics_LCC(A_in):
    """
    Compute the size of the size of all connected component (LCC)
    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    a list of intergers in which each elemnt is the size of one of the connected components
    """

    unique, counts = np.unique(connected_components(A_in)[1], return_counts=True)
    # LCC = np.where(connected_components(A_in)[1] == np.argmax(counts))[0]
    return counts


def statistics_wedge_count(A_in):
    """
    Compute the wedge count of the input graph
    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    The wedge count.
    """
    degrees = A_in.sum(axis=0).flatten()
    return float(np.sum(np.array([0.5 * x * (x - 1) for x in degrees])))


def statistics_claw_count(A_in):
    """
    Compute the claw count of the input graph; A star with 3 edges is called a claw
    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Claw count
    """

    degrees = A_in.sum(axis=0).flatten()
    return float(np.sum(np.array([1 / 6. * x * (x - 1) * (x - 2) for x in degrees])))


def statistics_triangle_count(A_in):
    """
    Compute the triangle count of the input graph
    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Triangle count
    """

    A_graph = nx.from_numpy_matrix(A_in)
    triangles = nx.triangles(A_graph)
    t = np.sum(list(triangles.values())) / 3
    return int(t)


def squares(g):
    """
    Count the number of squares for each node
    Parameters
    ----------
    g: igraph Graph object
       The input graph.
    Returns
    -------
    List with N entries (N is number of nodes) that give the number of squares a node is part of.
    """

    cliques = g.cliques(min=4, max=4)
    result = [0] * g.vcount()
    for i, j, k, l in cliques:
        result[i] += 1
        result[j] += 1
        result[k] += 1
        result[l] += 1
    return result


def statistics_square_count(A_in):
    """
    Compute the square count of the input graph
    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Square count
    """

    A_igraph = igraph.Graph.Adjacency((A_in > 0).tolist()).as_undirected()
    return int(np.sum(squares(A_igraph)) / 4)


def statistics_power_law_alpha(A_in):
    """
    Compute the power law coefficient of the degree distribution of the input graph
    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Power law coefficient
    """

    degrees = A_in.sum(axis=0).flatten()
    return powerlaw.Fit(degrees, xmin=max(np.min(degrees),1)).power_law.alpha


def statistics_gini(A_in):
    """
    Compute the Gini coefficient of the degree distribution of the input graph
    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Gini coefficient
    """

    n = A_in.shape[0]
    degrees = A_in.sum(axis=0).flatten()
    degrees_sorted = np.sort(degrees)
    G = (2 * np.sum(np.array([i * degrees_sorted[i] for i in range(len(degrees))]))) / (n * np.sum(degrees)) - (
                                                                                                               n + 1) / n
    return float(G)


def statistics_edge_distribution_entropy(A_in):
    """
    Compute the relative edge distribution entropy of the input graph.
    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Rel. edge distribution entropy
    """

    degrees = A_in.sum(axis=0).flatten()
    m = 0.5 * np.sum(np.square(A_in))
    n = A_in.shape[0]

    H_er = 1 / np.log(n) * np.sum(-degrees / (2 * float(m)) * np.log((degrees+.0001) / (2 * float(m))))
    return H_er


def statistics_compute_cpl(A):
    """Compute characteristic path length."""
    P = sp.csgraph.shortest_path(sp.csr_matrix(A))
    return P[((1 - np.isinf(P)) * (1 - np.eye(P.shape[0]))).astype(np.bool)].mean()


def compute_graph_statistics(A_in):
    """
    Parameters
    ----------
    A_in: numpy matrix
          The input adjacency matrix.

    Returns
    -------
    Dictionary containing the following statistics:
             * Maximum, minimum, mean degree of nodes
             * Size of the largest connected component (LCC)
             * Wedge count; Let a wedge be a two-hop path in an undirected graph.
             * Claw count; A star with 3 edges is called a claw.
             * Triangle count
             * Square count
             * Power law exponent
             * Gini coefficient
             * Relative edge distribution entropy
             * Assortativity; Assortativity quantifies the tendency of nodes being connected to similar nodes in a complex network. often Degree
             * Clustering coefficient; The local clustering coefficient of a vertex (node) in a graph quantifies how close its neighbours are to being a clique
             * Number of connected components
             * Intra- and inter-community density (if Z_obs is passed)
             * Characteristic path length
    """
    A = A_in.copy()

    if (A == A.T).all():
        print("Graph is symetric; Undirected")
    else:
        print("Graph is Asymetric; Directed")

    A_graph = nx.from_numpy_matrix(A)


    statistics = {}

    d_max, d_min, d_mean = statistics_degrees(A)

    # Degree statistics
    statistics['d_max'] = d_max
    statistics['d_min'] = d_min
    statistics['d'] = d_mean
    # print(statistics)

    # node number & edger number
    statistics['node_num'] = A_graph.number_of_nodes()
    statistics['edge_num'] = A_graph.number_of_edges()
    # print(statistics)

    # size of connected component
    CC = statistics_LCC(A)
    statistics['CC'] = CC
    # print(statistics)

    # wedge count
    statistics['wedge_count'] = statistics_wedge_count(A)
    # print(statistics)

    # claw count
    statistics['claw_count'] = statistics_claw_count(A)
    # print(statistics)

    # triangle count
    statistics['triangle_count'] = statistics_triangle_count(A)
    # print(statistics)

    # Square count # too expensive; commentd
    # statistics['square_count'] = statistics_square_count(A)
    # print(statistics)

    # power law exponent
    statistics['power_law_exp'] = statistics_power_law_alpha(A) #https://en.wikipedia.org/wiki/Scale-free_network
    # print(statistics)

    # gini coefficient
    statistics['gini'] = statistics_gini(A)
    # print(statistics)

    # Relative edge distribution entropy
    statistics['rel_edge_distr_entropy'] = statistics_edge_distribution_entropy(A)
    # print(statistics)

    # Assortativity
    statistics['assortativity'] = nx.degree_assortativity_coefficient(A_graph)
    print(statistics)

    # Clustering coefficient
    statistics['clustering_coefficient'] = 3 * statistics['triangle_count'] / (statistics['claw_count']+1)
    # print(statistics)

    # Number of connected components
    statistics['n_components'] = connected_components(A)[0]
    # print(statistics)

    # if Z_obs is not None:
    #     # inter- and intra-community density
    #     intra, inter = statistics_cluster_props(A, Z_obs)
    #     statistics['intra_community_density'] = intra
    #     statistics['inter_community_density'] = inter

    statistics['cpl'] = statistics_compute_cpl(A)
    # print(statistics)

    return statistics


