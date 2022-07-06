import logging
import plotter
import torch.nn.functional as F
import argparse
from model import *
from data import *
import pickle
import random as random
from GlobalProperties import *
from stat_rnn import mmd_eval
import time
import timeit
import dgl

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

subgraphSize = None
keepThebest = False

parser = argparse.ArgumentParser(description='Kernel VGAE')

parser.add_argument('-e', dest="epoch_number", default=20000, help="Number of Epochs to train the model", type=int)
parser.add_argument('-v', dest="Vis_step", default=1000, help="at every Vis_step 'minibatch' the plots will be updated")
parser.add_argument('-redraw', dest="redraw", default=False, help="either update the log plot each step")
parser.add_argument('-lr', dest="lr", default=0.0003, help="model learning rate")
parser.add_argument('-dataset', dest="dataset", default="PTC",
                    help="possible choices are:   wheel_graph,PTC, FIRSTMM_DB, star, triangular_grid, multi_community, NCI1, ogbg-molbbbp, IMDbMulti, grid, community, citeseer, lobster, DD")  # citeceer: ego; DD:protein
parser.add_argument('-graphEmDim', dest="graphEmDim", default=1024, help="the dimention of graph Embeding LAyer; z")
parser.add_argument('-graph_save_path', dest="graph_save_path", default=None,
                    help="the direc to save generated synthatic graphs")
parser.add_argument('-f', dest="use_feature", default=True, help="either use features or identity matrix")
parser.add_argument('-PATH', dest="PATH", default="model",
                    help="a string which determine the path in wich model will be saved")
parser.add_argument('-decoder', dest="decoder", default="FC", help="the decoder type, FC is only option in this rep")
parser.add_argument('-encoder', dest="encoder_type", default="AvePool",
                    help="the encoder: only option in this rep is 'AvePool'")  # only option in this rep is "AvePool"
parser.add_argument('-batchSize', dest="batchSize", default=200,
                    help="the size of each batch; the number of graphs is the mini batch")
parser.add_argument('-UseGPU', dest="UseGPU", default=True, help="either use GPU or not if availabel")
parser.add_argument('-model', dest="model", default="KernelAugmentedWithTotalNumberOfTriangles",
                    help="KernelAugmentedWithTotalNumberOfTriangles is the only option in this rep")
parser.add_argument('-device', dest="device", default="cuda:0", help="Which device should be used")
parser.add_argument('-task', dest="task", default="graphGeneration", help="only option in this rep is graphGeneration")
parser.add_argument('-BFS', dest="bfsOrdering", default=True, help="use bfs for graph permutations", type=bool)
parser.add_argument('-directed', dest="directed", default=True, help="is the dataset directed?!", type=bool)
parser.add_argument('-beta', dest="beta", default=None, help="beta coefiicieny", type=float)
parser.add_argument('-plot_testGraphs', dest="plot_testGraphs", default=True, help="shall the test set be printed",
                    type=float)

args = parser.parse_args()

encoder_type = args.encoder_type
graphEmDim = args.graphEmDim
visulizer_step = args.Vis_step
redraw = args.redraw
device = args.device
task = args.task
plot_testGraphs = args.plot_testGraphs
directed = args.directed
epoch_number = args.epoch_number
lr = args.lr
decoder_type = args.decoder
dataset = args.dataset  # possible choices are: cora, citeseer, karate, pubmed, DBIS
mini_batch_size = args.batchSize
use_gpu = args.UseGPU
use_feature = args.use_feature

graph_save_path = args.graph_save_path
graph_save_path = args.graph_save_path

if graph_save_path == None:
    graph_save_path = "MMD_" + encoder_type + "_" + decoder_type + "_" + dataset + "_" + task + "_" + args.model + "BFS" + str(
        args.bfsOrdering) + str(args.epoch_number) + str(time.time()) + "/"
from pathlib import Path

Path(graph_save_path).mkdir(parents=True, exist_ok=True)

# maybe to the beest way
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=graph_save_path + 'log.log', filemode='w', level=logging.INFO)

# **********************************************************************
# setting
print("KernelVGAE SETING: " + str(args))
logging.info("KernelVGAE SETING: " + str(args))
PATH = args.PATH  # the dir to save the with the best performance on validation data

kernl_type = []

if args.model == "KernelAugmentedWithTotalNumberOfTriangles":
    kernl_type = ["trans_matrix", "in_degree_dist", "out_degree_dist", "TotalNumberOfTriangles"]

    step_num = 5
    if dataset == "large_grid":
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 20, 100]
    elif dataset == "ogbg-molbbbp":
        # leision study
        alpha = [1, 1, 1, 1, 1, 0, 0, 0, 40, 1500]
        alpha = [0, 0, 0, 0, 0, 1, 1, 0, 40, 1500]
        alpha = [0, 0, 0, 0, 0, 0, 0, 1, 40, 1500]
        # -----------------------------------------
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 40, 1500]
    elif dataset == "PTC":

        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 40, 100]

    elif dataset == "FIRSTMM_DB":
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 50, 100]
    elif dataset == "DD":
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 50, 1000]
    elif dataset == "grid":
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 50, 2000]
    elif dataset == "lobster":
        # leision study
        alpha = [1, 1, 1, 1, 1, 0, 0, 0, 40, 2000]  # degree
        alpha = [0, 0, 0, 0, 0, 1, 1, 0, 40, 2000]  # degree
        alpha = [1, 1, 1, 1, 1, 0, 0, 0, 40, 2000]
        # -------------------------------------------------
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 40, 2000]
    elif dataset == "wheel_graph":
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 3000000, 20000 * 50000]
    elif dataset == "triangular_grid":
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 50, 2000]
    elif dataset == "tree":
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 50, 2000]
elif args.model == "kipf":
    alpha = [1, 1]
    step_num = 0

AutoEncoder = False

if AutoEncoder == True:
    alpha[-1] = 0

if args.beta != None:
    alpha[-1] = args.beta

print("kernl_type:" + str(kernl_type))
print("alpha: " + str(alpha) + " num_step:" + str(step_num))

logging.info("kernl_type:" + str(kernl_type))
logging.info("alpha: " + str(alpha) + " num_step:" + str(step_num))

bin_center = torch.tensor([[x / 10000] for x in range(0, 1000, 1)])
bin_width = torch.tensor([[9000] for x in range(0, 1000, 1)])  # with is propertion to revese of this value;

device = torch.device(device if torch.cuda.is_available() and use_gpu else "cpu")
print("the selected device is :", device)
logging.info("the selected device is :" + str(device))

# setting the plots legend
functions = ["Accuracy", "loss"]
if args.model == "kernel" or args.model == "KernelAugmentedWithTotalNumberOfTriangles":
    functions.extend(["Kernel" + str(i) for i in range(step_num)])
    functions.extend(kernl_type[1:])

if args.model == "TrianglesOfEachNode":
    functions.extend(kernl_type)

if args.model == "ThreeStepPath":
    functions.extend(kernl_type)

if args.model == "TotalNumberOfTriangles":
    functions.extend(kernl_type)

functions.append("Binary_Cross_Entropy")
functions.append("KL-D")

# ========================================================================


pltr = plotter.Plotter(save_to_filepath="kernelVGAE_Log", functions=functions)

synthesis_graphs = {"wheel_graph", "star", "triangular_grid", "DD", "ogbg-molbbbp", "grid", "small_lobster",
                    "small_grid", "community", "lobster", "ego", "one_grid"}


class NodeUpsampling(torch.nn.Module):
    def __init__(self, InNode_num, outNode_num, InLatent_dim=None):
        super(NodeUpsampling, self).__init__()
        self.Node_mlp = torch.nn.Linear(InNode_num * InLatent_dim, InLatent_dim * outNode_num)

    def forward(self, inTensor, activation=torch.nn.LeakyReLU(0.001)):
        Z = self.Node_mlp(inTensor.reshpe(inTensor.shape[0], -1))
        Z = torch.matmul(Z.reshpe(inTensor.shape[0], -1).permute(0, 2, 1), inTensor)

        return activation(Z)


class LatentMtrixTransformer(torch.nn.Module):
    def __init__(self, InNode_num, InLatent_dim=None, OutLatentDim=None):
        super(NodeUpsampling, self).__init__()
        self.Node_mlp = torch.nn.Linear(InNode_num * InLatent_dim, InNode_num * OutLatentDim)

    def forward(self, inTensor, activation=torch.nn.LeakyReLU(0.001)):
        Z = self.Node_mlp(inTensor.reshpe(inTensor.shape[0], -1))
        Z = torch.matmul(inTensor, Z.reshpe(inTensor.shape[-1], -1))

        return activation(Z)


# ============================================================================

def test_(number_of_samples, model, graph_size, path_to_save_g, remove_self=True, save_graphs=True):
    import os
    if not os.path.exists(path_to_save_g):
        os.makedirs(path_to_save_g)
    # model.eval()
    generated_graph_list = []
    if not os.path.isdir(path_to_save_g):
        os.makedirs(path_to_save_g)
    k = 0
    for g_size in graph_size:
        for j in range(number_of_samples):
            z = torch.tensor(numpy.random.normal(size=[1, model.embeding_dim]))
            z = torch.randn_like(z)
            start_time = time.time()

            adj_logit = model.decode(z.to(device).float())
            print("--- %s seconds ---" % (time.time() - start_time))
            logging.info("--- %s seconds ---" % (time.time() - start_time))
            reconstructed_adj = torch.sigmoid(adj_logit)
            sample_graph = reconstructed_adj[0].cpu().detach().numpy()
            # sample_graph = sample_graph[:g_size,:g_size]
            sample_graph[sample_graph >= 0.5] = 1
            sample_graph[sample_graph < 0.5] = 0
            G = nx.from_numpy_matrix(sample_graph)
            # generated_graph_list.append(G)
            f_name = path_to_save_g + str(k) + str(g_size) + str(j) + dataset
            k += 1
            # plot and save the generated graph
            # plotter.plotG(G, "generated" + dataset, file_name=f_name)
            if remove_self:
                G.remove_edges_from(nx.selfloop_edges(G))

            G.remove_nodes_from(list(nx.isolates(G)))
            generated_graph_list.append(G)
            if save_graphs:
                plotter.plotG(G, "generated" + dataset, file_name=f_name + "_ConnectedComponnents")
    # ======================================================
    # save nx files
    if save_graphs:
        nx_f_name = path_to_save_g + "_" + dataset + "_" + decoder_type + "_" + args.model + "_" + task
        with open(nx_f_name, 'wb') as f:
            pickle.dump(generated_graph_list, f)
    # # ======================================================
    return generated_graph_list


def EvalTwoSet(model, test_list_adj, graph_save_path, Save_generated=True, _f_name=None):
    generated_graphs = test_(1, model, [x.shape[0] for x in test_list_adj], graph_save_path, save_graphs=Save_generated)
    graphs_to_writeOnDisk = [nx.to_numpy_array(G) for G in generated_graphs]
    if Save_generated:
        np.save(graph_save_path + 'generatedGraphs_adj_' + str(_f_name) + '.npy', graphs_to_writeOnDisk,
                allow_pickle=True)


    logging.info(mmd_eval(generated_graphs, [nx.from_numpy_matrix(graph.toarray()) for graph in test_list_adj]))
    print("====================================================")
    logging.info("====================================================")

    print("result for subgraph with maximum connected componnent")
    logging.info("result for subgraph with maximum connected componnent")
    generated_graphs = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in generated_graphs if
                        not nx.is_empty(G)]
    logging.info(
        mmd_eval(generated_graphs, [nx.from_numpy_matrix(graph.toarray()) for graph in test_list_adj], diam=True))

    if Save_generated:
        graphs_to_writeOnDisk = [nx.to_numpy_array(G) for G in generated_graphs]
        np.save(graph_save_path + 'Single_comp_generatedGraphs_adj_' + str(_f_name) + '.npy', graphs_to_writeOnDisk,
                allow_pickle=True)

        graphs_to_writeOnDisk = [G.toarray() for G in test_list_adj]
        np.save(graph_save_path + 'testGraphs_adj_.npy', graphs_to_writeOnDisk, allow_pickle=True)


def get_subGraph_features(org_adj, subgraphs_indexes, kernel_model):
    subgraphs = []
    target_kelrnel_val = None

    for i in range(len(org_adj)):
        subGraph = org_adj[i]
        if subgraphs_indexes != None:
            subGraph = subGraph[:, subgraphs_indexes[i]]
            subGraph = subGraph[subgraphs_indexes[i], :]
        # Converting sparse matrix to sparse tensor
        subGraph = torch.tensor(subGraph.todense())
        subgraphs.append(subGraph)
    subgraphs = torch.stack(subgraphs).to(device)

    if kernel_model != None:
        target_kelrnel_val = kernel_model(subgraphs)
        target_kelrnel_val = [val.to("cpu") for val in target_kelrnel_val]
    subgraphs = subgraphs.to("cpu")
    torch.cuda.empty_cache()
    return target_kelrnel_val, subgraphs


# the code is a hard copy of https://github.com/orybkin/sigma-vae-pytorch
def log_guss(mean, log_std, samples):
    return 0.5 * torch.pow((samples - mean) / log_std.exp(), 2) + log_std + 0.5 * np.log(2 * np.pi)


def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor


def OptimizerVAE(reconstructed_adj, reconstructed_kernel_val, targert_adj, target_kernel_val, log_std, mean, alpha,
                 reconstructed_adj_logit, pos_wight, norm):
    loss = norm * torch.nn.functional.binary_cross_entropy_with_logits(reconstructed_adj_logit.float(),
                                                                       targert_adj.float(), pos_weight=pos_wight)

    norm = mean.shape[0] * mean.shape[1]
    kl = (1 / norm) * -0.5 * torch.sum(1 + 2 * log_std - mean.pow(2) - torch.exp(log_std).pow(2))

    acc = (reconstructed_adj.round() == targert_adj).sum() / float(
        reconstructed_adj.shape[0] * reconstructed_adj.shape[1] * reconstructed_adj.shape[2])
    kernel_diff = 0
    each_kernel_loss = []

    for i in range(len(target_kernel_val)):
        log_sigma = ((reconstructed_kernel_val[i] - target_kernel_val[i]) ** 2).mean().sqrt().log()
        log_sigma = softclip(log_sigma, -6)
        step_loss = log_guss(target_kernel_val[i], log_sigma, reconstructed_kernel_val[i]).mean()
        each_kernel_loss.append(step_loss.cpu().detach().numpy() * alpha[i])
        kernel_diff += step_loss * alpha[i]

    kernel_diff += loss * alpha[-2]
    kernel_diff += kl * alpha[-1]
    each_kernel_loss.append((loss * alpha[-2]).item())
    each_kernel_loss.append((kl * alpha[-1]).item())
    return kl, loss, acc, kernel_diff, each_kernel_loss


def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])


# test_(5, "results/multiple graph/cora/model" , [x**2 for x in range(5,10)])


# load the data

list_adj, list_x, list_label = list_graph_loader(dataset, return_labels=True)  # , _max_list_size=80)
# list_adj = list_adj[:400]
# list_x = list_x[:400]
# list_label = list_label[:400]

if args.bfsOrdering == True:
    list_adj = BFS(list_adj)

# list_adj, list_x, list_label = list_graph_loader(dataset, return_labels=True, _max_list_size=80)

# list_adj, _ = permute(list_adj, None)
self_for_none = True
if (decoder_type) in ("FCdecoder"):  # ,"FC_InnerDOTdecoder"
    self_for_none = True

if len(list_adj) == 1:
    test_list_adj = list_adj.copy()
    list_graphs = Datasets(list_adj, self_for_none, list_x, None)
else:
    max_size = None
    list_label = None
    list_adj, test_list_adj, list_x_train, _ = data_split(list_adj, list_x)
    val_adj = list_adj[:int(len(test_list_adj))]
    list_graphs = Datasets(list_adj, self_for_none, list_x_train, list_label, Max_num=max_size,
                           set_diag_of_isol_Zer=False)

    if plot_testGraphs:
        print("printing the test set...")
        for i, G in enumerate(test_list_adj):
            G = nx.from_numpy_matrix(G.toarray())
            plotter.plotG(G, graph_save_path+"_test_graph" + str(i))

print("#------------------------------------------------------")
fifty_fifty_dataset = list_adj + test_list_adj

fifty_fifty_dataset = [nx.from_numpy_matrix(graph.toarray()) for graph in fifty_fifty_dataset]
random.shuffle(fifty_fifty_dataset)
print("50%50 Evalaution of dataset")
logging.info(mmd_eval(fifty_fifty_dataset[:int(len(fifty_fifty_dataset)/2)],fifty_fifty_dataset[int(len(fifty_fifty_dataset)/2):],diam=True))

graphs_to_writeOnDisk = [nx.to_numpy_array(G) for  G in fifty_fifty_dataset]
np.save(graph_save_path+dataset+'_dataset.npy', graphs_to_writeOnDisk, allow_pickle=True)
print("#------------------------------------------------------")

SubGraphNodeNum = subgraphSize if subgraphSize != None else list_graphs.max_num_nodes
in_feature_dim = list_graphs.feature_size  # ToDo: consider none Synthasis data
nodeNum = list_graphs.max_num_nodes

degree_center = torch.tensor([[x] for x in range(0, SubGraphNodeNum, 1)])
degree_width = torch.tensor([[.1] for x in range(0, SubGraphNodeNum,
                                                 1)])  # ToDo: both bin's center and widtg also maximum value of it should be determinde auomaticly
# ToDo: both bin's center and widtg also maximum value of it should be determinde auomaticly

kernel_model = kernel(device=device, kernel_type=kernl_type, step_num=step_num,
                      bin_width=bin_width, bin_center=bin_center, degree_bin_center=degree_center,
                      degree_bin_width=degree_width)

if encoder_type == "AvePool":
    encoder = AveEncoder(in_feature_dim, [256], graphEmDim)
else:
    print("requested encoder is not implemented")
    exit(1)

if decoder_type == "FC":
    decoder = GraphTransformerDecoder_FC(graphEmDim, 256, nodeNum, directed)
else:
    print("requested decoder is not implemented")
    exit(1)

model = kernelGVAE(kernel_model, encoder, decoder, AutoEncoder,
                   graphEmDim=graphEmDim)  # parameter namimng, it should be dimentionality of distriburion
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr)

# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5000,6000,7000,8000,9000], gamma=0.5)

# pos_wight = torch.true_divide((list_graphs.max_num_nodes**2*len(list_graphs.processed_adjs)-list_graphs.toatl_num_of_edges),
#                               list_graphs.toatl_num_of_edges) # addrressing imbalance data problem: ratio between positve to negative instance
# pos_wight = torch.tensor(40.0)
# pos_wight/=10
num_nodes = list_graphs.max_num_nodes
# ToDo Check the effect of norm and pos weight

# target_kelrnel_val = kernel_model(target_adj)

list_graphs.shuffle()
start = timeit.default_timer()
# Parameters
step = 0
swith = False
print(model)
logging.info(model.__str__())
min_loss = float('inf')

if (subgraphSize == None):
    list_graphs.processALL(self_for_none=self_for_none)
    adj_list = list_graphs.get_adj_list()
    graphFeatures, _ = get_subGraph_features(adj_list, None, kernel_model)
    list_graphs.set_features(graphFeatures)

# 50%50 Evaluation

load_model = False
if load_model == True:  # I used this in line code to load a model #TODO: fix it
    # ========================================
    model_dir = "/local-scratch/kiarash/AAAI/Graph-Generative-Models/MMD_AvePool_FC_DD_graphGeneration_KernelAugmentedWithTotalNumberOfTrianglesBFSTrue100001651364417.4785793/"
    model.load_state_dict(torch.load(model_dir + "model_9999_3"))
    # EvalTwoSet(model, test_list_adj, model_dir+"/", Save_generated= False, )

# model_dir1 = "/local-scratch/kiarash/AAAI/Graph-Generative-Models/FinalResultHopefully/"
# model.load_state_dict(torch.load(model_dir1+"model_9999_3"))
# EvalTwoSet(model, test_list_adj, model_dir+"/", Save_generated= False, )


for epoch in range(epoch_number):

    list_graphs.shuffle()
    batch = 0
    for iter in range(0, max(int(len(list_graphs.list_adjs) / mini_batch_size), 1) * mini_batch_size, mini_batch_size):
        from_ = iter
        to_ = mini_batch_size * (batch + 1)
        # for iter in range(0, len(list_graphs.list_adjs), mini_batch_size):
        #     from_ = iter
        #     to_= mini_batch_size*(batch+1) if mini_batch_size*(batch+2)<len(list_graphs.list_adjs) else len(list_graphs.list_adjs)

        if subgraphSize == None:
            org_adj, x_s, node_num, subgraphs_indexes, target_kelrnel_val = list_graphs.get__(from_, to_, self_for_none,
                                                                                              bfs=subgraphSize)
        else:
            org_adj, x_s, node_num, subgraphs_indexes = list_graphs.get__(from_, to_, self_for_none, bfs=subgraphSize)

        if (type(decoder)) in [GraphTransformerDecoder_FC]:  #
            node_num = len(node_num) * [list_graphs.max_num_nodes]

        x_s = torch.cat(x_s)
        x_s = x_s.reshape(-1, x_s.shape[-1])

        model.train()
        if subgraphSize == None:
            _, subgraphs = get_subGraph_features(org_adj, None, None)
        else:
            target_kelrnel_val, subgraphs = get_subGraph_features(org_adj, subgraphs_indexes, kernel_model)

        # target_kelrnel_val = kernel_model(org_adj, node_num)

        # batchSize = [org_adj.shape[0], org_adj.shape[1]]

        batchSize = [len(org_adj), org_adj[0].shape[0]]

        # org_adj_dgl = [dgl.from_scipy(sp.csr_matrix(graph.cpu().detach().numpy())) for graph in org_adj]
        [graph.setdiag(1) for graph in org_adj]
        org_adj_dgl = [dgl.from_scipy(graph) for graph in org_adj]
        org_adj_dgl = dgl.batch(org_adj_dgl).to(device)
        pos_wight = torch.true_divide(sum([x.shape[-1] ** 2 for x in subgraphs]) - subgraphs.sum(), subgraphs.sum())

        reconstructed_adj, prior_samples, post_mean, post_log_std, generated_kernel_val, reconstructed_adj_logit = model(
            org_adj_dgl.to(device), x_s.to(device), batchSize, subgraphs_indexes)
        kl_loss, reconstruction_loss, acc, kernel_cost, each_kernel_loss = OptimizerVAE(reconstructed_adj,
                                                                                        generated_kernel_val,
                                                                                        subgraphs.to(device),
                                                                                        [val.to(device) for val in
                                                                                         target_kelrnel_val],
                                                                                        post_log_std, post_mean, alpha,
                                                                                        reconstructed_adj_logit,
                                                                                        pos_wight, 2)

        loss = kernel_cost

        tmp = [None for x in range(len(functions))]
        pltr.add_values(step, [acc.cpu().item(), loss.cpu().item(), *each_kernel_loss], tmp,
                        redraw=redraw)  # ["Accuracy", "loss", "AUC"])

        step += 1
        optimizer.zero_grad()
        loss.backward()

        if keepThebest and min_loss > loss:
            min_loss = loss.item()
            torch.save(model.state_dict(), "model")
        # torch.nn.utils.clip_grad_norm(model.parameters(),  1.0044e-05)
        optimizer.step()

        if (step + 1) % visulizer_step == 0:
            model.eval()
            pltr.redraw()
            if dataset in synthesis_graphs:
                dir_generated_in_train = "generated_graph_train/"
                if not os.path.isdir(dir_generated_in_train):
                    os.makedirs(dir_generated_in_train)
                rnd_indx = random.randint(0, len(node_num) - 1)
                sample_graph = reconstructed_adj[rnd_indx].cpu().detach().numpy()
                sample_graph = sample_graph[:node_num[rnd_indx], :node_num[rnd_indx]]
                sample_graph[sample_graph >= 0.5] = 1
                sample_graph[sample_graph < 0.5] = 0
                G = nx.from_numpy_matrix(sample_graph)
                plotter.plotG(G, "generated" + dataset,
                              file_name=graph_save_path + "generatedSample_At_epoch" + str(epoch))
            model.eval()
            if task == "graphGeneration":
                EvalTwoSet(model, val_adj, graph_save_path, Save_generated=True, _f_name=epoch)

                if ((step + 1) % visulizer_step * 2):
                    torch.save(model.state_dict(), graph_save_path + "model_" + str(epoch) + "_" + str(batch))
            model.train()
            # if reconstruction_loss.item()<0.051276 and not swith:
            #     alpha[-1] *=2
            #     swith = True
        k_loss_str = ""
        for indx, l in enumerate(each_kernel_loss):
            k_loss_str += functions[indx + 2] + ":"
            k_loss_str += str(l) + ".   "

        print(
            "Epoch: {:03d} |Batch: {:03d} | loss: {:05f} | reconstruction_loss: {:05f} | z_kl_loss: {:05f} | accu: {:03f}".format(
                epoch + 1, batch, loss.item(), reconstruction_loss.item(), kl_loss.item(), acc), k_loss_str)
        logging.info(
            "Epoch: {:03d} |Batch: {:03d} | loss: {:05f} | reconstruction_loss: {:05f} | z_kl_loss: {:05f} | accu: {:03f}".format(
                epoch + 1, batch, loss.item(), reconstruction_loss.item(), kl_loss.item(), acc) + " " + str(k_loss_str))
        batch += 1
        # scheduler.step()
model.eval()
torch.save(model.state_dict(), graph_save_path + "model_" + str(epoch) + "_" + str(batch))

stop = timeit.default_timer()
print("trainning time:", str(stop - start))
logging.info("trainning time: " + str(stop - start))
# save the train loss for comparing the convergence
import json

file_name = graph_save_path + "_" + encoder_type + "_" + decoder_type + "_" + dataset + "_" + task + "_" + args.model + "_elbo_loss.txt"

with open(file_name, "w") as fp:
    json.dump(list(np.array(pltr.values_train[-2]) + np.array(pltr.values_train[-1])), fp)

with open(file_name + "_CrossEntropyLoss.txt", "w") as fp:
    json.dump(list(np.array(pltr.values_train[-2])), fp)

with open(file_name + "_train_loss.txt", "w") as fp:
    json.dump(pltr.values_train[1], fp)

# save the log plot on the current directory
pltr.save_plot(graph_save_path + "KernelVGAE_log_plot")

if task == "graphGeneration":
    EvalTwoSet(model, test_list_adj, graph_save_path, Save_generated=True, _f_name="final_eval")
