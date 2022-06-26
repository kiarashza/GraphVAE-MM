#!/usr/bin/env python
# coding: utf-8

# In[26]:


#Visualization for Facebook Egos

## importing the required packages
import random
import time
import matplotlib.pyplot as plt


from sklearn.manifold import TSNE


def embeding_vis(features, labels):
    # Apply t-SNE transformation on node embeddings
    # tsne = TSNE(learning_rate=10, metric="cosine",perplexity = 100, n_components = 2, verbose = 0, n_iter = 5000, init ="pca")
    # node_embeddings_2d = tsne.fit_transform(features)

    f = plt.figure(figsize=(10, 8))
    plt.scatter(
        features[:, 0],
        features[:, 1],
        c=labels,
        cmap="jet",
    alpha = 0.7,
    )
    # plt.show()

    f.savefig("tsne_graph.png" )

import matplotlib.cm as cm

    


def make_emb_dict(embed_tensor):
    """Loads the embed file and creates a dictionary which key is the label and value is the embedding"""
    '''
    with open(filename) as f:
        content = f.readlines()
    content = [x.split() for x in content]
    '''

    emb_dicti = {}
    for i in range(1, len(embed_tensor)):
        label = i
        emb_dicti[label] =   [float(i) for i in embed_tensor[i][:]]
    return emb_dicti

#Creating dictionary for embeddings


def visualize_embed(emb_dict,circle_dict,ego):
    labels = []
    tokens=[]
    for word in list(circle_dict.keys())[:]:
            tokens.append(emb_dict[word])
            labels.append(circle_dict[word])
 
    time_start = time.time()
    X_tsne = TSNE(learning_rate=10,perplexity=100, n_components=2, verbose=0,n_iter=5000,init="pca").fit_transform(tokens)
    
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    plt.figure(figsize=(10, 10))
    #plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c= X_tsne[:, 0])
    
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c= labels, cmap=cm.tab20)
    i=0
    for x,y in zip(X_tsne[:, 0],X_tsne[:, 1]):
        label = labels[i]
        i += 1
        label=''
        plt.annotate(label, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(5,2), # distance from text to points (x,y)
                     ha='left') # horizontal alignment can be left, right or center
    #plt.legend(loc="upper left")
    plt.title('Ego : {}, Circles: {}' .format(ego, max(labels)))
    plt.show(block=False)
    plt.savefig("TSNE_Ego_{}".format(ego))
    

#For k-means
def visualize_embed_kmeans(emb_dict):
    labels = []
    tokens=[]
    for word in list(emb_dict.keys())[:]:
            labels.append(emb_dict[word])
 
    time_start = time.time()
    X_tsne = TSNE(learning_rate=10,perplexity=100, n_components=2, verbose=0,n_iter=5000,init="pca").fit_transform(labels)
    
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    #plt.figure(figsize=(10, 10))
    #plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c= X_tsne[:, 0])
    
    #plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    
    #plt.show(block=False)
    return X_tsne
   
    
def visualize_baseline(emb_dict,edge_label,dataset,decoder):
    labels = []
    tokens=[]
    '''
    for word in list(label_dict.keys())[:]:
            tokens.append(emb_dict[word])
            labels.append(label_dict[word])
    '''
 
    time_start = time.time()
    #X_tsne = TSNE(learning_rate=10,perplexity=100, n_components=2, verbose=0,n_iter=5000,init="pca").fit_transform(emb_dict)
    
    #print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    '''
    import numpy as np
    def unique(list1):
        x = np.array(list1)
        print(np.unique(x))
    '''
    
    
    plt.figure(figsize=(5, 5))
    #plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c= X_tsne[:, 0])

# Calling map() function on a dictionary
   
    l_label=len(edge_label)
    #l_emb=len(X_tsne)
    l_emb=len(emb_dict)
    
    import matplotlib.colors as mcol
    cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",["#FFFFFF","#0B0080"])


    
    if l_label>l_emb:
        #plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c= edge_label[:l_emb],s=2, cmap=cm.seismic)
        plt.scatter(emb_dict[:,0],emb_dict[:,1], c= edge_label[:l_emb],s=2, cmap=cm.seismic)
    elif l_label<l_emb:
        plt.scatter(emb_dict[:, 0], emb_dict[:, 1], c= edge_label[:l_label],s=2, cmap=cm.seismic)
    else:
        plt.scatter(emb_dict[:, 0], emb_dict[:, 1], c= edge_label,s=2,cmap=cm.seismic)
    
    i=0
    plt.show(block=False)
    plt.savefig("TSNE_Baseline_{}_{}".format(dataset,decoder))
    
   
