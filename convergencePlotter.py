#!/usr/bin/env python
# coding: utf-8

# In[260]:


#HETEROGENEOUS
# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.markers as markers



# save the train loss for comparing the convergence
import json
# kernel_file_name = 'small_grid_kernel_train_loss.txt'
# kernel_file_name = 'lobster_kernel_elbo_loss.txt'
# kipf_file_name = 'lobster_kipf_train_loss.txt'

def mini_batch_mereger(loss_list, num_step_in_epoch):
    counter = 0
    merged_list = []
    stp = []
    for x in loss_list:

        if counter==num_step_in_epoch:
            merged_list.append(np.average(stp))
            stp = []
            counter = 0
        stp.append(x)
        counter += 1
    return  merged_list




# kipf_file_name = 'convergence/lobster_kipf_CrossEntropyLoss.txt'
# kernel_file_name = 'convergence/lobster_kernel_CrossEntropyLoss.txt'

# kipf_file_name = 'convergence/grid_kipf_CrossEntropyLoss.txt'
# kernel_file_name = 'convergence/grid_kernel_CrossEntropyLoss.txt'




# plt.ylim(0.74, 0.98)
# multiple line plot
#plt.plot( 'L', 'IMDB', data=df, marker='^', markerfacecolor='blue', markersize=10, color='blue', linewidth=2, linestyle='dashed', label='IMDB')
DataSets = ["DD", "lobster", "grid"]
colros = ['red', 'green', 'blue']
for i,dataset in enumerate(DataSets):
    kipf_file_name = 'convergence/'+dataset+'_kipf_CrossEntropyLoss.txt'
    kernel_file_name = 'convergence/'+dataset+'_kernel_CrossEntropyLoss.txt'

    with open(kernel_file_name, "r") as fp:
        kenel_loss = json.load(fp)

    with open(kipf_file_name, "r") as fp:
        kipf_loss = json.load(fp)
        kipf_loss = [x*100 for x in kipf_loss]
        kenel_loss = [x * 100 for x in kenel_loss]
    if dataset == "DD":
        kenel_loss = mini_batch_mereger(kenel_loss, 8)
        kipf_loss = mini_batch_mereger(kipf_loss, 8)

    # Data
    df = pd.DataFrame({'x': np.array(list(range(len(kipf_loss)))), 'kernel': np.array(kenel_loss),
                       'kipf': np.array(kipf_loss)
                       })

    plt.plot( 'x', 'kernel', data=df, color="lightcoral", linewidth=1,
               label='DGLFRM-Kernel')
    plt.plot( 'x', 'kipf', data=df, color='#4b0082', linestyle=(0, (5, 5)), linewidth=1, label='DGLFRM',)

    # plt.ylim(0, 0.25)
    plt.legend(fontsize=18)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Reconstruction loss', fontsize=18)
    plt.ylim(0, 0.25)
    plt.savefig(dataset+'ConvergenceComparision')
    plt.show()
    plt.close()
    # plt.plot( 'x', 'kipf', data=df, marker='s', markerfacecolor='orange', markersize=7, color='green', linewidth=1,
#          linestyle='dashed', label='DBLP')

plt.show()

plt.legend()
# naming the x axis
plt.xlabel('Epoch', fontsize=15)
# naming the y axis
plt.ylabel('Reconstruction Loss', fontsize=15)
# xi = [16, 32, 64, 128,256]
# L = [16, 32, 64, 128,256]
# plt.xticks(xi, L)
plt.show()
# giving a title to my graph
# plt.title('Homogeneous Graphs')
plt.savefig('GridConvergenceComparision')


#!/usr/bin/env python
# coding: utf-8

# In[260]:


# #HETEROGENEOUS
# # libraries
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import matplotlib.markers as markers
#
#
#
# # save the train loss for comparing the convergence
# import json
# # kernel_file_name = 'small_grid_kernel_train_loss.txt'
# # kernel_file_name = 'lobster_kernel_elbo_loss.txt'
# # kipf_file_name = 'lobster_kipf_train_loss.txt'
#
# def mini_batch_mereger(loss_list, num_step_in_epoch):
#     counter = 0
#     merged_list = []
#     stp = []
#     for x in loss_list:
#
#         if counter==num_step_in_epoch:
#             merged_list.append(np.average(stp))
#             stp = []
#             counter = 0
#         stp.append(x)
#         counter += 1
#     return  merged_list
# dataset = "DD"
# kipf_file_name = 'convergence/DD_kipf_CrossEntropyLoss.txt'
# kernel_file_name = 'convergence/DD_kernel_CrossEntropyLoss.txt'
#
#
# # kipf_file_name = 'convergence/lobster_kipf_CrossEntropyLoss.txt'
# # kernel_file_name = 'convergence/lobster_kernel_CrossEntropyLoss.txt'
#
# # kipf_file_name = 'convergence/grid_kipf_CrossEntropyLoss.txt'
# # kernel_file_name = 'convergence/grid_kernel_CrossEntropyLoss.txt'
# with open(kernel_file_name, "r") as fp:
#     kenel_loss = json.load(fp)
#
# with open(kipf_file_name, "r") as fp:
#     kipf_loss = json.load(fp)
# if dataset == "DD":
#     kenel_loss = mini_batch_mereger(kenel_loss, 7)
#     kipf_loss = mini_batch_mereger(kipf_loss, 7)
#
# # Data
# df=pd.DataFrame({'x': np.array(list(range(len(kipf_loss)))), 'kernel' : np.array(kenel_loss),
#                 'kipf' : np.array(kipf_loss)
#                  })
#
# # df=pd.DataFrame({'d': [1,2,3,4], 'IMDB' : [0.85561,0.87,0.8852,0.8882],
# #                 'DBLP' : [0.91,0.9170,0.9200,0.9317],
# #                 'ACM': [0.9400,0.9694,0.9600,0.9579] })
# #
#
#
# # plt.ylim(0.74, 0.98)
# # multiple line plot
# #plt.plot( 'L', 'IMDB', data=df, marker='^', markerfacecolor='blue', markersize=10, color='blue', linewidth=2, linestyle='dashed', label='IMDB')
#
# for dataset in DataSets:
#
# plt.plot( 'x', 'kernel', data=df, color='red', linewidth=1,
#           linestyle=(0, (1, 10)), label='FC-Kernel')
# plt.plot( 'x', 'kipf', data=df, color='green', linewidth=1, label='FC',)
# # plt.plot( 'x', 'kipf', data=df, marker='s', markerfacecolor='orange', markersize=7, color='green', linewidth=1,
# #          linestyle='dashed', label='DBLP')
#
#
#
# plt.legend()
# # naming the x axis
# plt.xlabel('Epoch', fontsize=15)
# # naming the y axis
# plt.ylabel('Reconstruction Loss', fontsize=15)
# # xi = [16, 32, 64, 128,256]
# # L = [16, 32, 64, 128,256]
# # plt.xticks(xi, L)
# plt.show()
# # giving a title to my graph
# # plt.title('Homogeneous Graphs')
# plt.savefig('GridConvergenceComparision')
