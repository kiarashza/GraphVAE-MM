import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# Fixing random state for reproducibility
np.random.seed(17680801)


# N = 50
# x = np.random.rand(N)
# y = np.random.rand(N)
# colors = np.random.rand(N)
# area = (30 * np.random.rand(N))**2  # 0 to 15 point radii
#
# plt.scatter(x, y, s=area, c=colors, alpha=0.5, label="group")
# plt.show()

def scatter_plot(x, y, label, x_label=None, y_label=None, color=None, size = None, marker='o'):
    if color== None:
        color = list(np.random.rand(1))*len(y)
    else:
        if len(color)==1:
            color = [color]*len(y)

    if size== None:
        size = [220]*len(y)
    else:
        size = [size]*len(y)


    plt.scatter(x, y,label=label, s=size, c=color, marker=marker )
    # plt.legend()
    # plt.show()

x = []
y= []
#grid ["DGLFRM", "DGLFRM_kernel","FC","Graphit", "MolGAN","GRaphRNN", GRAN]
x.append([1.21+1.69+0.95+4.65e-8,
          0.29+0.36+.004+7.11e-11,
           0.6683434164351226 + 0.9166756024236481 + 1.1829290880172039e-09 + 0.47795558608554334,
          1.15+1.38+0.71+1.32e-7,
        1.28+1.45+0.86+2.01e-4,
0.2+8e-4+9e-3+4.77e-10,
          0.00020350383183043164 + 0.0007532475990124077 + 2.4968760392596323e-10 + 1.908820149787438e-05])
y_ = [3111,3701,2896,3518,1903,8120, 36000]
y_ = [i/3000 for i in y_]# train time
y_ = [0.007, 0.009,.01,.71,.67, 240/32, 16.39] # test time

y.append(y_)
#
# #Lobstrer
# x.append([1.17+1.56+0.91+8.83e-9, .07+0.24+0.08+3.25e-9,0.15+0.32+0.14+1.49e-5,0.50+1.27+0.51+2.58e-8,0.73+1.50+0.52+3.21e-6,0.31601008928483987+ 0.12150526603225442 + 2.763758736068489e-05 + 0.1977557878636842])
x.append([1.17+1.56+0.91+8.83e-9,
          .07+0.24+0.08+3.25e-9,
          1.2069365125448477 + 1.6567490928327155 + 6.116650408394264e-08 + 0.9629536717358085,
          0.50+1.27+0.51+2.58e-8,
          0.73+1.50+0.52+3.21e-6,
          0.531703064278104 +0.016682225706711185 + 5.030186540633252e-07 + 0.14102576635960284,
          0.20192341930496194 + 0.3364085341605061 + 2.0757983909547306e-07 + 0.3662906431963764])


y_ = [902,1003,982,1307,1567, 4072.20,153*60] #
y_ = [i/3000 for i in y_]
y_ = [0.001, 0.001, 0.03, .62,.52, 73.71/32, 8.29] # test time

y.append(y_)
# #protein
#grid ["DGLFRM", "DGLFRM_kernel","FC","Graphit", "MolGAN","GRaphRNN", GRAN]
x.append([0.80+0.95+0.80+2.33e-6,
          0.80+0.06+0.85+1.84e-8,
          0.8751901666892531 + 1.7465036275667756 + 4.656198939967382e-06 + 0.8125335680398535,
          1.67+1.24+1.09+1.78e-2,
          1.06+0.67+0.93+1.65e-3,
          0.8276922170823197 + 0.880401626449386 + 1.4698442227789599e-06 + 0.38816746847553585,
          0.1298129228067626 + 0.4424109404015142 + 6.593974921642598e-09 + 0.036483706339843325])
y_ = [49814,62455,47845,61864,42368, 150103.41014504433, 1045*60]
y_ = [i/3000 for i in y_]
y_ =[0.012, 0.012, 0.017, 0.89,1.21, 550.9710731506348/32, 89.10]

y.append(y_)



# color = np.array([['b','g','r'],]*3).transpose()
# rows = [mpatches.Patch(color=color[i, 0],hatch="O") for i in range(3)]
# label_row = ['1', '2', '3']
# plt.legend(rows , label_row , loc=2)
#
# plt.show()

colors = ["cornflowerblue","black" , "silver", "pink", "springgreen", "coral", "lightblue"]
marker = ['o', 'v', '^']
marker_label = ["Grid", "Lobster", "Protein"]
if 1==2:


    for i,_ in enumerate(y):
        scatter_plot(x[i], y[i],label="hi", marker = marker[i], color=colors )

    texts = ["FC", "FC_kernel","DGLFRM","Graphit", "MolGAN", "GraphRNN", "GRAN"]
    patches = [ plt.plot([],[], marker="o", ms=10, ls="", mec=None, color=colors[i],
                label="{:s}".format(texts[i]) )[0]  for i in range(len(texts)) ]
    mrker = [plt.plot([], [], marker[i], markerfacecolor='w',
                        markeredgecolor='k',label="{:s}".format(marker_label[i]))[0] for i in range(3)]

    plt.legend(handles=patches+mrker,numpoints=1 )

    plt.xlabel('Sum of MMD over the graph statistics')
    plt.ylabel('Average Trainning Time per Epoch(Second)')

    plt.show()

#===============================================================================================
# plting each dataset in differen plt
texts = ["FC", "FC_kernel", "DGLFRM", "Graphite", "MolGAN", "GraphRNN","GRAN"]
texts = ["DGLFRM", "DGLFRM_kernel","FC","Graphite", "MolGAN","GraphRNN", "GRAN"]
marker  = ['o', 'v', '^','p','D','h','H']
dataset=["grid", "lobster","DD"]
for i,_ in enumerate(y):
    for j in range(len(marker)):
        scatter_plot([x[i][j]], [y[i][j]], marker = marker[j], color=colors[j], label="" )

    patches = [plt.plot([], [], marker=marker[i], ms=10, ls="", mec=None, color=colors[i],
                        label="{:s}".format(texts[i]))[0] for i in range(len(texts))]
    plt.legend(handles=patches, numpoints=1, fontsize=15)

    plt.xlabel(' Sum of the 4 MMD terms',fontsize=15)
    plt.ylabel('Average generation time of a mini-batch (S)',fontsize=15)
    plt.savefig("performance_test"+dataset[i], bbox_inches="tight")
    plt.show()
    plt.close()
print("wait")