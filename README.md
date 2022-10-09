# GraphVAE-MM
This is the original implementation of Micro and Macro Level Graph Modeling for Graph Variational Auto-Encoders.

## Code Overview
main.py includes the training pipeline and also micro-macro objective functions implementation. Source codes for loading real graph datasets and generating synthetic graphs are included in data.py. All the Python packages used in our experiments are provided in environment.yml. Generated graph samples for each of the datasets are provided in the "ReportedResult/" directory,both in the pickle and png format. This directory also includes the log files and hyperparameters details used to train the GraphVAE-MM on each of the datasets.

## Comparing Generated **Triangle Grid** Graphs by GraphVAE-MM and GraphVAE
The first and second column shows generated graphs by GraphVAE-MM and GraphVAE. Samples are selected randomly. The generated samples for each of the datasets are provided at ReportedResult/.
| Tests| GraphVAE-MM             |  GraphVAE |
|:-------------------------:|:-------------------------:|:-------------------------:|
![]() |![](https://github.com/GraphVAE-MM/GraphVAE-MM/blob/main/ReportedResult/MMD_AvePool_FC_triangular_grid_graphGeneration_KernelAugmentedWithTotalNumberOfTrianglesBFSTrue200001652199375.4136705/01710triangular_grid_ConnectedComponnents.png) |![](https://github.com/GraphVAE-MM/GraphVAE-MM/blob/main/ReportedResult/MMD_AvePool_FC_triangular_grid_graphGeneration_kipfBFSTrue200001651972897.5996404/121500triangular_grid_ConnectedComponnents.png)|
![]() |![](https://github.com/GraphVAE-MM/GraphVAE-MM/blob/main/ReportedResult/MMD_AvePool_FC_triangular_grid_graphGeneration_KernelAugmentedWithTotalNumberOfTrianglesBFSTrue200001652199375.4136705/01810triangular_grid_ConnectedComponnents.png)| ![](https://github.com/GraphVAE-MM/GraphVAE-MM/blob/main/ReportedResult/MMD_AvePool_FC_triangular_grid_graphGeneration_kipfBFSTrue200001651972897.5996404/01810triangular_grid_ConnectedComponnents.png)
![]() |![](https://github.com/GraphVAE-MM/GraphVAE-MM/blob/main/ReportedResult/MMD_AvePool_FC_triangular_grid_graphGeneration_KernelAugmentedWithTotalNumberOfTrianglesBFSTrue200001652199375.4136705/12720triangular_grid_ConnectedComponnents.png) | ![](https://github.com/GraphVAE-MM/GraphVAE-MM/blob/main/ReportedResult/MMD_AvePool_FC_triangular_grid_graphGeneration_kipfBFSTrue200001651972897.5996404/101040triangular_grid_ConnectedComponnents.png)
![]() |![](https://github.com/GraphVAE-MM/GraphVAE-MM/blob/main/ReportedResult/MMD_AvePool_FC_triangular_grid_graphGeneration_KernelAugmentedWithTotalNumberOfTrianglesBFSTrue200001652199375.4136705/101040triangular_grid_ConnectedComponnents.png) | ![](https://github.com/GraphVAE-MM/GraphVAE-MM/blob/main/ReportedResult/MMD_AvePool_FC_triangular_grid_graphGeneration_kipfBFSTrue200001651972897.5996404/101170triangular_grid_ConnectedComponnents.png)
![]() |![](https://github.com/GraphVAE-MM/GraphVAE-MM/blob/main/ReportedResult/MMD_AvePool_FC_triangular_grid_graphGeneration_KernelAugmentedWithTotalNumberOfTrianglesBFSTrue200001652199375.4136705/16910triangular_grid_ConnectedComponnents.png) | ![](https://github.com/GraphVAE-MM/GraphVAE-MM/blob/main/ReportedResult/MMD_AvePool_FC_triangular_grid_graphGeneration_kipfBFSTrue200001651972897.5996404/111020triangular_grid_ConnectedComponnents.png)


## Model Convergence
The figures below show the convergence of GraphVAE-MM for each term in the objective function,  the local and global graph properties loss and KL penalty. For brevity, only two datasets are illustrated. These plots are generated automatically after each "Vis_step" iteration, Vis_step is an input argument. Here kernel1 to kernel 5 corresponds to 1 to 5 Step transition probability kernels and loss is the sum of the terms. 
![This is triangle graph train loss convergence](https://github.com/GraphVAE-MM/GraphVAE-MM/blob/main/Concerge_triangle.png)
![This is ogb graph train loss convergence](https://github.com/GraphVAE-MM/GraphVAE-MM/blob/main/Concerge_ogb.png)



## Cite
Please cite our paper if you use this code in your research work.
