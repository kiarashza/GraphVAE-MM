#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 12:52:38 2020

@author: parmis
"""


from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix, f1_score, classification_report
from sklearn.metrics.cluster import normalized_mutual_info_score,adjusted_rand_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset
import plotter


def get_metrices(labels_test, labels_pred):
    accuracy = accuracy_score(labels_test, labels_pred)
    micro_recall = recall_score(labels_test, labels_pred, average='micro')
    macro_recall = recall_score(labels_test, labels_pred, average='macro')
    micro_precision = precision_score(labels_test, labels_pred, average='micro')
    macro_precision = precision_score(labels_test, labels_pred, average='macro')
    micro_f1 = f1_score(labels_test, labels_pred, average='micro')
    macro_f1 = f1_score(labels_test, labels_pred, average='macro')

    result = classification_report(labels_test, labels_pred, digits=4)
    conf_matrix = confusion_matrix(labels_test, labels_pred)
    return labels_test, labels_pred , accuracy, micro_recall, macro_recall, micro_precision, macro_precision , micro_f1, macro_f1, conf_matrix, result
    


def knn(features, labels):
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=42) #split into test and train
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(features_train, labels_train)
    labels_pred = clf.predict(features_test)
    return get_metrices(labels_test, labels_pred)



def logistiic_regression(features, labels):
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=42) #split into test and train
    clf = LogisticRegressionCV(
         Cs=10, cv=10, scoring="accuracy", verbose=False, multi_class="ovr", max_iter=10000
    )
    clf.fit(features_train, labels_train)
    labels_pred = clf.predict(features_test)
    return get_metrices(labels_test, labels_pred)


def kmeans(labels_true,labels_pred):
    nmi_arth=normalized_mutual_info_score(labels_true, labels_pred, average_method='arithmetic')
    nmi_geo=normalized_mutual_info_score(labels_true, labels_pred, average_method='geometric')
    nmi_min=normalized_mutual_info_score(labels_true, labels_pred, average_method='min')
    nmi_max=normalized_mutual_info_score(labels_true, labels_pred, average_method='max')
    ari=adjusted_rand_score(labels_true, labels_pred)
    return nmi_arth,nmi_geo,nmi_min,nmi_max,ari


def NN(features, labels, val_feature=None,  val_label=None, test_feature=None,  test_label=None, hidden_size = 64,num_epochs = 30, use_the_best_model=True):
    # Hyper-parameters 
    input_size = features.shape[1]
    # hidden_size = 64

    batch_size = 100
    learning_rate = 0.001
    
    num_classes = len(np.unique(labels, return_counts=False))
    if type(test_feature)!=np.ndarray:
        y = torch.Tensor(labels).type(torch.LongTensor)
        X = torch.Tensor(features)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    else:
        X_train = torch.Tensor(features)
        y_train = torch.Tensor(labels).type(torch.LongTensor)
        X_val = torch.Tensor(val_feature)
        y_val = torch.Tensor(val_label).type(torch.LongTensor)
        X_test = torch.Tensor(test_feature)
        y_test =torch.Tensor(test_label).type(torch.LongTensor)


    train_dataset = TensorDataset(X_train,y_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                batch_size=batch_size, 
                                                shuffle=False)
    val_dataset = TensorDataset(X_val,y_val)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                              batch_size=batch_size, 
                                              shuffle=False)
    
    test_dataset = TensorDataset(X_test,y_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=batch_size, 
                                              shuffle=False)
    
    
    
    
    # Fully connected neural network with one hidden layer
    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNet, self).__init__()
            np.random.seed(0)
            # torch.seed(0)
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

            self.input_size = input_size
            self.l1 = nn.Linear(input_size, hidden_size) 
            self.relu = nn.ReLU()
            self.l2 = nn.Linear(hidden_size, num_classes)  
        
        def forward(self, x):
            out = self.l1(x)
            out = self.relu(out)
            out = self.l2(out)
            # no activation and no softmax at the end
            return out
    
    model = NeuralNet(input_size, hidden_size, num_classes)
    loss_function = nn.CrossEntropyLoss()
    
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    plt = plotter.Plotter(functions=["loss"])
    # Train the model

    outputs = model(X_val)
    best_recorded_loss = criterion(outputs, y_val).detach().numpy()
    for epoch in range(num_epochs):
        model.train()
        train_loss, valid_loss = [], []
        for i, (features, labels) in enumerate(train_loader):  
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        ## evaluation part 
        model.eval()
        for features, labels in val_loader:
            output = model(features)
            loss = loss_function(output, labels)
            valid_loss.append(loss.item())
        # print ("Epoch:", epoch, "Training Loss: ", np.mean(train_loss), "Valid Loss: ", np.mean(valid_loss))

        outputs = model(X_train)
        train_loss = criterion(outputs, y_train)

        outputs = model(X_val)
        val_loss = criterion(outputs, y_val)
        plt.add_values(epoch, [train_loss.item()], [val_loss.item()])
        print(val_loss)
        if(best_recorded_loss>val_loss.detach().numpy() or epoch==1):
            print("saving model")
            best_recorded_loss = val_loss.detach().numpy()
            torch.save(model.state_dict(), "nn_model")


    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    if use_the_best_model:
        model = NeuralNet(input_size, hidden_size, num_classes)
        model.load_state_dict(torch.load("nn_model", map_location="cpu"))
    with torch.no_grad():
        labels_pred=torch.zeros(0,dtype=torch.long, device='cpu')
        labels_test=torch.zeros(0,dtype=torch.long, device='cpu')
        for features, labels in test_loader:
            outputs = model(features)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            # Append batch prediction results
            labels_pred=torch.cat([labels_pred,predicted.view(-1).cpu()])
            labels_test=torch.cat([labels_test,labels.view(-1).cpu()])
            
    return get_metrices(labels_test, labels_pred)
