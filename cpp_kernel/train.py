import os
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from generate_train_test_datasets import load_pickle, generate_text_graph
from models import GCNNet


def load_datasets():
    print("Loading data...")
    df_data_path = "./data/df_data.pkl"
    graph_path = "./data/text_graph.pkl"
    if not os.path.isfile(df_data_path) or not os.path.isfile(graph_path):
        print("Building datasets and graph from raw data... Note this will take quite a while...")
        generate_text_graph()
    df_data = load_pickle("df_data.pkl")
    G = load_pickle("text_graph.pkl")
    
    print("Building adjacency and degree matrices...")
    A = nx.to_numpy_matrix(G, weight="weight")
    A = A + np.eye(G.number_of_nodes())
    degrees = []
    for d in G.degree(weight=None):
        if d == 0:
            degrees.append(0)
        else:
            degrees.append(d[1]**(-0.5))
    degrees = np.diag(degrees)
    X = np.eye(G.number_of_nodes())     # Features are just identity matrix
    A_hat = degrees@A@degrees
    f = X
    
    print("Splitting labels for training and inferring...")
    ### stratified test samples
    test_idxs = []
    for b_id in df_data["b"].unique():
        dum = df_data[df_data["b"] == b_id]
        if len(dum) >= 4:
            test_idxs.extend(list(np.random.choice(dum.index, size=round(0.1*len(dum)), replace=False)))

    # select only certain labelled nodes for semi-supervised GCN
    selected = []
    for i in range(len(df_data)):
        if i not in test_idxs:
            selected.append(i)

    labels_selected = [l for idx, l in enumerate(df_data["b"]) if idx in selected]
    labels_not_selected = [l for idx, l in enumerate(df_data["b"]) if idx not in selected]
    f = torch.from_numpy(f).float().to('cuda')
    print("Split into %d train and %d test lebels." % (len(labels_selected), len(labels_not_selected)))
    return f, X, A_hat, selected, labels_selected, labels_not_selected, test_idxs


def evaluate(output, labels_e):
    _, labels = output.cpu().max(1)
    labels = labels.numpy()
    return sum([(e-1) for e in labels_e] == labels)/len(labels)


if __name__ == "__main__":
    f, X, A_hat, selected, labels_selected, labels_not_selected, test_idxs = load_datasets()

    net = GCNNet(A_hat, X.shape[1], hidden_size_1=330, hidden_size_2=130, num_classes=66).to('cuda')
    criterion = nn.CrossEntropyLoss().to('cuda')
    optimizer = optim.Adam(net.parameters(), lr=0.1)
    
    print("Starting training process...")
    net.train()
    evaluation_trained = []
    for e in range(1000):
        optimizer.zero_grad()
        output = net(f)
        loss = criterion(output[selected], torch.tensor(labels_selected).long().to('cuda') -1)
        loss.backward()
        optimizer.step()
        if e % 20 == 0:
            # Evaluation
            net.eval()
            with torch.no_grad():
                pred_labels = net(f)
                trained_accuracy = evaluate(output[selected], labels_selected)
                untrained_accuracy = evaluate(pred_labels[test_idxs], labels_not_selected)
            print("[Epoch %d]: Evaluation accuracy of trained nodes: %.7f" % (e, trained_accuracy))
            print("[Epoch %d]: Evaluation accuracy of test nodes: %.7f" % (e, untrained_accuracy))
            net.train()
        optimizer.step()
        
    print("Training finished.")