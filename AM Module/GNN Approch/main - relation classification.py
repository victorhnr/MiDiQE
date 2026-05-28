# -*- coding: utf-8 -*-
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

NUM_EPOCHS = 500
model_name = "roberta-large"
max_length = 76

target_dataset = "AAEC"

print(f"GNN Component Classification {target_dataset} - {NUM_EPOCHS} epochs - Encoder type {model_name}")

import pandas as pd
import re
import statistics

# Contruct a GNN model
import dgl.nn as dglnn

class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.fc = nn.LSTM(in_dim, out_dim, bidirectional=False)
        # equation (2)
        self.attn_fc = nn.LSTM(2*out_dim, 1, bias=False)

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z)[0]
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        # equation (1)
        z = self.fc(h)
        self.g.ndata['z'] = z[0]
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))

class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.g = g
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(g, int(hidden_dim * num_heads), out_dim, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, h):
        h = self.dropout(h)
        h = self.layer1(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.layer2(h)
        h = F.relu(h)
        return h


class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.Wg = nn.LSTM(2*in_features, in_features)
        self.Wl = nn.LSTM(2*76, in_features)
        self.W2 = nn.Linear(2*in_features, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        f_u = edges.src['feat']
        h_v = edges.dst['h']
        f_v = edges.dst['feat']
        lang = self.Wl(torch.cat([f_u, f_v], 1))[0]
        graph = self.Wg(torch.cat([h_u, h_v], 1))[0]
        score = self.W2(torch.cat([lang, graph], 1))
        return {'score': score}

    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


class Model(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super().__init__()
        self.gat = GAT(g, in_dim, hidden_dim, hidden_dim, num_heads)
        self.pred = MLPPredictor(hidden_dim, out_dim)
    def forward(self, x):
        h = self.gat(x)
        return self.pred(g, h)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = "./dataset/"

# Training
df_component_train = pd.read_csv(path+"component_train.csv", header=0)[["text_id", "component_tokens", "labels"]]
df_component_train = df_component_train[df_component_train.text_id.str.contains(target_dataset)]
df_component_train["tokens"]= df_component_train.component_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())

df_relation_train = pd.read_csv(path+"relation_train.csv", header=0)[["text_id", "source_tokens", "target_tokens", "labels"]]
df_relation_train = df_relation_train[df_relation_train.text_id.str.contains(target_dataset)]
df_relation_train["source_tokens"]= df_relation_train.source_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())
df_relation_train["target_tokens"]= df_relation_train.target_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())

# Validation
df_component_val = pd.read_csv(path+"component_val.csv", header=0)[["text_id", "component_tokens", "labels"]]
df_component_val = df_component_val[df_component_val.text_id.str.contains(target_dataset)]
df_component_val["tokens"]= df_component_val.component_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())

df_relation_val = pd.read_csv(path+"relation_val.csv", header=0)[["text_id", "source_tokens", "target_tokens", "labels"]]
df_relation_val = df_relation_val[df_relation_val.text_id.str.contains(target_dataset)]
df_relation_val["source_tokens"]= df_relation_val.source_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())
df_relation_val["target_tokens"]= df_relation_val.target_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())

# Test
df_component_test = pd.read_csv(path+"component_test.csv", header=0)[["text_id", "component_tokens", "labels"]]
df_component_test = df_component_test[df_component_test.text_id.str.contains(target_dataset)]
df_component_test["tokens"]= df_component_test.component_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())

df_relation_test = pd.read_csv(path+"relation_test.csv", header=0)[["text_id", "source_tokens", "target_tokens", "labels"]]
df_relation_test = df_relation_test[df_relation_test.text_id.str.contains(target_dataset)]
df_relation_test["source_tokens"]= df_relation_test.source_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())
df_relation_test["target_tokens"]= df_relation_test.target_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())

from sklearn.metrics import f1_score


num_attacks = df_relation_train[df_relation_train.labels == "attack"].shape[0] + df_relation_train[df_relation_train.labels == "Against"].shape[0]
num_sup = df_relation_train[df_relation_train.labels == "support"].shape[0] + df_relation_train[df_relation_train.labels == "For"].shape[0]

class_weights = torch.Tensor([1.0, float(num_sup/num_attacks)]).to(device)

def set_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)
    torch.use_deterministic_algorithms = True


def evaluate(g, features, labels, mask, model):
  # model.eval()
  with torch.no_grad():
    logits = model(features)
    logits = logits[mask]
    labels = labels[mask]
    _, indices = torch.max(logits, dim=1)
    f1 = f1_score(average="micro", y_true=list(labels.to("cpu")), y_pred=list(indices.to("cpu")))
    f1_macro = f1_score(average="macro", y_true=list(labels.to("cpu")), y_pred=list(indices.to("cpu")))
    return {"f1_micro": f1, "f1_macro": f1_macro}

data = {
    "train": {
        "component": df_component_train,
        "relation": df_relation_train
    },
    "test": {
        "component": df_component_test,
        "relation": df_relation_test
    },
    "val": {
        "component": df_component_val,
        "relation": df_relation_val
    },
}

from transformers import AutoTokenizer



tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True, use_fast=False)

possible_labels = {
    "Premise": 0,
    "Claim": 1,
    "MajorClaim": 2,
}

possible_relations = {
    "support": 0,
    "For": 0,
    "attack": 1,
    "Against": 1
}


node_id = 0
node_features = list()
node_labels = list()
source_nodes = list()
target_nodes = list()
edge_labels = list()
train_mask = list()
val_mask = list()
test_mask = list()
e_train_mask = list()
e_val_mask = list()
e_test_mask = list()
text_nodes = dict()

for key in data.keys():

    df_component = data[key]["component"]
    text_ids = df_component.text_id.unique()
    df_relation = data[key]["relation"]

    for num, text_id in enumerate(text_ids):
        df_c = df_component[df_component.text_id == text_id]
        df_r = df_relation[df_relation.text_id == text_id]
        components = df_c.to_dict('records')
        relations = df_r.to_dict('records')

        for component in components:
            to_be_token = f"{component['component_tokens']}"
            node_tokenized = tokenizer.encode(to_be_token, padding="max_length", max_length=max_length)
            node_label = possible_labels[component["labels"]]
            node_key = f"{text_id}: {component['tokens']}"
            text_nodes[node_key] = {
                "node_text_id": text_id,
                "node_id": node_id,
                "node_text": component["component_tokens"],
                "node_tokenized": node_tokenized,
                "node_label": node_label
            }
            node_features.append(node_tokenized)

            node_labels.append(node_label)
            train_mask.append(key == "train")
            val_mask.append(key == "val")
            test_mask.append(key == "test")
            node_id = node_id + 1

        for relation in relations:
            source_key = f"{text_id}: {relation['source_tokens']}"
            target_key = f"{text_id}: {relation['target_tokens']}"
            edge_label = possible_relations[relation["labels"]]
            source_nodes.append(text_nodes[source_key]["node_id"])
            target_nodes.append(text_nodes[target_key]["node_id"])
            edge_labels.append(edge_label)
            e_train_mask.append(key == "train")
            e_val_mask.append(key == "val")
            e_test_mask.append(key == "test")

import dgl
import torch

# Creating graphs
g = dgl.graph((source_nodes, target_nodes))
g.ndata['feat'] = torch.Tensor(node_features)
g.ndata['label'] = torch.Tensor(node_labels).to(torch.long)
g.edata['label'] = torch.Tensor(edge_labels).to(torch.long)
g.ndata["train_mask"] = torch.Tensor(train_mask).bool()
g.ndata["val_mask"] = torch.Tensor(val_mask).bool()
g.ndata["test_mask"] = torch.Tensor(test_mask).bool()
g.edata["train_mask"] = torch.Tensor(e_train_mask).bool()
g.edata["val_mask"] = torch.Tensor(e_val_mask).bool()
g.edata["test_mask"] = torch.Tensor(e_test_mask).bool()
g = g.int().to(device)

del df_component_train
del df_relation_train
del df_relation_test
del df_component_test
del df_component_val
del df_relation_val
del data
del source_nodes
del target_nodes

# Training GNN

macro = list()
micro = list()
node_features = g.ndata['feat']
edge_label = g.edata['label']
train_mask = g.edata['train_mask']
val_mask = g.edata["val_mask"]
n_train_mask = g.ndata['train_mask']
n_val_mask = g.edata["val_mask"]

seeds = [0, 9, 27, 42, 1871, 1977, 1994, 2000, 2020, 2023]
path=f"./models/{target_dataset}/component/model.pth"

macro = list()
f1 = list()

for i, seed in enumerate(seeds):
  print(f"{i+1}/{len(seeds)}")
  set_seeds(seed)
  in_size = node_features.shape[1]
  out_size = 2
  model_edge = Model(g, in_size, int(in_size/2), out_size, 2).to(device)
  loss_fcn = nn.CrossEntropyLoss(weight=class_weights)
  opt = torch.optim.Adam(model_edge.parameters(), lr=2e-3, weight_decay=0.01)
  best_metric = 0
  best_loss = 0
  for epoch in range(NUM_EPOCHS):
    model_edge.train()
    pred = model_edge(node_features)
    loss = loss_fcn(pred[train_mask], edge_label[train_mask].to(torch.long))
    opt.zero_grad()
    loss.backward()
    opt.step()
    acc = evaluate(g, node_features, edge_label, val_mask, model_edge)
    f_macro = acc["f1_macro"]
    if best_loss == 0:
      best_loss = loss
    if f_macro >= best_metric:
      if f_macro == best_metric and loss > best_loss:
        pass
      else:
        best_metric = f_macro
        best_loss = loss
        torch.save(model_edge.state_dict(), path)
    print(f"Epoch {epoch+1}, loss: {loss}, metrics: {acc}")
  model_edge.load_state_dict(torch.load(path))
  acc = evaluate(g, node_features, edge_label, g.edata["test_mask"], model_edge)
  macro.append(acc["f1_macro"])
  f1.append(acc["f1_micro"])
  print(f"\nModel test metrics: {acc} \n")

print(f"Average F1: {statistics.fmean(f1)}")
print(f"Average Macro: {statistics.fmean(macro)}")
