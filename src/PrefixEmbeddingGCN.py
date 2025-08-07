import torch
import pandas as pd

from torch_geometric.data import Data
import torch.optim as optim
from torch.utils.data import DataLoader 
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
from torch.utils.data import Dataset

from torch_geometric.data import Batch

import optuna 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from itertools import combinations
import os
import numpy as np
from datetime import datetime, timedelta


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# Data preparation
def prepare_data(event_encode, core_encode, scaled_time_diffs):
    data_list_event = []
    for i in range(len(event_encode)):
        node_features = torch.tensor(event_encode[i], dtype=torch.float)
        node_core = torch.tensor(core_encode[i], dtype=torch.long)
        num_events = (node_core[:, 0] != -1).sum()
        
        edge_index = torch.tensor([[j, j+1] for j in range(num_events-1)], dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(scaled_time_diffs[i][:num_events-1], dtype=torch.float).view(-1, 1)
        
        #event_ids = node_core[:num_events].view(-1) 
        event_ids = node_core[:num_events]
        graph_data = Data(x=node_features[:num_events], edge_index=edge_index, edge_attr=edge_attr, event_ids=event_ids)
        graph_data.num_nodes = num_events
        data_list_event.append(graph_data)
    return data_list_event


class CustomDataset(Dataset):
    def __init__(self, event_features, sequence_features, y):
        self.event_features = event_features
        self.sequence_features = sequence_features
        self.y = y

    def __len__(self):
        return len(self.event_features)

    def __getitem__(self, idx):
        return self.event_features[idx], self.sequence_features[idx], self.y[idx]

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    for batch in loader:
        event_data, sequence_features, labels = batch
        event_data = event_data.to(device)
        sequence_features = sequence_features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        output = model(event_data, sequence_features)
        loss = criterion(output, labels)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
       
    accuracy = correct / len(loader.dataset)
    loss = total_loss / len(loader.dataset)
    return loss, accuracy

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            event_data, sequence_features, labels = batch
            event_data = event_data.to(device)
            sequence_features = sequence_features.to(device)
            labels = labels.to(device)
            
            output = model(event_data, sequence_features)
            loss = criterion(output, labels)
            total_loss += loss.item() * labels.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
    
    accuracy = correct / len(loader.dataset)
    loss = total_loss / len(loader.dataset)
    return loss, accuracy

class EarlyStopping: 
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_loss_updated = False

    def __call__(self, val_loss):
        self.best_loss_updated = False
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_loss_updated = True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

def custom_collate(batch):
    event_data, seq_features, labels = zip(*batch)
    return (Batch.from_data_list(event_data),
            torch.stack(seq_features),
            torch.tensor(labels))

class PrefixGCNClassifier(nn.Module):
    def __init__(self, 
                 num_event_features, 
                 gcn_hidden_dims,
                 num_embedding_features, 
                 embedding_dims,
                 gcn_hidden_dims_embedding, 
                 gcn_hidden_dims_concat,
                 num_sequence_features, 
                 fc_hidden_dims,
                 fc_hidden_dims_concat, 
                 output_dim):
        super(PrefixGCNClassifier, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=num_embedding_features, embedding_dim=embedding_dims)
        self.gcn_embed = GCNConv(embedding_dims, gcn_hidden_dims_embedding)
        self.gcn_event = GCNConv(num_event_features, gcn_hidden_dims)
        self.gcn_concat = GCNConv(gcn_hidden_dims + gcn_hidden_dims_embedding, gcn_hidden_dims_concat)

        self.seq_proj = nn.Linear(num_sequence_features, fc_hidden_dims)
        self.concat_proj = nn.Linear(gcn_hidden_dims_concat + fc_hidden_dims, fc_hidden_dims_concat)

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(fc_hidden_dims_concat, output_dim)
        )

    def forward(self, data, sequence_features):
        # embedding path
        d = self.embedding(data.event_ids.squeeze(-1))  # ensure shape: [N,] not [N, 1]
        d = self.gcn_embed(d, data.edge_index, edge_weight=data.edge_attr)

        # event features path
        f = data.x
        f[f == -1] = 0  # replace -1 with 0s to avoid invalid input to GCN
        f = self.gcn_event(f, data.edge_index, edge_weight=data.edge_attr)

        # concat GCN outputs
        x = torch.cat([d, f], dim=1)
        x = self.gcn_concat(x, data.edge_index, edge_weight=data.edge_attr)
        graph_emb = global_mean_pool(x, data.batch)

        # sequence-level features
        seq_out = self.seq_proj(sequence_features)
        seq_out_concat = torch.cat([graph_emb, seq_out], dim=1)
        seq_out_concat = self.concat_proj(seq_out_concat)

        # classification head
        out = self.classifier(seq_out_concat)
        return out  # NO softmax here

def f1_eva(model, loader, device, k=3):
    model.eval()
    correct_topk = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            event_data, sequence_features, labels = batch
            event_data = event_data.to(device)
            sequence_features = sequence_features.to(device)
            labels = labels.to(device)
            
            output = model(event_data, sequence_features)            
            pred = output.argmax(dim=1)
            #correct += pred.eq(labels).sum().item()
            # Append for F1 score calculation
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())    
                        
            topk_preds = torch.topk(output, k=k, dim=1).indices  # shape: [batch_size, k]

            # Check if the label is in the top-k predictions
            correct_topk += sum([labels[i] in topk_preds[i] for i in range(labels.size(0))])
            total += labels.size(0)
    
    class_report = classification_report(all_labels, all_preds, digits=4)
    topk_accuracy = correct_topk / total

    return class_report, topk_accuracy

def get_misclassified_samples(model, loader, device):
    model.eval()
    errors = []

    with torch.no_grad():
        for batch in loader:
            event_data, sequence_features, labels = batch
            event_data = event_data.to(device)
            sequence_features = sequence_features.to(device)
            labels = labels.to(device)

            outputs = model(event_data,sequence_features)
            preds = outputs.argmax(dim=1)

            for i in range(labels.size(0)):
                if preds[i] != labels[i]:
                    errors.append({
                        "pred": preds[i].item(),
                        "label": labels[i].item(),
                        "sequence_feats": sequence_features[i].cpu().numpy(),
                        "event_feats": event_data[i].x.cpu().numpy() if hasattr(event_data[i], 'x') else None,
                        "embedding_feats": event_data[i].event_ids.cpu().numpy() if hasattr(event_data[i], 'x') else None
                    })
    return errors

def cluster_errors(errors, num_clusters, use='sequence_feats', method='pca'):
    """
    Cluster misclassified samples with automatic dimensionality handling.
    
    Args:
        errors: List of error dictionaries from get_misclassified_samples()
        num_clusters: Number of clusters to find
        use: Which features to use ('sequence_feats', 'event_feats', or 'embedding_feats')
        method: Dimensionality reduction method ('pca' or 'tsne')
    """
    # Extract features and labels
    features = []
    labels = []
    
    for e in errors:
        if e[use] is not None:
            features.append(e[use].flatten())
            labels.append((e['label'], e['pred']))
    
    if not features:
        print(f"No valid {use} features found in errors")
        return None
    
    features = np.array(features)
    print(f"Feature matrix shape: {features.shape}")
    
    # Handle case where we have fewer dimensions than requested components
    n_samples, n_features = features.shape
    effective_components = min(2, n_features, n_samples-1) if n_samples > 1 else 1
    
    # Dimensionality reduction
    if method == 'pca':
        if n_features < 2:
            print("Not enough features for PCA (need ≥2), using raw features")
            reduced = features[:, :effective_components]
        else:
            reducer = PCA(n_components=effective_components)
            reduced = reducer.fit_transform(features)
    elif method == 'tsne':
        if len(features) < 2:
            print("Not enough samples for t-SNE (need ≥2), using raw features")
            reduced = features[:, :effective_components]
        else:
            reducer = TSNE(n_components=2, perplexity=min(30, len(features)-1))
            reduced = reducer.fit_transform(features)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")
    
    # Clustering (only if we have enough samples)
    if len(features) >= num_clusters:
        kmeans = KMeans(n_clusters=min(num_clusters, len(features)), random_state=0)
        cluster_ids = kmeans.fit_predict(features)
    else:
        print(f"Not enough samples ({len(features)}) for {num_clusters} clusters")
        cluster_ids = np.zeros(len(features))
    
    # Visualization
    plt.figure(figsize=(10, 8))
    # Extract feature type for title (gets 'event' from 'event_feats')
    feature_type = use.split('_')[0] if '_' in use else use
    
    if reduced.shape[1] == 1:
        # 1D case - plot along x-axis with random y jitter
        x = reduced[:, 0]
        y = np.random.uniform(-0.1, 0.1, size=len(x))
        scatter = plt.scatter(x, y, c=cluster_ids, cmap='tab10', alpha=0.6)
        plt.xlabel(f"Principal Component 1 ({feature_type} features)")
        plt.yticks([])
        plt.title(f"Error Pattern Distribution ({feature_type} features)\n1D Projection")
    else:
        # 2D plot
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], 
                              c=cluster_ids, cmap='tab10', alpha=0.6)
        plt.xlabel(f"Principal Component 1 ({feature_type} features)")
        plt.ylabel(f"Principal Component 2 ({feature_type} features)")
        plt.title(f"Error Pattern Clusters ({feature_type} features)\n{method.upper()} Projection")

    plt.colorbar(scatter, label='Cluster ID')
    
    # Add some annotations for small datasets
    if len(labels) <= 100:
        for i, coord in enumerate(reduced):
            true, pred = labels[i]
            if reduced.shape[1] == 1:
                plt.text(coord[0], y[i], f"{true}→{pred}", fontsize=8, ha='center')
            else:
                plt.text(coord[0], coord[1], f"{true}→{pred}", fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    return cluster_ids