import torch
import pandas as pd

from torch_geometric.data import Data
import torch.optim as optim

import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
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
import matplotlib.pyplot as plt
import seaborn as sns

from torch_geometric.nn import GATConv

from sklearn.preprocessing import MultiLabelBinarizer
from collections import defaultdict

from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader 

from torch_scatter import scatter_min, scatter_max
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

# Data preparation

def prepare_data_core(event_encode, core_encode, scaled_time_diffs, node_times):
    data_list_event = []
    data_list_core = []
    for i in range(len(event_encode)):
        node_features = torch.tensor(event_encode[i], dtype=torch.float)
        node_core = torch.tensor(core_encode[i], dtype=torch.long)
        time = torch.tensor(node_times[i], dtype=torch.float)
        num_events = (node_core[:, 0] != -1).sum()
        
        edge_index = torch.tensor([[j, j+1] for j in range(num_events-1)], dtype=torch.long).t().contiguous()
        #edge_attr = torch.tensor(scaled_time_diffs[i][:num_events-1], dtype=torch.float).view(-1, 1)
        
        event_ids = node_core[:num_events]
        
        graph_data = Data(x=node_features[:num_events], edge_index=edge_index, event_ids=event_ids)
        graph_data.num_nodes = num_events
        graph_data.time = time
        data_list_event.append(graph_data)
    
    return data_list_event

def prepare_data_y(event_encode, y_encode):
    data_list = []
    for i in range(len(event_encode)):
        node_features = torch.tensor(event_encode[i], dtype=torch.float)
        labels = torch.tensor(y_encode[i], dtype=torch.long)
        num_events = (node_features[:, 0] != -1).sum()
        data_list.append(labels[:num_events])
    return data_list

class CustomDataset(Dataset):
    def __init__(self, event_features, y):
        self.event_features = event_features
        self.y = y

    def __len__(self):
        return len(self.event_features)

    def __getitem__(self, idx):
        return self.event_features[idx], self.y[idx]
    
def custom_collate_fn(batch):
    event_data_list, label_list = zip(*batch)
    batch_event = Batch.from_data_list(event_data_list)
    padded_labels = pad_sequence([lbl.squeeze(1) for lbl in label_list], batch_first=True, padding_value=-1)

    return batch_event, padded_labels

def normalize_attention_minmax(attn_raw, edge_indices):
    selected = attn_raw[edge_indices]
    min_vals = selected.min(dim=0, keepdim=True)[0]
    max_vals = selected.max(dim=0, keepdim=True)[0]
    normed = (selected - min_vals) / (max_vals - min_vals + 1e-8)
    return normed.mean(dim=1)

def min_max_normalize(x):
    x = x.squeeze()
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def normalize_edge_scores(score_tensor, edge_indices):
    selected = score_tensor[edge_indices]
    min_vals = selected.min(dim=0, keepdim=True)[0]
    max_vals = selected.max(dim=0, keepdim=True)[0]
    normed = (selected - min_vals) / (max_vals - min_vals + 1e-8)
    return normed.mean(dim=1) 


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total_tokens = 0
    #all_labels = []
    #all_preds = []

    for event_data, labels in loader:
        event_data = event_data.to(device)
        labels = labels.to(device)  # [batch_size, max_seq_len]

        optimizer.zero_grad()
        output = model(event_data)  # [total_nodes, vocab_size]
    
        # Align shape
        output = output.view(-1, output.size(-1))  # [total_nodes, vocab_size]
        labels = labels.view(-1)                   # [total_nodes]
        
        # Apply mask before loss
        mask = labels != -1
        #output = output[mask]
        labels = labels[mask]
        
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
        
        # Accuracy
        pred = output.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total_tokens += labels.size(0)
        
        # Append for F1 score calculation
        #all_labels.extend(labels.cpu().numpy())
        #all_preds.extend(pred.cpu().numpy())
        
    accuracy = correct / total_tokens
    loss = total_loss / total_tokens
    
    return loss, accuracy
    
def evaluate(model, loader, criterion, device, return_attention=False):
    model.eval()
    total_loss = 0
    correct = 0
    total_tokens = 0

    all_attn_maps = []
    #all_event_ids = []

    with torch.no_grad():
        for event_data, labels in loader:
            event_data = event_data.to(device)
            labels = labels.to(device)
            #e_ids =  event_data.event_ids.view(-1)

            if return_attention:
                output, attn_data = model(event_data, return_attention=True)
                #all_event_ids.extend(e_ids)
            else:
                output = model(event_data)

            output = output.view(-1, output.size(-1))
            labels = labels.view(-1)

            mask = labels != -1
            labels = labels[mask]

            loss = criterion(output, labels)
            total_loss += loss.item() * labels.size(0)

            pred = output.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total_tokens += labels.size(0)

            if return_attention:
                batch_vector = attn_data['batch']
                num_graphs = batch_vector.max().item() + 1

                for graph_idx in range(num_graphs):
                    node_mask = batch_vector == graph_idx
                    node_indices = node_mask.nonzero(as_tuple=True)[0]

                    edge_mask = (
                        node_mask[attn_data['edge_index'][0]] &
                        node_mask[attn_data['edge_index'][1]]
                    )
                    edge_indices = edge_mask.nonzero(as_tuple=True)[0]

                    if edge_indices.numel() == 0:
                        continue

                    # Reindex nodes
                    old2new = {old.item(): new for new, old in enumerate(node_indices)}
                    edge_index_sub = attn_data['edge_index'][:, edge_indices].clone()
                    for j in range(edge_index_sub.size(1)):
                        edge_index_sub[0, j] = old2new[edge_index_sub[0, j].item()]
                        edge_index_sub[1, j] = old2new[edge_index_sub[1, j].item()]

                    # Normalize attention
                    alpha_embed_norm = normalize_attention_minmax(attn_data['alpha_embed'], edge_indices)
                    alpha_event_norm = normalize_attention_minmax(attn_data['alpha_event'], edge_indices) 
                    alpha_final_norm = normalize_attention_minmax(attn_data['alpha_final'], edge_indices)

                    # Normalize decay & edge scores
                    decay_embed = min_max_normalize(attn_data['decay_embed'][edge_indices]) if 'decay_embed' in attn_data else None
                    decay_event = min_max_normalize(attn_data['decay_event'][edge_indices]) if 'decay_event' in attn_data else None
                    decay_final = min_max_normalize(attn_data['decay_final'][edge_indices]) if 'decay_final' in attn_data else None


                    graph_attn = {
                        'alpha_embed': alpha_embed_norm,
                        'alpha_event': alpha_event_norm,
                        'alpha_final': alpha_final_norm,
                        'decay_embed': decay_embed,
                        'decay_event': decay_event,
                        'decay_final': decay_final,
                        'edge_index': edge_index_sub,
                        'time': attn_data['time'][edge_indices],
                        'batch': batch_vector[node_indices],
                        'graph_idx': graph_idx 
                    }
                    all_attn_maps.append(graph_attn)

    accuracy = correct / total_tokens
    loss = total_loss / total_tokens

    if return_attention:
        #return loss, accuracy, all_attn_maps, all_event_ids
        return loss, accuracy, all_attn_maps
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

class TimeAwareGATConv(GATConv):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, lambda_decay=0.1, **kwargs):
        super(TimeAwareGATConv, self).__init__(in_channels, out_channels, heads=heads, concat=concat,  **kwargs)
        
        self.lambda_decay = lambda_decay
        self.att = nn.Parameter(torch.Tensor(heads, 2 * out_channels))
        nn.init.xavier_uniform_(self.att)
        self._decay = None

    def edge_attention(self, x_i, x_j, edge_attr):
        # x_i, x_j: [E, H, C]
        cat_ij = torch.cat([x_i, x_j], dim=-1)  # [E, H, 2C]
        alpha = torch.einsum('ehc,hc->eh', cat_ij, self.att)  # [E, H]
        #print("Raw attention scores:", alpha)
        alpha = F.leaky_relu(alpha, self.negative_slope)

        if edge_attr is not None:
            time_diff = edge_attr
            decay = torch.exp(-self.lambda_decay * time_diff).unsqueeze(-1)  # [E, 1]
            alpha = alpha * decay 
            #decay = torch.clamp(1.0 - self.lambda_decay * time_diff, min=0.01).unsqueeze(-1)
            #alpha = alpha + torch.log(decay + 1e-6) # Time-aware weighting
            #decay = self.lambda_decay * time_diff.unsqueeze(-1)  # [E, 1], linear form
            #alpha = alpha - decay
            self._decay = decay.detach().cpu()
        return alpha

    def message(self, x_j, x_i, edge_attr, index, ptr, size_i):
        alpha = self.edge_attention(x_i, x_j, edge_attr)
        #alpha = softmax(alpha, index, ptr, size_i)  # [E, H]
        self._alpha = alpha  # Save for visualization
        return x_j * alpha.unsqueeze(-1)  # [E, H, C]

    def forward(self, x, edge_index, edge_attr=None, return_attention=False):
        H, C = self.heads, self.out_channels

        # Project input and reshape for multi-head
        x = self.lin(x)  # [N, H*C]
        x = x.view(-1, H, C)  # [N, H, C]

        # Run propagate
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

        if self.concat:
            out = out.view(-1, H * C)  # [N, H*C]
        else:
            out = out.mean(dim=1)  # [N, C]

        if return_attention:
            return out, self._alpha  # [N, *], [E, H]
        return out


class DualGATTimeAwareModel(nn.Module):
    def __init__(self, num_event_features, num_embedding_features, embedding_dims, 
                 gat_hidden_dim_event, gat_hidden_dim_embed, gat_hidden_dim_concat, 
                 output_dim, num_heads, lambda_decay):
        super(DualGATTimeAwareModel, self).__init__()
        
        edge_dim =1

        self.embedding = nn.Embedding(num_embeddings=num_embedding_features, embedding_dim=embedding_dims)

        self.gat_embed = TimeAwareGATConv(embedding_dims, gat_hidden_dim_embed, heads=num_heads, concat=True, edge_dim=edge_dim, lambda_decay=lambda_decay)
        self.gat_event = TimeAwareGATConv(num_event_features, gat_hidden_dim_event, heads=num_heads, concat=True, edge_dim=edge_dim, lambda_decay=lambda_decay)

        concat_input_dim = (gat_hidden_dim_embed + gat_hidden_dim_event) * num_heads
        self.gat_concat = TimeAwareGATConv(concat_input_dim, gat_hidden_dim_concat, heads=num_heads, concat=True, edge_dim=edge_dim, lambda_decay=lambda_decay)

        final_dim = gat_hidden_dim_concat * num_heads
        self.fc = nn.Linear(final_dim, output_dim)

    def forward(self, data_event, return_attention=False):
        edge_attr = data_event.time
        #time = data_event.time
        edge_index = data_event.edge_index
        
        x_embed = self.embedding(data_event.event_ids.view(-1))  
        
        if return_attention:
            x_embed, attn_embed = self.gat_embed(x_embed,  edge_index=edge_index, edge_attr=edge_attr, return_attention=True)
        else:
            x_embed = self.gat_embed(x_embed, edge_index=edge_index, edge_attr=edge_attr)

        x_event = data_event.x
        if return_attention:
            x_event, attn_event = self.gat_event(x_event, edge_index=edge_index, edge_attr=edge_attr, return_attention=True)
        else:
            x_event = self.gat_event(x_event, edge_index=edge_index, edge_attr=edge_attr)

        x = torch.cat([x_embed, x_event], dim=1)
        if return_attention:
            x, attn_final = self.gat_concat(x, edge_index=edge_index, edge_attr=edge_attr, return_attention=True)
        else:
            x = self.gat_concat(x, edge_index=edge_index, edge_attr=edge_attr)

        out = self.fc(x)

        if return_attention:
            return out, {
                'alpha_embed': attn_embed.detach().cpu(),
                'alpha_event': attn_event.detach().cpu(),
                'alpha_final': attn_final.detach().cpu(),
                'edge_index': edge_index.detach().cpu(),
                'time': edge_attr.detach().cpu(),
                'decay_embed': self.gat_embed._decay.detach().cpu() if self.gat_embed._decay is not None else None,
                'decay_event': self.gat_event._decay.detach().cpu() if self.gat_event._decay is not None else None,
                'decay_final': self.gat_concat._decay.detach().cpu() if self.gat_concat._decay is not None else None,
                'batch': data_event.batch.detach().cpu()
            }
        return out

def vectorized_find_longest(attention_data):
    lengths = torch.tensor([s['time'].shape[0] for s in attention_data])
    max_idx = torch.argmax(lengths).item()
    return attention_data[max_idx], max_idx, lengths[max_idx].item()

def get_samples_in_node_range(attention_data, min_nodes, max_nodes, max_samples=None):
    """
    Returns samples and their indices where node count ∈ [min_nodes, max_nodes]
    
    Args:
        attention_data: List of attention dicts
        min_nodes: Minimum node count (inclusive)
        max_nodes: Maximum node count (inclusive)
        max_samples: Optional limit on number of samples to return
    
    Returns:
        Tuple of (matching_samples, sample_indices)
    """
    matched = [(idx, s) for idx, s in enumerate(attention_data) 
               if min_nodes <= s['time'].shape[0] <= max_nodes]
    
    if max_samples:
        matched = matched[:max_samples]
    
    # Unzip into separate lists
    if matched:
        sample_indices, samples = zip(*matched)
        return list(samples), list(sample_indices)
    return [], []

   
def predict(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for event_data, labels in loader:
            event_data = event_data.to(device)
            #embed_data = embed_data.to(device)
            labels = labels.to(device)

            output = model(event_data)
            output = output.view(-1, output.size(-1))
            labels = labels.view(-1)

            mask = labels != -1
            labels = labels[mask]

            preds = output.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_outputs.append(output.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_outputs = torch.cat(all_outputs)
    return all_preds, all_labels, all_outputs

def top_k_accuracy(output, target, k=3):
    """Compute Top-k accuracy."""
    top_k_preds = output.topk(k, dim=1).indices  # [batch_size, k]
    target = target.view(-1, 1)  # [batch_size, 1]
    correct = top_k_preds.eq(target).sum().item()
    total = target.size(0)
    return correct / total

def predict_per_sequence(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for event_data, labels in loader:
            event_data = event_data.to(device)
            labels = labels.to(device)  # shape: [batch_size, padded_len]

            output = model(event_data)  # shape: [total_nodes, vocab_size]

            # Compute true lengths per sequence (non-padded elements)
            valid_lengths = (labels != -1).sum(dim=1).tolist()

            # Get predicted labels
            preds_flat = output.argmax(dim=1)  # [total_nodes]

            # Recover per-sequence predictions using lengths
            idx = 0
            for i, length in enumerate(valid_lengths):
                if length == 0:
                    all_preds.append([])
                    all_labels.append([])
                    continue
                
                # Get the non-padded portion of labels for this sequence
                non_padded_labels = labels[i][:length]  # This automatically excludes padding
                
                # Get corresponding predictions
                preds_seq = preds_flat[idx:idx+length].cpu().tolist()
                labels_seq = non_padded_labels.cpu().tolist()
                
                all_preds.append(preds_seq)
                all_labels.append(labels_seq)
                idx += length

    return all_preds, all_labels

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def average_bleu_score(preds_seq, labels_seq, max_n=4):
    """
    Computes average BLEU score with smoothing for empty n-gram cases.
    
    Args:
        preds_seq: List of predicted sequences (each a list of tokens)
        labels_seq: List of reference sequences (each a list of tokens)
        max_n: Maximum n-gram order to use (default: 4)
    
    Returns:
        Average BLEU score across all non-empty pairs
    """
    if len(preds_seq) != len(labels_seq):
        raise ValueError("Predictions and labels must have same length")
    
    bleu_scores = []
    smoother = SmoothingFunction().method1  # Choose smoothing method
    
    for pred, label in zip(preds_seq, labels_seq):
        if len(label) == 0 or len(pred) == 0:
            continue  # skip empty sequences
            
        # Handle case where prediction is shorter than max_n
        actual_n = min(max_n, len(pred), len(label))
        weights = tuple([1/actual_n] * actual_n)  # Uniform weights
        
        try:
            score = sentence_bleu([label], pred, 
                                 weights=weights,
                                 smoothing_function=smoother)
            bleu_scores.append(score)
        except:
            continue  # skip problematic cases
    
    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

from pyxdameraulevenshtein import damerau_levenshtein_distance as dld

def compute_dls_and_exact_match(pred_seqs, true_seqs):
    """Compute sequence similarity metrics with proper edge case handling.
    
    Args:
        pred_seqs: List of predicted sequences (e.g., [[1,2,3], [4,5]])
        true_seqs: List of ground truth sequences (same structure as pred_seqs)
    
    Returns:
        avg_dls: Average Damerau-Levenshtein similarity (0-1, higher is better)
        exact_match_acc: Fraction of perfectly matched sequences
    """
    if len(pred_seqs) != len(true_seqs):
        raise ValueError("pred_seqs and true_seqs must have equal length")
    
    dls_scores = []
    exact_match_count = 0
    total_sequences = 0

    for pred, true in zip(pred_seqs, true_seqs):
        # Skip empty sequences if they exist
        if not pred and not true:
            continue
            
        total_sequences += 1
        
        # Handle empty sequence cases
        if not pred or not true:
            similarity = 0.0  # Consider empty and non-empty completely dissimilar
        else:
            dist = dld(pred, true)
            max_len = max(len(pred), len(true))
            similarity = 1 - dist / max_len
        
        dls_scores.append(similarity)

        # Exact match requires identical content and length
        if len(pred) == len(true) and all(p == t for p, t in zip(pred, true)):
            exact_match_count += 1

    # Handle case where all sequences were empty
    if total_sequences == 0:
        return 0.0, 0.0

    avg_dls = sum(dls_scores) / total_sequences
    exact_match_acc = exact_match_count / total_sequences

    return avg_dls, exact_match_acc

def sequence_level_top_k_accuracy(model, loader, device, k=3):
    """Calculate what percentage of complete sequences have ALL predictions in top-k"""
    model.eval()
    total_correct_sequences = 0
    total_sequences = 0

    with torch.no_grad():
        for event_data, labels in loader:
            event_data = event_data.to(device)
            #embed_data = embed_data.to(device)
            labels = labels.to(device)  # shape: [batch_size, padded_len]

            output = model(event_data)  # [total_nodes, vocab_size]
            top_k_preds = output.topk(k, dim=1).indices  # [total_nodes, k]
            
            # Get true lengths per sequence
            valid_lengths = (labels != -1).sum(dim=1).tolist()
            
            ptr = 0  # Pointer into flattened output
            for i, length in enumerate(valid_lengths):
                if length == 0:
                    continue  # Skip empty sequences
                
                # Get only non-padded labels for this sequence
                seq_labels = labels[i][:length]  # [length]
                
                # Get corresponding top-k predictions
                seq_top_k = top_k_preds[ptr:ptr+length]  # [length, k]
                
                # Check if ALL labels are in top-k predictions
                correct = True
                for pos in range(length):
                    if seq_labels[pos] not in seq_top_k[pos]:
                        correct = False
                        break
                
                if correct:
                    total_correct_sequences += 1
                total_sequences += 1
                ptr += length

    return total_correct_sequences / total_sequences if total_sequences > 0 else 0.0

def analyze_sequence_errors(model, loader, device, k=3):
    """Analyze prediction errors per position and type.
    
    Args:
        model: Your trained model
        loader: Data loader
        device: torch device
        k: Top-k accuracy to consider
        
    Returns:
        pos_errors: Error counts per position
        error_types: Dictionary of {(predicted, true): count}
        seq_length_stats: Statistics about sequence lengths
    """
    # First get per-sequence predictions
    preds_seq, labels_seq = predict_per_sequence(model, loader, device)
    
    error_positions = []
    error_types = defaultdict(int)
    seq_lengths = []
    
    for pred, label in zip(preds_seq, labels_seq):
        if not pred or not label:  # Skip empty sequences
            continue
            
        seq_lengths.append(len(label))
        
        for pos in range(len(label)):
            true_label = label[pos]
            predicted_label = pred[pos]
            
            if predicted_label != true_label:
                error_positions.append(pos)
                error_types[(predicted_label, true_label)] += 1
    
    # Calculate statistics
    pos_errors = np.bincount(error_positions) if error_positions else np.array([])
    
    seq_length_stats = {
        'min': min(seq_lengths) if seq_lengths else 0,
        'max': max(seq_lengths) if seq_lengths else 0,
        'mean': np.mean(seq_lengths) if seq_lengths else 0,
        'median': np.median(seq_lengths) if seq_lengths else 0
    }
    
    # Print report
    print("\n=== Error Analysis Report ===")
    print(f"Total sequences analyzed: {len(preds_seq)}")
    print(f"Total sequences with errors: {len([1 for p,l in zip(preds_seq,labels_seq) if p != l])}")
    print(f"Sequence length stats: {seq_length_stats}")
    
    if len(pos_errors) > 0:
        print("\nError distribution by position:")
        for pos, count in enumerate(pos_errors):
            print(f"Position {pos}: {count} errors ({count/sum(pos_errors):.1%} of all errors)")
    
    if error_types:
        print("\nMost common error types:")
        sorted_errors = sorted(error_types.items(), key=lambda x: -x[1])[:10]
        for (pred, true), count in sorted_errors:
            print(f"Predicted {pred} instead of {true}: {count} times")
    
    return pos_errors, dict(error_types), seq_length_stats

def predict_per_sequence_with_probs(model, loader, device, k=1):
    model.eval()
    all_preds = []
    all_labels = []
    all_topk = []  # Stores top-k predictions for each position
    
    with torch.no_grad():
        for event_data, labels in loader:
            event_data = event_data.to(device)
            labels = labels.to(device)

            output = model(event_data)
            probs = torch.softmax(output, dim=1)
            topk_probs, topk_indices = torch.topk(probs, k, dim=1)
            
            # Compute true lengths per sequence
            valid_lengths = (labels != -1).sum(dim=1).tolist()
            
            idx = 0
            for i, length in enumerate(valid_lengths):
                if length == 0:
                    all_preds.append([])
                    all_labels.append([])
                    all_topk.append([])
                    continue
                    
                # Get only non-padded portion
                non_padded_labels = labels[i][:length]  # shape: [length]
                
                # Get corresponding predictions
                seq_topk = topk_indices[idx:idx+length].cpu().tolist()
                seq_labels = non_padded_labels.cpu().tolist()
                
                all_preds.append([x[0] for x in seq_topk])  # Top-1 predictions
                all_labels.append(seq_labels)
                all_topk.append(seq_topk)  # All top-k predictions
                idx += length

    return all_preds, all_labels, all_topk

def sequence_level_top_k_analysis(preds_topk, labels):
    """
    Args:
        preds_topk: List of lists, where each sublist contains top-k predictions 
                   for each position in a sequence (from predict_per_sequence_with_probs)
        labels: List of lists containing true labels (without padding)
    
    Returns:
        top_k_accuracy: Fraction of sequences where ALL true labels are in top-k predictions
        error_stats: Dictionary of error statistics
    """
    total_sequences = len(labels)
    correct_sequences = 0
    error_stats = {
        'wrong_positions': [],
        'common_errors': defaultdict(int)
    }

    for seq_topk, seq_labels in zip(preds_topk, labels):
        if not seq_labels:  # Skip empty sequences
            continue
            
        sequence_correct = True
        for pos, (topk_preds, true_label) in enumerate(zip(seq_topk, seq_labels)):
            if true_label not in topk_preds:
                sequence_correct = False
                error_stats['wrong_positions'].append(pos)
                error_stats['common_errors'][(topk_preds[0], true_label)] += 1
                # No break here if you want full error analysis
        
        if sequence_correct:
            correct_sequences += 1

    # Calculate statistics
    accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0.0
    
    # Position-wise error distribution
    pos_errors = np.bincount(error_stats['wrong_positions']) if error_stats['wrong_positions'] else []
    error_stats['position_errors'] = {pos: count for pos, count in enumerate(pos_errors)}
    
    # Most common errors
    error_stats['top_errors'] = dict(sorted(
        error_stats['common_errors'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:5])

    return accuracy, error_stats

def show_error_sequences(preds_topk, labels, num=3):
    for i, (seq_topk, seq_labels) in enumerate(zip(preds_topk, labels)):
        if any(true not in topk for topk, true in zip(seq_topk, seq_labels)):
            print(f"\nError Sequence #{i+1}:")
            print(f"True: {seq_labels}")
            print(f"Pred: {[topk[0] for topk in seq_topk]}")
            print("Mismatches:")
            for pos, (topk, true) in enumerate(zip(seq_topk, seq_labels)):
                if true not in topk:
                    print(f"Pos {pos}: True {true} not in {topk}")
            num -= 1
            if num == 0: break


import pandas as pd
from collections import defaultdict

def get_length_bins_from_attention(attention_data, bin_size=30, max_len=None, save_path=None):
    """
    Generate length bins and count samples per bin.

    Returns:
        length_ranges: list of (min_len, max_len, range_name)
        bin_df: pd.DataFrame with columns ['range_name', 'min_len', 'max_len', 'count']
    """
    lengths = [attn['edge_index'].shape[1] + 1 for attn in attention_data]
    if max_len is None:
        max_len = max(lengths)

    bin_edges = list(range(0,  ((max_len // bin_size) + 1) * bin_size + 1, bin_size))

    length_ranges = []
    bin_counts = defaultdict(int)

    for i in range(len(bin_edges) - 1):
        min_len = bin_edges[i]
        max_len_bin = bin_edges[i+1]
        range_name = f"{min_len}-{max_len_bin - 1}"
        length_ranges.append((min_len, max_len_bin, range_name))

    # Count samples per bin
    for length in lengths:
        for min_len, max_len_bin, range_name in length_ranges:
            if min_len <= length < max_len_bin:
                bin_counts[range_name] += 1
                break

    # Build DataFrame
    bin_df = pd.DataFrame([
        {'range_name': rn, 'min_len': mn, 'max_len': mx - 1, 'count': bin_counts.get(rn, 0)}
        for (mn, mx, rn) in length_ranges
    ])

    if save_path:
        bin_df.to_csv(save_path, index=False)
        print(f"✅ Bin count saved to: {save_path}")

    return length_ranges, bin_df


def get_quantile_bins_from_attention(attention_data, num_bins=25):
    lengths = [attn['edge_index'].shape[1] + 1 for attn in attention_data]
    quantiles = np.quantile(lengths, np.linspace(0, 1, num_bins+1))
    
    length_ranges = []
    for i in range(num_bins):
        min_len = int(np.floor(quantiles[i]))
        max_len = int(np.ceil(quantiles[i+1])) + 1
        range_name = f"{min_len}-{max_len-1}"
        length_ranges.append((min_len, max_len, range_name))
    return length_ranges

import pandas as pd
from collections import defaultdict

def compute_importance_stats(attention_data, length_ranges, save_path=None):
    """
    Compute importance statistics for Rank 1, 2, 3 nodes per length range.
    Optionally save to CSV.
    
    Returns: 
        df_summary: pd.DataFrame
        range_stats: {range: {rank: {node: count}}}
        range_importances: {range: {rank: [imp1, imp2, ...]}}
    """
    range_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    range_importances = defaultdict(lambda: defaultdict(list))
    df_rows = []

    for attn_dict in attention_data:
        seq_length = attn_dict['edge_index'].shape[1]  # edges = seq_len - 1
        seq_len_adjusted = seq_length + 1  # node count
        
        for min_len, max_len, range_name in length_ranges:
            if min_len <= seq_len_adjusted < max_len:
                importance = attn_dict['decay_final'] * attn_dict['alpha_final']
                ranked_nodes = torch.argsort(importance, descending=True).tolist()
                
                for rank, node in enumerate(ranked_nodes[:3], start=1):
                    range_stats[range_name][rank][node] += 1
                    range_importances[range_name][rank].append(importance[node].item())
                break

    def get_min_len(bin_name):
        return int(bin_name.split('-')[0].replace('<', '').replace('>', ''))

    for range_name in sorted(range_stats.keys(), key=get_min_len):
        row_data = {'Length Range': range_name}
        for rank in [1, 2, 3]:
            node_counts = range_stats[range_name][rank]
            if not node_counts:
                row_data[f'Rank{rank}_Node'] = None
                row_data[f'Rank{rank}_Dominance'] = 0.0
                row_data[f'Rank{rank}_MeanImp'] = 0.0
                continue
            
            top_node, count = max(node_counts.items(), key=lambda x: x[1])
            dominance = (count / sum(node_counts.values())) * 100
            mean_imp = torch.tensor(range_importances[range_name][rank]).mean().item()

            row_data[f'Rank{rank}_Node'] = top_node
            row_data[f'Rank{rank}_Dominance'] = round(dominance, 2)
            row_data[f'Rank{rank}_MeanImp'] = round(mean_imp, 4)

        df_rows.append(row_data)

    df_summary = pd.DataFrame(df_rows)

    if save_path:
        df_summary.to_csv(save_path, index=False)
        print(f"📄 Saved importance summary to: {save_path}")
    
    return range_stats, range_importances, df_summary

import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import PowerNorm

def plot_heatmap_from_stats(range_stats, range_importances, save_path=None):
    """
    Generate publication-quality heatmap with journal-ready formatting.
    
    Args:
        range_stats: {range: {rank: {node: count}}}
        range_importances: {range: {rank: [imp1, imp2, ...]}}
        save_path: Path to save figure (optional)
    """
    # Prepare heatmap data
    heatmap_data = defaultdict(dict)
    all_nodes = set()
    
    # Collect all nodes
    for range_name in range_stats:
        for rank in range_stats[range_name]:
            all_nodes.update(range_stats[range_name][rank].keys())
    
    max_node = max(all_nodes) if all_nodes else 0
    
    # Process data
    for range_name in sorted(range_stats.keys(), key=lambda x: int(x.split('-')[0].replace('<', '').replace('>', ''))):
        for node in range(max_node + 1):
            importances = []
            for rank in [1, 2, 3]:
                if node in range_stats[range_name][rank]:
                    importances.extend(range_importances[range_name][rank])
            heatmap_data[range_name][node] = np.mean(importances) if importances else np.nan
    
    df_heatmap = pd.DataFrame(heatmap_data).T
    
    # Create optimized blue-orange colormap
    colors = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6",
              "#4292c6", "#2171b5", "#08519c", "#08306b",
              "#fff5eb", "#fee6ce", "#fdd0a2", "#fdae6b",
              "#fd8d3c", "#f16913", "#d94801", "#8c2d04"]
    cmap = LinearSegmentedColormap.from_list("blue_orange", colors, N=256)
    
    # Create figure with journal-quality settings
    fig, ax = plt.subplots(figsize=(15, 8), dpi=300)  # Increased height for better proportions
    plt.rcParams['font.family'] = 'Arial'  # Journal-preferred font
    plt.rcParams['pdf.fonttype'] = 42  # Ensure text editable in PDF
    
    # Create heatmap with enhanced parameters
    hm = sns.heatmap(
        df_heatmap, 
        cmap=cmap,
        cbar_kws={
            'label': 'Mean Importance Score',
            'shrink': 0.75,
            'pad': 0.02,
            'aspect': 10
        },
        vmin=0,
        vmax=1,
        square=False,
        linewidths=0.5,  # Slightly thicker lines
        linecolor='white',
        annot=False,
        norm=PowerNorm(gamma=0.4),
        ax=ax
    )
    
    # Enhanced colorbar
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10, width=0.5)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.set_ylabel('Mean Importance Score', 
                      fontsize=12, 
                      rotation=270, 
                      labelpad=20,
                      fontweight='normal')
    
    # Annotate important cells with improved styling
    if len(df_heatmap.columns) <= 40:
        threshold = np.nanpercentile(df_heatmap.values, 85)
        for y in range(df_heatmap.shape[0]):
            for x in range(df_heatmap.shape[1]):
                val = df_heatmap.iloc[y, x]
                if not np.isnan(val) and val >= threshold:
                    ax.text(x + 0.5, y + 0.5, f'{val:.2f}',
                           ha='center', va='center', fontsize=9,
                           color='white' if val > 0.5 else '#333333',
                           bbox=dict(boxstyle='round', 
                                    facecolor='white' if val <= 0.5 else '#444444',
                                    alpha=0.7,
                                    edgecolor='none',
                                    pad=0.1))
    
    # Axis styling
    ax.set_xlabel("Node Position", fontsize=12, labelpad=10)
    ax.set_ylabel("Sequence Length Range", fontsize=12, labelpad=10)
    
    # Tick adjustments
    ax.tick_params(axis='both', which='both', labelsize=10, 
                  length=3, width=0.5, pad=2)
    plt.xticks(rotation=45, ha='right')
    
    # Title with journal-style formatting
    plt.title("Node Importance Heatmap by Position and Sequence Length", 
             fontsize=14, pad=16, fontweight='semibold')
    
    # Borders and grid
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('#555555')
        spine.set_linewidth(0.8)
    
    # Alternative: Place inside axes coordinates
    ax.text(0.98, 0.98,  # x=98% of x-axis, y=98% of y-axis
        f"Range: {df_heatmap.min().min():.2f}-{df_heatmap.max().max():.2f}\n"
        f"Mean: {np.nanmean(df_heatmap.values):.2f} ± {np.nanstd(df_heatmap.values):.2f}",
        transform=ax.transAxes,  # Use axes coordinates
        ha='right', va='top', fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))
    
    plt.tight_layout()
    
    # Save in multiple formats if path provided
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save as PDF (vector format for journals)
        plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        
        # Save as TIFF (high-res for submission systems)
        plt.savefig(f"{save_path}.tiff", format='tiff', bbox_inches='tight', dpi=600)
        
        # Save as PNG (for review)
        plt.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=300)
    
    plt.show()

from matplotlib import patheffects
def plot_rank_dominance(range_stats, title="Top Node Dominance by Rank and Length Range", save_path=None):
    """
    Generate publication-quality bar plot of rank dominance with journal-ready formatting.
    
    Args:
        range_stats: {range: {rank: {node: count}}} 
        title: Plot title (optional)
        save_path: Path to save figure (without extension)
    """
    # Set up journal-quality figure
    plt.figure(figsize=(12, 7), dpi=300)  # Slightly larger for better proportions
    plt.rcParams['font.family'] = 'Arial'  # Journal-preferred font
    plt.rcParams['pdf.fonttype'] = 42  # Ensure text remains editable in PDF
    
    # Custom color palette
    palette = {
        "Rank 1": "#7A5195",  # purpley
        "Rank 2": "#009FDF",  # clear blue
        "Rank 3": "#FF6E54"   # blood orange
    }
    
    # Prepare DataFrame
    rank_dominance = []
    for range_name in sorted(range_stats.keys(), 
                           key=lambda x: int(x.split('-')[0].replace('<', '').replace('>', ''))):
        for rank in [1, 2, 3]:
            node_counts = range_stats[range_name][rank]
            if not node_counts:
                continue  # Skip empty ranks
            
            total = sum(node_counts.values())
            top_node, count = max(node_counts.items(), key=lambda x: x[1])
            rank_dominance.append({
                "Range": range_name,
                "Rank": f"Rank {rank}",
                "Dominance (%)": (count / total) * 100,
                "Top Node": top_node
            })
    
    df_rank = pd.DataFrame(rank_dominance)
    
    # Create plot with enhanced parameters
    ax = sns.barplot(
        data=df_rank, 
        x="Range", 
        y="Dominance (%)", 
        hue="Rank", 
        palette=palette,
        order=sorted(df_rank["Range"].unique(), 
                    key=lambda x: int(x.split('-')[0].replace('<', '').replace('>', ''))),
        edgecolor="white",
        linewidth=1.0,  # Slightly thicker borders
        saturation=0.95,
        err_kws={'linewidth': 1.0} # Error bar width if applicable
    )
    
    # Enhanced annotations
    for i, (_, row) in enumerate(df_rank.iterrows()):
        n_ranks = len(df_rank["Rank"].unique())
        bar_width = 0.8 / n_ranks
        x_pos = i // n_ranks - 0.4 + bar_width * (i % n_ranks) + bar_width/2
        
        # Smart text positioning
        text_y_pos = max(5, row["Dominance (%)"] / 2)  # Ensure minimum height
        
        ax.text(
            x_pos,
            text_y_pos,
            f"N{row['Top Node']}",  # More compact label
            ha='center',
            va='center',
            rotation=90,
            fontsize=9,  # Slightly larger
            color="white",
            fontweight='bold',
            path_effects=[patheffects.withStroke(linewidth=2, foreground="#333333")]  # Text outline
        )
    
    # Journal-quality formatting
    plt.title(title, fontsize=14, pad=20, fontweight='semibold')
    plt.ylim(0, 110)
    plt.xlabel("Sequence Length Range", fontsize=12, labelpad=10)
    plt.ylabel("Dominance (%)", fontsize=12, labelpad=10)
    
    # Enhanced legend
    legend = plt.legend(title="Rank", 
                       bbox_to_anchor=(1.02, 1), 
                       loc='upper left',
                       frameon=True,
                       framealpha=1,
                       edgecolor='#333333')
    legend.get_title().set_fontweight('semibold')
    
    # Add statistical summary inside axes (top right)
    ax.text(0.98, 0.98,  # x=98%, y=98% of axes coordinates
            f"Max: {df_rank['Dominance (%)'].max():.1f}%\n"
            f"Min: {df_rank['Dominance (%)'].min():.1f}%\n"
            f"Avg: {df_rank['Dominance (%)'].mean():.1f}%",
            transform=ax.transAxes,  # Critical - uses axes coordinates
            ha='right', va='top', 
            fontsize=10)
    
    # Grid and spines
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    plt.tight_layout()
    
    # Save in multiple formats if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Vector formats
        plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f"{save_path}.eps", format='eps', bbox_inches='tight')
        
        # High-res raster
        plt.savefig(f"{save_path}.tiff", format='tiff', bbox_inches='tight', dpi=600)
        
        # For review
        plt.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=300)
    
    plt.show()

def plot_importance_timeline(attention_data, min_length=100, max_length=None, figsize=(20, 8), save_path=None):
    """
    Publication-quality timeline plot with optimized annotations and visual hierarchy.
    
    Args:
        attention_data: List of attn_dicts (from evaluate() with return_attention=True)
        min_length: Minimum sequence length to consider (default: 100)
        max_length: Maximum sequence length to consider (optional)
        figsize: Figure dimensions (default: (20, 8))
        save_path: Path to save figure (without extension)
    """
    # Find first sequence meeting length criteria
    def length_check(attn_dict):
        length = attn_dict['edge_index'].max().item() + 1
        meets_min = length >= min_length
        meets_max = (max_length is None) or (length <= max_length)
        return meets_min and meets_max
    
    long_seq = next(
        (attn_dict for attn_dict in attention_data if length_check(attn_dict)),
        None
    )
    
    if long_seq is None:
        length_range = f">= {min_length}" if max_length is None else f"between {min_length}-{max_length}"
        print(f"No sequences found with length {length_range}")
        return
    
    # Prepare data
    time = long_seq['time'].cpu().numpy()
    decay = long_seq['decay_final'].cpu().numpy()
    attention = long_seq['alpha_final'].cpu().numpy()
    importance = decay * attention
    node_numbers = np.arange(1, len(time)+1)  # Nodes numbered from 1 to N
    
    # Additional attention components
    alpha_embed = long_seq['alpha_embed'].cpu().numpy()
    alpha_event = long_seq['alpha_event'].cpu().numpy()
    
    # Create figure with journal-quality proportions
    plt.figure(figsize=figsize, dpi=300)
    ax = plt.gca()
    
    # Calculate bar width based on time differences
    time_diffs = np.diff(time)
    avg_time_diff = np.mean(time_diffs) if len(time_diffs) > 0 else 0.02
    bar_width = avg_time_diff * 0.25
    
    # --- Visual Hierarchy Improvements ---
    # 1. Plot vertical lines first (background)
    for i, t in enumerate(time):
        alpha = 0.3 if i % max(1, len(time)//20) == 0 else 0.1
        ax.axvline(x=t, color='#888888', linestyle='-', alpha=alpha, linewidth=0.8)
    
    # 2. Plot attention bars (middle ground)
    for i, t in enumerate(time):
        ax.bar(t, alpha_embed[i], width=bar_width, color='#17becf', alpha=0.8,
              edgecolor='white', linewidth=0.5, label='Embed Attention' if i == 0 else "")
        ax.bar(t, alpha_event[i], width=bar_width, color='#e377c2', alpha=0.8,
              edgecolor='white', linewidth=0.5, label='Event Attention' if i == 0 else "")
    
    # 3. Plot curves (foreground)
    plt.plot(time, decay, label='Decay', color='#4e79a7', linestyle='--', alpha=0.7, linewidth=2.5)
    plt.plot(time, attention, label='Final Attention', color='#f28e2b', linestyle=':', alpha=0.9, linewidth=2.5)
    plt.plot(time, importance, label='Importance (decay × attention)', color='#59a14f', linewidth=3.5)
    
    # --- Optimized Annotation Placement Algorithm ---
    def get_non_overlapping_positions(times, y_max, num_labels=20):
        """Smart algorithm to find non-overlapping label positions"""
        positions = []
        time_range = max(times) - min(times)
        min_x_spacing = time_range * 0.05  # Minimum x spacing
        min_y_spacing = y_max * 0.08      # Minimum y spacing
        
        # Always include first, last, and top importance nodes
        key_indices = [0, len(times)-1, *np.argsort(importance)[-3:][::-1]]
        
        for i in sorted(set(key_indices + list(np.linspace(0, len(times)-1, num_labels, dtype=int)))):
            t = times[i]
            y_pos = y_max * 0.98  # Start at top
            
            # Find first non-overlapping position
            while any(abs(t - pos[0]) < min_x_spacing and abs(y_pos - pos[1]) < min_y_spacing 
                     for pos in positions):
                y_pos -= min_y_spacing
                if y_pos < y_max * 0.2:  # Don't go too low
                    y_pos = y_max * 0.98  # Reset to top and accept some overlap
                    break
            
            positions.append((t, y_pos, i))
        return positions

    # Get optimal label positions
    y_max = ax.get_ylim()[1]
    label_positions = get_non_overlapping_positions(time, y_max)
    
    # Plot node labels at optimized positions
    for t, y_pos, i in label_positions:
        ax.text(t, y_pos, f'N{node_numbers[i]}', 
               ha='center', va='top', rotation=45, fontsize=10,
               color='#333333', alpha=0.9, fontweight='semibold',
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # --- Top Event Highlighting ---
    top_indices = np.argsort(importance)[-3:][::-1]
    label_offsets = [0.95, 0.75, 0.55]  # Vertically staggered
    
    for idx, pos in zip(top_indices, label_offsets):
        t = time[idx]
        node = node_numbers[idx]
        
        # Check if this overlaps with existing node labels
        existing_labels = [(pos[0], pos[1]) for pos in label_positions]
        y_pos = y_max * pos
        while any(abs(t - x) < bar_width*3 and abs(y_pos - y) < y_max*0.1 
                 for (x, y) in existing_labels):
            y_pos -= y_max * 0.05
        
        ax.axvline(x=t, color='#e15759', linestyle='-', alpha=0.6, linewidth=2)
        ax.text(t, y_pos, f'Top{top_indices.tolist().index(idx)+1} (N{node})', 
               ha='center', va='center', rotation=0, fontsize=11,
               bbox=dict(facecolor='white', alpha=0.9, edgecolor='#e15759',
                         boxstyle='round,pad=0.3', linewidth=1.5))
    
    # --- Publication-Quality Formatting ---
    plt.gca().invert_xaxis()
    plt.xlabel("Normalized Time (1.0 = start, 0.0 = end)", fontsize=13, labelpad=10)
    plt.ylabel("Normalized Attention Score", fontsize=13, labelpad=10)
    
    # Dynamic title with statistical summary
    title = (f"Attention Component Dynamics\n"
             f"Sequence Length: {len(time)} nodes | "
             f"Max Importance: {importance.max():.2f} at N{node_numbers[np.argmax(importance)]}")
    plt.title(title, fontsize=15, pad=20, fontweight='semibold')
    
    # Professional legend
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([
        plt.Line2D([0], [0], color='#e15759', linewidth=2, alpha=0.6),
        plt.Line2D([0], [0], color='#888888', linewidth=0.8, alpha=0.3)
    ])
    labels.extend(['Top Events', 'All Nodes'])
    
    legend = plt.legend(handles, labels, 
               loc='center left', 
               bbox_to_anchor=(1.02, 0.5),
               framealpha=1,
               borderaxespad=0.,
               title='Components',
               title_fontsize=12,
               fontsize=11)
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_edgecolor('#cccccc')
    
    # Fine-tuned grid and spines
    ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    
    # Save in multiple formats if path provided
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Vector formats (editable in Illustrator)
        plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f"{save_path}.eps", format='eps', bbox_inches='tight', dpi=300)
        
        # High-resolution raster formats
        plt.savefig(f"{save_path}.tiff", format='tiff', bbox_inches='tight', dpi=600)
        plt.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=300)
    
    plt.show()
    return plt.gcf()
    return plt.gcf()

from scipy import stats
def plot_critical_windows(attention_data, length_ranges, figsize=(12, 6), compare_ranges=None, alpha=0.05, save_path=None):
    """
    Identifies important temporal windows across sequence lengths with publication-quality output.
    
    Args:
        attention_data: List of attention dictionaries
        length_ranges: List of (min_len, max_len, range_name) tuples
        figsize: Figure dimensions (default: (12, 6))
        compare_ranges: List of range pairs to statistically compare
        alpha: Significance threshold (default: 0.05)
        save_path: Path to save figure (without extension)
    """
    # Set up journal-quality figure
    plt.figure(figsize=figsize, dpi=300)
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['pdf.fonttype'] = 42
    
    # Calculate window importance statistics
    window_stats = []
    for min_len, max_len, range_name in length_ranges:
        group_seqs = [attn for attn in attention_data 
                     if min_len <= (attn['edge_index'].shape[1] + 1) < max_len]
        
        if not group_seqs:
            continue
            
        # Find most important positions
        pos_importance = defaultdict(list)
        for seq in group_seqs:
            importance = seq['decay_final'] * seq['alpha_final']
            ranked_pos = torch.argsort(importance, descending=True)[:3]  # Top 3 positions
            for pos in ranked_pos:
                pos_importance[pos.item()].append(importance[pos].item())
        
        # Calculate statistics
        for pos, imp_values in pos_importance.items():
            window_stats.append({
                'Range': range_name,
                'Position': pos,
                'Mean Importance': np.mean(imp_values),
                'Frequency': len(imp_values)/len(group_seqs)
            })
    
    df = pd.DataFrame(window_stats)
    
    # Create plot with enhanced parameters
    ax = sns.scatterplot(
        data=df,
        x='Position',
        y='Mean Importance',
        size='Frequency',
        hue='Range',
        sizes=(50, 300),  # Increased size range
        palette='viridis',
        alpha=0.85,
        edgecolor='white',
        linewidth=0.5
    )

    # Enhanced statistical comparisons
    if compare_ranges:
        sig_results = []
        for i, (range1, range2) in enumerate(compare_ranges):
            group1 = df[df['Range']==range1]['Mean Importance']
            group2 = df[df['Range']==range2]['Mean Importance']
            
            if len(group1) > 1 and len(group2) > 1:
                _, p = stats.ttest_ind(group1, group2)
                stars = '*' * sum(p < alpha/(2**i) for i in range(1, 3))
                sig_results.append(f"{range1} vs {range2}: p={p:.2e}{stars}")
        
        # Add annotation box
        if sig_results:
            ax.text(0.98, 0.98, 
                   "Statistical Comparisons:\n" + "\n".join(sig_results),
                   transform=ax.transAxes,
                   ha='right',
                   va='top',
                   fontsize=10,
                   bbox=dict(facecolor='white', alpha=0, 
                            edgecolor='#cccccc', pad=1, boxstyle='round'))
    
    # Journal-quality formatting
    plt.title("Critical Window Identification Across Length Ranges", 
             fontsize=14, pad=20, fontweight='semibold')
    plt.xlabel("Node Position in Sequence", fontsize=12, labelpad=10)
    plt.ylabel("Mean Importance Score", fontsize=12, labelpad=10)
    
    # Enhanced grid and spines
    ax.grid(True, alpha=0.2, linestyle=':')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Professional legend
    handles, labels = ax.get_legend_handles_labels()
    legend = plt.legend(
        handles[:len(length_ranges)+1],
        labels[:len(length_ranges)+1],
        title='Length Range',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=True,
        framealpha=1,
        edgecolor='#333333'
    )
    legend.get_title().set_fontweight('semibold')
    
    plt.tight_layout()
    
    # Save in multiple formats if path provided
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Vector formats
        plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f"{save_path}.eps", format='eps', bbox_inches='tight', dpi=300)
        
        # High-res raster
        plt.savefig(f"{save_path}.tiff", format='tiff', bbox_inches='tight', dpi=600)
        plt.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=300)
    
    plt.show()
    return df

from scipy import stats
def generate_pairwise_ttest_table(stats_df, length_ranges, alpha=0.05):
    """
    Generates a publication-ready pairwise comparison table from critical windows data.
    
    Args:
        stats_df: DataFrame returned by plot_critical_windows()
        length_ranges: List of (min_len, max_len, range_name) tuples
        alpha: Significance threshold
        
    Returns:
        A DataFrame with pairwise comparison results
    """
    # Extract unique range names
    range_names = [name for _, _, name in length_ranges]
    
    # Initialize results
    results = []
    
    # Perform all pairwise comparisons
    for i, range1 in enumerate(range_names):
        for range2 in range_names[i+1:]:
            group1 = stats_df[stats_df['Range'] == range1]['Mean Importance']
            group2 = stats_df[stats_df['Range'] == range2]['Mean Importance']
            
            # Only compare if we have enough samples
            if len(group1) > 1 and len(group2) > 1:
                t_stat, p_value = stats.ttest_ind(group1, group2)
                
                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt(((len(group1)-1)*group1.std()**2 + 
                                    (len(group2)-1)*group2.std()**2) / 
                                    (len(group1) + len(group2) - 2))
                cohens_d = (group1.mean() - group2.mean()) / pooled_std
                
                # Determine significance
                stars = '*' * sum(p_value < alpha/(2**i) for i in range(1, 3))
                
                results.append({
                    'Comparison': f"{range1} vs {range2}",
                    'Mean Diff': group1.mean() - group2.mean(),
                    't-statistic': t_stat,
                    'p-value': p_value,
                    'Cohen\'s d': cohens_d,
                    'Significance': stars,
                    'n1': len(group1),
                    'n2': len(group2)
                })
    
    # Convert to DataFrame and format
    results_df = pd.DataFrame(results)
    
    # Formatting for publication
    if not results_df.empty:
        results_df['p-value'] = results_df['p-value'].apply(lambda x: f"{x:.3e}")
        results_df['Mean Diff'] = results_df['Mean Diff'].apply(lambda x: f"{x:.3f}")
        results_df['Cohen\'s d'] = results_df['Cohen\'s d'].apply(lambda x: f"{x:.2f}")
        results_df['t-statistic'] = results_df['t-statistic'].apply(lambda x: f"{x:.2f}")
        
        # Reorder columns
        results_df = results_df[[
            'Comparison', 'Mean Diff', 't-statistic', 
            'p-value', 'Cohen\'s d', 'Significance',
            'n1', 'n2'
        ]]
    
    return results_df

def get_top_significant_ranges(pairwise_table, num_top=10):
    """
    Extracts the most significant length range comparisons from pairwise results.
    
    Args:
        pairwise_table: DataFrame from generate_pairwise_ttest_table()
        num_top: Number of top comparisons to return (default: 10)
        
    Returns:
        tuple: (formatted DataFrame, list of comparison tuples)
    """
    # Create a copy to avoid SettingWithCopyWarning
    sig_comparisons = pairwise_table.copy()
    
    # Filter significant comparisons (at least one star)
    sig_comparisons = sig_comparisons[sig_comparisons['Significance'].str.len() > 0]
    
    # If no significant results, return empty structures
    if sig_comparisons.empty:
        return pd.DataFrame(columns=pairwise_table.columns), []
    
    # Convert p-values to float for sorting (on the copy)
    sig_comparisons.loc[:, 'p-value'] = sig_comparisons['p-value'].astype(float)
    
    # Sort by significance
    sig_comparisons.loc[:, 'star_count'] = sig_comparisons['Significance'].str.len()
    sig_comparisons = sig_comparisons.sort_values(
        by=['star_count', 'p-value', 'Cohen\'s d'],
        ascending=[False, True, False]
    )
    
    # Take top N or all if fewer than N
    top_comparisons = sig_comparisons.head(min(num_top, len(sig_comparisons)))
    
    # Create the tuple format for plotting
    comparison_tuples = []
    for comp in top_comparisons['Comparison']:
        range1, range2 = comp.split(' vs ')
        comparison_tuples.append((range1, range2))
    
    # Format p-values for display (on the final output copy)
    formatted_comparisons = top_comparisons.copy()
    formatted_comparisons.loc[:, 'p-value'] = formatted_comparisons['p-value'].apply(lambda x: f"{x:.3e}")
    formatted_comparisons = formatted_comparisons.drop(columns='star_count')
    
    return formatted_comparisons, comparison_tuples

def calculate_window_metrics(attention_data, length_ranges):
    """Modified to preserve numeric order"""
    metrics = []
    
    # First create a mapping from range names to sortable keys
    range_order = {name: (min_len, max_len) 
                  for min_len, max_len, name in length_ranges}
    
    for attn in attention_data:
        seq_len = attn['edge_index'].shape[1] + 1
        
        for min_len, max_len, range_name in length_ranges:
            if min_len <= seq_len < max_len:
                importance = attn['alpha_final'] * attn['decay_final']
                peak_attn = importance.max().item()
                threshold = 0.5 * peak_attn
                attention_span = (importance > threshold).sum().item() / seq_len
                peak_pos = torch.argmax(importance).item() / seq_len
                
                metrics.append({
                    'Length Range': range_name,
                    'Sort Key': min_len,  # Add numeric key for sorting
                    'Peak Attention': peak_attn,
                    'Attention Span': attention_span,
                    'Peak Position': peak_pos,
                    'Sequence Length': seq_len
                })
                break
    
    df = pd.DataFrame(metrics)
    return df.sort_values('Sort Key')

def aggregate_metrics(df_metrics):
    """Generates journal-ready statistics table while preserving order"""
    # Create ordered categories based on original sorting
    ordered_ranges = df_metrics['Length Range'].unique()
    df_metrics['Length Range'] = pd.Categorical(
        df_metrics['Length Range'],
        categories=ordered_ranges,
        ordered=True
    )
    
    # Groupby will now maintain this order
    agg_stats = df_metrics.groupby('Length Range', observed=True).agg({
        'Peak Attention': ['mean', 'std', 'count'],
        'Attention Span': ['mean', 'std'],
        'Peak Position': ['mean', 'std']
    })
    
    return agg_stats.round(3).rename(columns={
        'mean': 'Mean',
        'std': 'SD',
        'count': 'N'
    })

def plot_window_metrics(df_metrics, figsize=(20, 7), save_path=None):
    """
    Publication-quality visualization of window attention metrics.
    Resolves layout warnings while maintaining beautiful styling.
    
    Args:
        df_metrics: DataFrame containing the metrics to plot
        figsize: Figure dimensions (default: (20, 7))
    """
    # Create figure with adjusted proportions
    fig = plt.figure(figsize=figsize, dpi=300, constrained_layout=True)
    gs = fig.add_gridspec(1, 3, wspace=0.25)
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]
    
    # Custom color palettes
    palettes = {
        'peak': sns.light_palette("#4e79a7", n_colors=len(df_metrics['Length Range'].unique())),
        'span': sns.light_palette("#59a14f", n_colors=len(df_metrics['Length Range'].unique())),
        'pos': sns.light_palette("#e15759", n_colors=len(df_metrics['Length Range'].unique()))
    }
    
    # --- Metric 1: Peak Attention ---
    sns.boxplot(data=df_metrics, x='Length Range', y='Peak Attention',
               hue='Length Range', palette=palettes['peak'], ax=axes[0], width=0.7,
               linewidth=1.5, fliersize=4, legend=False, dodge=False)
    axes[0].set_title("Peak Attention Intensity", pad=12, fontsize=13, fontweight='semibold')
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Attention Score", labelpad=10)
    
    # Add median value annotations
    medians = df_metrics.groupby('Length Range', observed=False)['Peak Attention'].median()
    for i, (_, m) in enumerate(medians.items()):
        axes[0].text(i, m+0.03, f'{m:.2f}', 
                    ha='center', va='bottom', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, pad=2, edgecolor='none'))
    
    # --- Metric 2: Attention Span ---
    sns.violinplot(data=df_metrics, x='Length Range', y='Attention Span',
                  hue='Length Range', palette=palettes['span'], ax=axes[1], cut=0,
                  inner='quartile', linewidth=1.5, saturation=0.8, legend=False, dodge=False)
    axes[1].set_title("Proportion of Sequence Attended", pad=12, fontsize=13, fontweight='semibold')
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Proportion Attended", labelpad=10)
    
    # Add subtle distribution markers
    for i in range(1, 10):
        axes[1].axhline(i*0.1, color='white', alpha=0.1, linewidth=0.5, zorder=0)
    
    # --- Metric 3: Peak Position ---
    sns.stripplot(data=df_metrics, x='Length Range', y='Peak Position',
                 hue='Length Range', palette=palettes['pos'], ax=axes[2], alpha=0.7, 
                 jitter=0.25, size=5, linewidth=0.5, edgecolor='white', legend=False, dodge=False)
    
    # Enhanced pointplot with modern errorbar specification
    sns.pointplot(data=df_metrics, x='Length Range', y='Peak Position',
                 errorbar=('ci', 95), color='black', ax=axes[2],
                 markersize=10, linestyle='none', capsize=0.2,
                 err_kws={'linewidth': 1.5})
    axes[2].set_title("Normalized Position of Peak Attention", pad=12, fontsize=13, fontweight='semibold')
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Normalized Position", labelpad=10)
    
    # --- Unified Formatting ---
    for ax in axes:
        # X-axis formatting
        ax.set_xlabel("Sequence Length Range", labelpad=10)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        
        # Grid and spines
        ax.grid(axis='y', alpha=0.2, linestyle=':', linewidth=0.8)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        
        # Light background for better readability
        ax.set_facecolor('#f9f9f9')
    
    # Main title with descriptive caption
    fig.suptitle("Dynamic Window Attention Metrics by Sequence Length Group\n"
                "Comparative Analysis of Attention Patterns Across Input Lengths", 
                y=1.1, fontsize=14, fontweight='bold')
    
    # Add informative footer
    plt.figtext(0.5, -0.03, 
               f"Analysis based on {len(df_metrics)} samples | "
               f"{len(df_metrics['Length Range'].unique())} length groups | "
               f"Median values annotated",
               ha='center', fontsize=10, color='#555555')
    
    # Use constrained_layout instead of tight_layout
    fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.1)

    # Save in multiple formats if path provided
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Vector formats
        plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f"{save_path}.eps", format='eps', bbox_inches='tight', dpi=300)
        
        # High-res raster
        plt.savefig(f"{save_path}.tiff", format='tiff', bbox_inches='tight', dpi=600)
        plt.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=300)
    
    plt.show()