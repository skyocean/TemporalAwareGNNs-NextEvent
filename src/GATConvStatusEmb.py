import torch
import pandas as pd

from torch_geometric.data import Data
import torch.optim as optim

import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset

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

from torch_geometric.nn import GATConv

from sklearn.preprocessing import MultiLabelBinarizer
from collections import defaultdict

from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader 


def prepare_data_core_2edges(event_encode, core_encode, scaled_time_diffs, edge_types_encoded):
    data_list_event = []

    for i in range(len(event_encode)):
        node_features = torch.tensor(event_encode[i], dtype=torch.float)
        node_core = torch.tensor(core_encode[i], dtype=torch.long)
        num_events = (node_core[:, 0] != -1).sum()

        # Edge index: chain-like sequence
        edge_index = torch.tensor([[j, j + 1] for j in range(num_events - 1)], dtype=torch.long).t().contiguous()

        # Edge attributes (kept separate!)
        time_diffs = scaled_time_diffs[i][:num_events - 1]
        edge_types = edge_types_encoded[i][:num_events - 1]

        edge_type_tensor = torch.tensor(edge_types, dtype=torch.long)
        edge_time_tensor = torch.tensor(time_diffs, dtype=torch.float).view(-1, 1)

        #event_ids = node_core[:num_events].view(-1)
        event_ids = node_core[:num_events]

        # Create event-level graph
        graph_data = Data(
            x=node_features[:num_events],
            edge_index=edge_index,
            event_ids=event_ids,
            num_nodes=num_events
        )
        graph_data.edge_type = edge_type_tensor
        graph_data.edge_time_diff = edge_time_tensor

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


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total_tokens = 0

    with torch.no_grad():
        for event_data, labels in loader:
            event_data = event_data.to(device)
            #embed_data = embed_data.to(device)
            labels = labels.to(device)

            output = model(event_data)

            output = output.view(-1, output.size(-1))
            labels = labels.view(-1)

            # Apply mask before loss
            mask = labels != -1
            #output = output[mask]
            labels = labels[mask]
        
            loss = criterion(output, labels)        
            total_loss += loss.item() * labels.size(0)
        
            # Accuracy
            pred = output.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total_tokens += labels.size(0)
        
    accuracy = correct / total_tokens
    loss = total_loss / total_tokens
    return loss, accuracy
    
def custom_collate_fn(batch):
    event_data_list, label_list = zip(*batch)
    batch_event = Batch.from_data_list(event_data_list)
    padded_labels = pad_sequence([lbl.squeeze(1) for lbl in label_list], batch_first=True, padding_value=-1)

    return batch_event, padded_labels

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

class DualGAT2EdgesModel(nn.Module):
    def __init__(self, 
                 num_event_features,
                 num_embedding_features,
                 embedding_dims,
                 gat_hidden_dim_event,
                 gat_hidden_dim_embed,
                 gat_hidden_dim_concat,
                 output_dim,
                 num_heads,
                 num_edge_types,
                 edge_type_dim):
        super(DualGAT2EdgesModel, self).__init__()

        # Core event embedding path
        self.embedding = nn.Embedding(num_embeddings=num_embedding_features, embedding_dim=embedding_dims)

        # Edge-type embedding
        self.edge_type_emb = nn.Embedding(num_embeddings=num_edge_types, embedding_dim=edge_type_dim)

        # Total edge_attr dimension = time_diff (1) + edge_type_dim
        edge_attr_dim = 1 + edge_type_dim

        # GAT layers for embedded event path
        self.gat_embed = GATConv(embedding_dims, gat_hidden_dim_embed, heads=num_heads,
                                 concat=True, edge_dim=edge_attr_dim)

        # GAT layers for raw event feature path
        self.gat_event = GATConv(num_event_features, gat_hidden_dim_event, heads=num_heads,
                                 concat=True, edge_dim=edge_attr_dim)

        # GAT for combined path
        concat_input_dim = (gat_hidden_dim_embed + gat_hidden_dim_event) * num_heads
        self.gat_concat = GATConv(concat_input_dim, gat_hidden_dim_concat, heads=num_heads,
                                  concat=True, edge_dim=edge_attr_dim)

        # Output
        final_dim = gat_hidden_dim_concat * num_heads
        self.fc = nn.Linear(final_dim, output_dim)

    def forward(self, data_event):
        # ---- Build edge_attr ----
        edge_type = data_event.edge_type             # [E]
        edge_time = data_event.edge_time_diff        # [E, 1]

        type_vec = self.edge_type_emb(edge_type)     # [E, edge_type_dim]
        edge_attr = torch.cat([edge_time, type_vec], dim=-1)  # [E, 1 + edge_type_dim]

        # ---- Embedding path ----
        x_embed = self.embedding(data_event.event_ids.view(-1))   # [N, embedding_dim]
        x_embed = self.gat_embed(x_embed, data_event.edge_index, edge_attr=edge_attr)

        # ---- Event feature path ----
        x_event = self.gat_event(data_event.x, data_event.edge_index, edge_attr=edge_attr)

        # ---- Concatenation ----
        x = torch.cat([x_embed, x_event], dim=1)
        x = self.gat_concat(x, data_event.edge_index, edge_attr=edge_attr)

        # ---- Output ----
        out = self.fc(x)
        return out

def predict(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for event_data, labels in loader:
            event_data = event_data.to(device)
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
            
            # For top-k analysis, we need to get model's top-k at this position
            # Since predict_per_sequence only returns argmax, we need to modify it
            # Here's a workaround:
            # We'll consider it correct if either:
            # 1. The argmax is correct (top-1)
            # 2. Or if we're checking top-k and the argmax is wrong, 
            #    but the true label is in top-k (we'll approximate this)
            
            # For proper top-k analysis, you'd need to modify predict_per_sequence
            # to return top-k predictions. For now, this gives you position analysis:
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
            #embed_data = embed_data.to(device)
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