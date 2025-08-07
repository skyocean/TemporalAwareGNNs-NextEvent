from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder, StandardScaler
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from datetime import datetime, timedelta

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

import numpy as np
import pandas as pd

def custom_onehot_encode(data, categorical_columns, missing_value):
    """
    Custom one-hot encoding to handle '<no_desc>' as a missing category, encoding it differently.

    Args:
    data (DataFrame): The DataFrame containing the data.
    categorical_columns (list): The names of the categorical columns to encode.
    missing_value (str): The placeholder in the data that indicates missing values.

    Returns:
    Tuple[numpy.ndarray, OneHotEncoder]: The custom one-hot encoded matrix and the fitted encoder.
    """
    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Fit and transform the data
    data_encoded = encoder.fit_transform(data[categorical_columns])

    # Create a mask where data is missing_value
    mask = data[categorical_columns] == missing_value

    # Convert mask to match the shape of data_encoded
    expanded_mask = np.column_stack([mask[col] for col in categorical_columns for _ in range(len(encoder.categories_[categorical_columns.index(col)]))])

    # Apply the mask, setting encoded rows to -1 where the original data was missing
    data_encoded[expanded_mask] = -1

    return data_encoded, encoder

def onehot_encode(data, categorical_columns):
    """
    Perform one-hot encoding on specified categorical columns of a DataFrame.

    Args:
    data (DataFrame): The DataFrame containing categorical data.
    categorical_columns (list): List of column names to be one-hot encoded.

    Returns:
    numpy.ndarray: An array containing the one-hot encoded data.
    """
    encoder = OneHotEncoder(sparse_output=False)
    data_encoded = encoder.fit_transform(data[categorical_columns])
    return data_encoded, encoder

def custom_scale_encode(data, numerical_columns):
    """
    Scales numerical columns of a DataFrame where -1 indicates missing values.

    Args:
    data (DataFrame): DataFrame containing the data data.
    numerical_columns (list): List of column names to be scaled.

    Returns:
    DataFrame: A DataFrame with scaled numerical data, where -1 values are kept intact.
    """

    data = data.copy()
    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(index=data.index, columns=numerical_columns)
    
    for col in numerical_columns:
        # Replace -1 with NaN for scaling purposes
        valid_data = data[col].replace(-1, np.nan).dropna()
        if not valid_data.empty:
            # Fit scaler only on valid, non-missing data
            scaler.fit(valid_data.values.reshape(-1, 1))
            # Apply scaling to non-missing values
            data_scaled.loc[data[col] != -1, col] = scaler.transform(data[col][data[col] != -1].values.reshape(-1, 1)).flatten()
    
    # Replace NaNs with -1 in the scaled data
    data_scaled.fillna(-1, inplace=True)

    return data_scaled.values, scaler

def median_scale_encode(data, numerical_columns):
    """
    Scale numerical columns of a DataFrame, handling missing values with median imputation.

    Args:
    data (DataFrame): The DataFrame containing numerical data.
    numerical_columns (list): List of column names to be scaled.

    Returns:
    numpy.ndarray: An array containing the scaled numerical data.
    """
    data = data.copy()
    # Handle -1 as NaN for numerical features and replace with the median
    data[numerical_columns] = data[numerical_columns].replace(-1, np.nan)
    data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].median())
    
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[numerical_columns])
    return data_scaled, scaler


def encode_pad_event(event, cat_col_event, num_col_event, case_index, cat_mask=False, num_mask=False, eos = True):
    """
    encode sequence level features
    
    Parameters:
    - event Dateframe including event level features.
    - cat_col_event: List of column names of categorical features 
    - num_col_event: List of column names of numrical features; -1 respenting NAN
    - case_index: column name of sequence index
    
    Returns:
    - Encoding event level features and padding with the same length ready for input to the model.
    """
    

   # Custom encoding for categorical columns
    if cat_col_event:
        if cat_mask:        
            event_encoded, encoder = custom_onehot_encode(event, cat_col_event, "<NO_DESC>")
        else:        
            event_encoded, encoder = onehot_encode(event, cat_col_event)            
        combined_features_bulk = event_encoded
        
    # Apply median scaling
    if num_col_event:
        if num_mask:
            event_scaled, scaler = custom_scale_encode(event, num_col_event)
        else:            
            event_scaled, scaler = median_scale_encode(event, num_col_event)  
            
        if combined_features_bulk.size == 0:
            combined_features_bulk = event_scaled
        else:
            # Combine encoded categorical data and normalized numerical data
            combined_features_bulk = np.hstack((combined_features_bulk, event_scaled))
    
    # Prepare sequences
    encoded_sequences = []

    feature_length = combined_features_bulk.shape[1]  # Maximum features in a single stack
    if eos:
        eos_token = np.zeros((1, feature_length))
        
    for _, group in event.groupby(case_index):
        group_indices = group.index
        group_combined_features = combined_features_bulk[group_indices]

        if eos:
            # Append EOS token
            group_combined_features_with_eos = np.vstack([group_combined_features, eos_token])
            encoded_sequences.append(group_combined_features_with_eos)
        else:
            encoded_sequences.append(group_combined_features)
    
    # Pad sequences
    padded_sequences = pad_sequences(encoded_sequences, padding='post', dtype='float32', value = -1)
    
    return padded_sequences

def encode_pad_sequence(sequence, cat_col_seq, num_col_seq, cat_mask = False, num_mask = False):
    """
    Encode sequence level features.
    
    Parameters:
    - sequence: DataFrame including sequence level features.
    - cat_col_seq: List of column names of categorical features.
    - num_col_seq: List of column names of numerical features; -1 representing NAN.
    
    Returns:
    numpy.ndarray: Array of combined features ready for model input.
    """
    # Initialize combined_features_bulk
    combined_features_bulk = np.array([])

    if cat_col_seq:
        if cat_mask:
             sequence_encoded, encoder = custom_onehot_encode(sequence, cat_col_seq)
        else:
            # Apply one-hot encoding
            sequence_encoded, encoder = onehot_encode(sequence, cat_col_seq)
        combined_features_bulk = sequence_encoded
    
    if num_col_seq:
        if num_mask:
            # Apply median scaling
            sequence_scaled, scaler = custom_scale_encode(sequence, num_col_seq)
        else:            
            sequence_scaled, scaler = median_scale_encode(sequence, num_col_seq)
        if combined_features_bulk.size == 0:
            combined_features_bulk = sequence_scaled
        else:# Combine encoded categorical data and scaled numerical data
            combined_features_bulk = np.hstack((combined_features_bulk, sequence_scaled))
    
    return combined_features_bulk


def scale_time_differences(event, sequence, start_time_col, case_index):
    """
    Scale time differences between events within sequences using Min-Max scaling.

    Parameters:
    - event: DataFrame containing event data with start times.
    - sequence: DataFrame containing sequence information.
    - start_time_col: Column name in 'event' DataFrame for start times.
    - case_index: Column name in 'sequence' DataFrame for case identifier.

    Returns:
    - scaled_time_diffs_list: List of scaled time differences for each sequence.
    """
    # Convert start_time_col to datetime if not already
    event[start_time_col] = pd.to_datetime(event[start_time_col])

    # List to store all time differences
    all_time_diffs = []

    # Calculate time differences and collect in all_time_diffs
    for i in range(len(sequence)):
        times = event[event[case_index] == sequence.iloc[i][case_index]][start_time_col].values
        time_diffs = np.diff(times) / np.timedelta64(1, 's')  # Compute time differences in seconds
        all_time_diffs.extend(time_diffs)  # Collect all time differences

    # Convert the list to a NumPy array and reshape to 2D array
    all_time_diffs_array = np.array(all_time_diffs).reshape(-1, 1)

    # Initialize MinMaxScaler
    scaler = MinMaxScaler()

    # Apply Min-Max scaling to the entire array
    scaled_all_time_diffs = scaler.fit_transform(all_time_diffs_array)

    # Reshape the scaled array back to the original shape if needed
    scaled_all_time_diffs_flattened = scaled_all_time_diffs.flatten()

    # Split the scaled time differences back to the sequences
    scaled_time_diffs_list = []
    index = 0

    for i in range(len(sequence)):
        times = event[event[case_index] == sequence.iloc[i][case_index]][start_time_col].values
        num_diffs = len(np.diff(times))
        scaled_time_diffs_list.append(scaled_all_time_diffs_flattened[index:index + num_diffs])
        index += num_diffs
    
    return scaled_time_diffs_list


def encode_event_prefix_label(event, core_event, cat_col_event, num_col_event, case_index, prefix_size, cat_mask=False, num_mask=False):
    """
    Encode event sequence features and generate subsequences.

    Parameters:
    - event (DataFrame): Event-level data.
    - core_event (str): Column representing the core event.
    - cat_col_event (list): List of categorical feature columns.
    - num_col_event (list): List of numerical feature columns.
    - case_index (str): Column representing sequence index.
    - prefix_size (int): Size of subsequence.
    - cat_mask (bool): Use custom encoding for categorical data.
    - num_mask (bool): Use custom scaling for numerical data.

    Returns:
    - np.array: Encoded core event subsequences with shape `(total_subsequences, prefix_size, 1)`.
    - np.array: Encoded feature subsequences with shape `(total_subsequences, prefix_size, attributes_number)`.
    - np.array: Encoded y with shape `(total_subsequences, )`.
    - int: size of input core event.
    - int: size of output event.
    """

    event_copy = event[core_event].copy().to_frame()  # Convert to DataFrame
    event_copy.loc[len(event_copy)] = "EOS"  # Append EOS as a new row

    # Label encode Y using the SAME categories as one-hot encoder
    label_encoder = LabelEncoder()
    event_labels = label_encoder.fit_transform(event_copy[core_event])

    # Remove EOS from event_copy to align indices
    event_copy = event_copy[:-1] 

    # Drop last row (EOS) from event for X
    event_encoded = event_labels[:-1].reshape(-1, 1)  # Reshape for (num_samples, 1)

    # EOS code 
    eos_encoding = event_labels[-1] 
    y_labels = event_labels[:-1]

    combined_features_bulk = np.array([])  # Initialize empty feature matrix

    # Encode categorical features
    if cat_col_event:
        if cat_mask:        
            event_cat_encoded, _ = custom_onehot_encode(event, cat_col_event, "<NO_DESC>")
        else:        
            event_cat_encoded, _ = onehot_encode(event, cat_col_event)            
        
        combined_features_bulk = event_cat_encoded
        
    # Encode numerical features
    if num_col_event:
        if num_mask:
            event_scaled, _ = custom_scale_encode(event, num_col_event)
        else:            
            event_scaled, _ = median_scale_encode(event, num_col_event)  
            
        if combined_features_bulk.size == 0:
            combined_features_bulk = event_scaled
        else:
            combined_features_bulk = np.hstack((combined_features_bulk, event_scaled))

    # Prepare X and y
    event_values = []  # Changed from np.array([]) to list
    encoded_subsequences = []
    y_values = []

    # Iterate over each sequence
    for _, group in event.groupby(case_index):
        group_indices = group.index
        group_features = combined_features_bulk[group_indices]
        input_events = event_encoded[group_indices]  # Label-encoded event core
        predict_events = y_labels[group_indices]    # Labels for next event

        # Generate subsequences of `prefix_size` using a sliding window
        sequence_length = len(group_features)
        for i in range(sequence_length - prefix_size + 1):
            subseq = group_features[i : i + prefix_size]
            encoded_subsequences.append(subseq)

            # Append event core as (prefix_size, 1) shape
            event_values.append(input_events[i : i + prefix_size])  

            if i + prefix_size < sequence_length:
                y_values.append(predict_events[i + prefix_size])  # Predict next event
            else:  # Append EOS encoding at the end of y
                y_values.append(eos_encoding)

    # Convert to NumPy arrays
    event_values = np.array(event_values, dtype = np.int64)  # Ensure proper shape
    encoded_subsequences = np.array(encoded_subsequences, dtype=np.float32)
    y_values = np.array(y_values, dtype=np.int64)

    input_event_size = len(label_encoder.classes_) - 1  # Without EOS
    output_size = len(label_encoder.classes_)  # With EOS

    return event_values, encoded_subsequences, y_values, input_event_size, output_size


def encode_event_prefix(event, core_event, cat_col_event, num_col_event, case_index, prefix_size, cat_mask=False, num_mask=False):
    """
    Encode event sequence features and generate subsequences.

    Parameters:
    - event (DataFrame): Event-level data.
    - core_event (str): Column representing the core event.
    - cat_col_event (list): List of categorical feature columns.
    - num_col_event (list): List of numerical feature columns.
    - case_index (str): Column representing sequence index.
    - prefix_size (int): Size of subsequence.
    - cat_mask (bool): Use custom encoding for categorical data.
    - num_mask (bool): Use custom scaling for numerical data.

    Returns:
    - np.array: Encoded event and event feature subsequences with shape `(total_subsequences, prefix_size, attributes_number)`.
    - np.array: Encoded y with shape `(total_subsequences, )`.
    - int: size of output event.
    """

    event_copy = event[core_event].copy().to_frame()  # Convert to DataFrame
    event_copy.loc[len(event_copy)] = "EOS"  # Append EOS as a new row

    # One-hot encode core_event (includes EOS as a category)
    event_encoded, encoder = onehot_encode(event_copy, [core_event])  
    categories = encoder.categories_[0]  # Get ordered categories (includes EOS)

    # 2. Label encode Y using the SAME categories as one-hot encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(categories)  # Explicitly use one-hot encoder's categories
    y_labels = label_encoder.transform(event_copy[core_event])

    # Remove the EOS row from X (but keep it in Y)
    event_encoded = event_encoded[:-1]
    event_copy = event_copy[:-1]  # Also remove EOS from event_copy to align indices

    # EOS code 
    eos_encoding = y_labels[-1]     
    y_labels = y_labels[:-1]

    combined_features_bulk = np.array([])  # Initialize empty feature matrix

    # Encode categorical features
    if cat_col_event:
        if cat_mask:        
            event_cat_encoded, _ = custom_onehot_encode(event, cat_col_event, "<NO_DESC>")
        else:        
            event_cat_encoded, _ = onehot_encode(event, cat_col_event)            
        
        combined_features_bulk = event_cat_encoded
        
    # Encode numerical features
    if num_col_event:
        if num_mask:
            event_scaled, _ = custom_scale_encode(event, num_col_event)
        else:            
            event_scaled, _ = median_scale_encode(event, num_col_event)  
            
        if combined_features_bulk.size == 0:
            combined_features_bulk = event_scaled
        else:
            combined_features_bulk = np.hstack((combined_features_bulk, event_scaled))
   
    # add the core_event back to the bulk
    combined_features_bulk = np.hstack((event_encoded, combined_features_bulk))

    # Prepare X and y
    encoded_subsequences = []
    y_values = []

    # Iterate over each sequence
    for _, group in event.groupby(case_index):
        group_indices = group.index
        group_features = combined_features_bulk[group_indices]
        predict_events = y_labels[group_indices]    # Labels for next event

        # Generate subsequences of `prefix_size` using a sliding window
        sequence_length = len(group_features)
        for i in range(sequence_length - prefix_size + 1):
            subseq = group_features[i : i + prefix_size]
            encoded_subsequences.append(subseq)

            if i + prefix_size < sequence_length:
                y_values.append(predict_events[i + prefix_size])  # Predict next event
            else:  # Append EOS encoding at the end of y
                y_values.append(eos_encoding)

    encoded_subsequences = np.array(encoded_subsequences, dtype=np.float32)
    y_values = np.array(y_values, dtype=np.int64)

    output_size = len(label_encoder.classes_)  # With EOS

    return encoded_subsequences, y_values, output_size

def encode_label_event(event, core_event, case_index):
    """
    Generate prefix subsequences and next-event labels for training.

    Parameters:
    - event (DataFrame): Event-level data.
    - core_event: Column name for core event.
    - case_index (str): Column representing sequence index.
    - prefix_size (int): Size of prefix for input.

    Returns:
    - np.array:  Encoded input core event (num_sequences, core_event_size, 1)
    - np.array: Encoded out cputore event (num_sequences, core_event_size, 1)
    - int: Input size (without EOS)
    - int: Output size (with EOS)
    """
    le_event = LabelEncoder()
    all_events = event[core_event].tolist() + ['EOS']  # Include EOS
    le_event.fit(all_events)

    event['event_encoded_col'] = le_event.transform(event[core_event])
    EOS = le_event.transform(['EOS'])[0]

    event_values = []
    y_values = []

    for _, group in event.groupby(case_index):
        input_events = group['event_encoded_col'].tolist()

        # Input: all except last
        event_values.append(input_events)

        # Output: shifted left + EOS
        y_seq = input_events[1:] + [EOS]
        y_values.append(y_seq)

    # Pad to the max sequence length
    padded_event_values = pad_sequences(event_values, padding='post', dtype='int64', value=-1)
    padded_y_values = pad_sequences(y_values, padding='post', dtype='int64', value=-1)

    input_size = len(le_event.classes_)  # Without EOS but need one for embedding
    output_size = len(le_event.classes_)  # With EOS

    return padded_event_values[..., np.newaxis], padded_y_values[..., np.newaxis], input_size, output_size, le_event

def node_time_list(event, start_time_col, case_index):
    # Convert to datetime if not already
    event[start_time_col] = pd.to_datetime(event[start_time_col])
    event['unix_time'] = event[start_time_col].astype('int64') // 1_000_000_000  # seconds

    all_time_list = []

    for _, group in event.groupby(case_index):
        delta = group['unix_time'].values[-1] - group['unix_time'].values[:-1]
        # Normalize per sequence
        if len(delta) > 0:
            norm_delta = delta / (delta.max() + 1e-8)  # avoid division by zero
        else:
            norm_delta = delta  # edge case: empty delta
        all_time_list.append(norm_delta)

    return all_time_list

def event_transition_edge(event, sequence, status, case_index):
    """
    Label encode transition types between events (with substatus) within sequences.

    Parameters:
    - event: DataFrame with columns including event name, start time, and status.
    - sequence: DataFrame with one column [case_index] to identify each sequence.
    - status: Column name in 'event' DataFrame for the combination of the status of event (e.g., "complete", "start") and the event
    - case_index: Column name in both 'event' and 'sequence' identifying each case.

    Returns:
    - event_transition_list: Nested list. Outer list per sequence, inner list of label-encoded transition types.
    """

    # 3. Group by case
    grouped = event.groupby(case_index)

    # 4. Extract transitions
    all_transitions = []  # Nested list of transitions (string form) per case
    all_transition_strings = []  # Flattened list for global label encoding

    for cid in sequence[case_index]:
        if cid not in grouped.groups:
            all_transitions.append([])
            continue

        group = grouped.get_group(cid)
        ev_list = group[status].tolist()

        # Build pairwise transitions (i.e., edge from i to i+1)
        transitions = []
        for i in range(len(ev_list) - 1):
            edge = ev_list[i] + "→" + ev_list[i + 1]
            transitions.append(edge)
            all_transition_strings.append(edge)

        all_transitions.append(transitions)

    # 5. Label encode all transition strings
    le = LabelEncoder()
    le.fit(all_transition_strings)
    
    # 6. Encode each transition list using the fitted encoder
    event_transition_list = []
    for transitions in all_transitions:
        if transitions:
            encoded = le.transform(transitions)
            event_transition_list.append(np.array(encoded, dtype=np.int64))
        else:
            event_transition_list.append(np.array([], dtype=np.int64))

    trans_size = len(le.classes_)  + 1 #for embedding

    return event_transition_list, le, trans_size

def scale_time_differences_fast(event, start_time_col, case_index):
    """
    Faster version: scales time differences using groupby and avoids row-wise filtering.
    """

    event[start_time_col] = pd.to_datetime(event[start_time_col])
    
    # Group events by case
    grouped = event.sort_values(by=[case_index, start_time_col]).groupby(case_index)
    
    # Compute time differences per case and store in flat list and lengths
    time_diffs_list = []
    lengths = []

    for _, group in grouped:
        times = group[start_time_col].values
        diffs = np.diff(times) / np.timedelta64(1, 's')
        time_diffs_list.append(diffs)
        lengths.append(len(diffs))

    # Flatten all diffs
    all_diffs = np.concatenate(time_diffs_list).reshape(-1, 1)

    # Scale
    scaler = MinMaxScaler()
    scaled_all_diffs = scaler.fit_transform(all_diffs).flatten()

    # Split back
    scaled_time_diffs_list = []
    index = 0
    for l in lengths:
        scaled_time_diffs_list.append(scaled_all_diffs[index:index + l])
        index += l

    return scaled_time_diffs_list

def scale_time_differences_fast_fixed(event, sequence, start_time_col, case_index):
    """
    Fixed faster version that maintains the same order as the slower version.
    """
    event[start_time_col] = pd.to_datetime(event[start_time_col])
    
    # Sort events by case and time for consistent ordering
    event_sorted = event.sort_values(by=[case_index, start_time_col])
    
    # Group events by case and compute time differences
    grouped = event_sorted.groupby(case_index)
    
    # Create a dictionary to store time differences by case_id
    case_time_diffs = {}
    for case_id, group in grouped:
        times = group[start_time_col].values
        diffs = np.diff(times) / np.timedelta64(1, 's')
        case_time_diffs[case_id] = diffs
    
    # Collect time differences in the same order as the sequence DataFrame
    time_diffs_list = []
    for i in range(len(sequence)):
        case_id = sequence.iloc[i][case_index]
        if case_id in case_time_diffs:
            time_diffs_list.append(case_time_diffs[case_id])
        else:
            # Handle case where no time differences exist (single event case)
            time_diffs_list.append(np.array([]))
    
    # Flatten all diffs for scaling (only non-empty arrays)
    non_empty_diffs = [diffs for diffs in time_diffs_list if len(diffs) > 0]
    
    if len(non_empty_diffs) == 0:
        # No time differences to scale
        return [np.array([]) for _ in range(len(sequence))]
    
    all_diffs = np.concatenate(non_empty_diffs).reshape(-1, 1)
    
    # Scale
    scaler = MinMaxScaler()
    scaled_all_diffs = scaler.fit_transform(all_diffs).flatten()
    
    # Split back maintaining original order
    scaled_time_diffs_list = []
    index = 0
    for diffs in time_diffs_list:
        if len(diffs) > 0:
            scaled_time_diffs_list.append(scaled_all_diffs[index:index + len(diffs)])
            index += len(diffs)
        else:
            scaled_time_diffs_list.append(np.array([]))
    
    return scaled_time_diffs_list

def length_stratified_split(event_feature_list, test_size=0.2, n_bins=5):
    sequence_lengths = [data.x.shape[0] for data in event_feature_list]
    
    # Create length bins
    min_len, max_len = min(sequence_lengths), max(sequence_lengths)
    bin_edges = np.linspace(min_len, max_len + 1, n_bins + 1)
    
    # Assign each sequence to a bin
    bins = np.digitize(sequence_lengths, bin_edges) - 1
    bins = np.clip(bins, 0, n_bins - 1)  # Ensure bins are in valid range
    
    train_indices = []
    test_indices = []
    
    # Split each bin proportionally
    for bin_id in range(n_bins):
        bin_indices = [i for i, b in enumerate(bins) if b == bin_id]
        if len(bin_indices) == 0:
            continue
            
        # Calculate how many from this bin go to test
        n_test = max(1, int(len(bin_indices) * test_size))
        n_train = len(bin_indices) - n_test
        
        # Sort by length within bin for consistent splitting
        bin_indices_with_lengths = [(i, sequence_lengths[i]) for i in bin_indices]
        bin_indices_with_lengths.sort(key=lambda x: x[1])
        
        # Split: shorter sequences to train, longer to test within each bin
        train_indices.extend([idx for idx, _ in bin_indices_with_lengths[:n_train]])
        test_indices.extend([idx for idx, _ in bin_indices_with_lengths[n_train:]])
    
    return train_indices, test_indices
