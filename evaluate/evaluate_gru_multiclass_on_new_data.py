"""
Evaluate GRU Multiclass Model on New MQTT Server Data
======================================================
This script evaluates a GRU multiclass model (with best hyperparameters) on a new 
preprocessed dataset from another MQTT server.

Usage:
------
Option 1: Use a pre-trained model (RECOMMENDED - no training needed)
  1. Edit CONFIG['pretrained_model_path'] to point to your trained model (.pth file)
  2. Edit CONFIG['test_data_file'] to point to your new preprocessed CSV file
  3. Run: python evaluate_gru_multiclass_on_new_data.py

Option 2: Train a new model first (if no pre-trained model available)
  1. Edit CONFIG['train_data_file'] to point to training data
  2. Edit CONFIG['test_data_file'] to point to your new preprocessed CSV file
  3. Run: python evaluate_gru_multiclass_on_new_data.py

Option 3: Command line arguments
  python evaluate_gru_multiclass_on_new_data.py <test_data_file> [pretrained_model_path]
  
  Example:
  python evaluate_gru_multiclass_on_new_data.py EMQX_preprocessed.csv best_gru_multiclass_model.pth

Requirements:
-------------
- The new preprocessed data should have the same format as the training data
- Must include 'Tag' column (multiclass labels: -1=Normal, 1-30, 32=violation types)
- Must include 'tcp.stream' column for sequence grouping
- Should have 'seq_position' or 'frame.number' for ordering
- If using pre-trained model, it must match the hyperparameters in this script

Output:
-------
- Results: evaluation_results_multiclass_new_server.json
- Confusion matrix: confusion_matrix_multiclass_new_server.png
- Per-class metrics: per_class_metrics_multiclass_new_server.png (optional)

Author: Maryam
Date: December 2025
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_curve, auc
)
from datetime import datetime
import json
import os
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

from models import GRUClassifier

# =============================================================================
# CONFIGURATION - Best Hyperparameters for GRU Multiclass Classification
# =============================================================================

# Best hyperparameters from nested CV results
BEST_HYPERPARAMETERS = {
    'max_seq_length': 100,
    'gru_hidden_size': 128,
    'gru_num_layers': 2,
    'gru_bidirectional': True,
    'learning_rate': 0.001,
    'dropout': 0.3,
    'batch_size': 64,
    'num_epochs': 50,
    'early_stopping_patience': 15,
    'random_seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# File paths
# NOTE: Edit paths below to point to your files
CONFIG = {
    'pretrained_model_path': 'best_gru_multiclass_model.pth',
    'train_data_file': 'mqtt_preprocessed_v2.csv',
    'test_data_file': '../traces/emqx/EMQX_preprocessed.csv',
    'model_save_path': 'best_gru_multiclass_model.pth',
    'results_save_path': 'Result/evaluation_results_multiclass.json',
    'confusion_matrix_save_path': 'Result/confusion_matrix_multiclass.png',
    'per_class_metrics_save_path': 'Result/per_class_metrics_multiclass.png'
}

# Feature columns (same as in train_gru_multiclass_model.py)
FEATURE_COLUMNS = [
    'time.delta.prevPkt', 'kalive.Violated', 'Direction',
    'mqtt.len', 'mqtt.msgtype', 'mqtt.hdr_reserved',
    'mqtt.qos', 'mqtt.dupflag', 'msgid_value', 'msgid_is_present',
    'mqtt.topic_len', 'topic_depth', 'mqtt.topic.null', 'mqtt.topic.wildcard',
    'mqtt.topic.invalidUTF', 'mqtt.topic.empty',
    'mqtt.willtopic_len', 'mqtt.willtopic.null', 'mqtt.willtopic.wildcard',
    'mqtt.willtopic.empty', 'mqtt.willmsg_len', 'mqtt.msg.empty', 'mqtt.conflag.willflag',
    'protoname_is_valid', 'mqtt.kalive', 'mqtt.conflag.reserved',
    'mqtt.connack.reason_code', 'mqtt.disconnect.reason_code',
    'mqtt.subscription_options_numeric', 'mqtt.subscription_options_reserved_numeric',
    'mqtt.suback.reason_code_numeric', 'mqtt.unsuback.reason_code_numeric',
    'suback_out_of_order', 'unsuback_out_of_order', 'subscribe_msgid_mismatch',
    'unsubscribe_msgid_mismatch', 'subscribe_ack_missing', 'unsubscribe_ack_missing',
    'pending_subscribe_count', 'pending_unsubscribe_count', 'msgid_order_deviation',
    'pingreq_missing', 'time_since_last_client_control', 'ping.missed'
]

SEQUENCE_CONSTRUCTION_FEATURES = ['seq_position', 'seq_length', 'is_first_packet', 'is_same_frame']

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =============================================================================
# DATASET CLASS
# =============================================================================

class MQTTSequenceDataset(Dataset):
    """PyTorch Dataset for MQTT sequences (multiclass version)."""
    
    def __init__(self, sequences, labels, max_seq_length):
        self.sequences = sequences
        self.labels = labels
        self.max_seq_length = max_seq_length
        self.num_features = sequences[0].shape[1] if len(sequences) > 0 else 0
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        if len(seq) > self.max_seq_length:
            seq = seq[-self.max_seq_length:]
        elif len(seq) < self.max_seq_length:
            padding = np.zeros((self.max_seq_length - len(seq), self.num_features))
            seq = np.vstack([padding, seq])
        
        return torch.FloatTensor(seq), torch.LongTensor([label])

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_data(filepath, feature_columns):
    """
    Load preprocessed data and prepare sequences for multiclass classification.
    
    Parameters:
    -----------
    filepath : str
        Path to preprocessed CSV file
    feature_columns : list
        List of feature column names to use
        
    Returns:
    --------
    tuple: (sequences, labels, available_features, tag_to_class, class_names, num_classes)
    """
    print(f"\nLoading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df):,} packets from {df['tcp.stream'].nunique():,} streams")
    
    # Get available features (exclude sequence construction features)
    available_features = [
        col for col in feature_columns 
        if col in df.columns and col not in SEQUENCE_CONSTRUCTION_FEATURES
    ]
    print(f"  Using {len(available_features)} features")
    
    # Check for missing features
    missing_features = [col for col in feature_columns 
                       if col not in df.columns and col not in SEQUENCE_CONSTRUCTION_FEATURES]
    if missing_features:
        print(f"  WARNING: {len(missing_features)} features missing from dataset")
        if len(missing_features) <= 10:
            print(f"    Missing: {missing_features}")
        else:
            print(f"    Missing: {missing_features[:10]}... (and {len(missing_features)-10} more)")
        print(f"  Model will use {len(available_features)} available features")
    
    # Prepare sequences
    sequences = []
    tags = []
    
    for stream_id, group in df.groupby('tcp.stream'):
        group = group.sort_values('seq_position' if 'seq_position' in group.columns else 'frame.number')
        seq_features = group[available_features].values.astype(np.float32)
        
        # Get Tag (violation type) - use -1 for normal
        if 'Tag' in group.columns:
            seq_tag = int(group['Tag'].iloc[0]) if not pd.isna(group['Tag'].iloc[0]) else -1
        else:
            print("  WARNING: 'Tag' column not found. Using default tag -1 (Normal).")
            seq_tag = -1
        
        sequences.append(seq_features)
        tags.append(seq_tag)
    
    print(f"  Created {len(sequences):,} sequences")
    
    # Encode tags: -1 becomes class 0 (normal), others become 1, 2, 3, ...
    unique_tags = sorted(set(tags))
    tag_to_class = {tag: idx for idx, tag in enumerate(unique_tags)}
    class_names = {idx: f"Tag_{tag}" if tag != -1 else "Normal" for idx, tag in enumerate(unique_tags)}
    
    labels = [tag_to_class[tag] for tag in tags]
    num_classes = len(unique_tags)
    
    # Print class distribution
    print(f"\n  Class distribution:")
    class_counts = pd.Series(labels).value_counts().sort_index()
    for class_idx, count in class_counts.items():
        print(f"    Class {class_idx} ({class_names[class_idx]}): {count:,} sequences ({count/len(labels)*100:.2f}%)")
    
    return sequences, labels, available_features, tag_to_class, class_names, num_classes

def align_features(train_features, test_features):
    """
    Align feature lists to ensure both datasets use the same features.
    
    Parameters:
    -----------
    train_features : list
        Feature columns from training data
    test_features : list
        Feature columns from test data
        
    Returns:
    --------
    tuple: (aligned_train_features, aligned_test_features, common_features)
    """
    # Find common features
    common_features = [f for f in train_features if f in test_features]
    
    # Find missing in test
    missing_in_test = [f for f in train_features if f not in test_features]
    
    # Find extra in test
    extra_in_test = [f for f in test_features if f not in train_features]
    
    if missing_in_test:
        print(f"\n  WARNING: {len(missing_in_test)} features present in training but missing in test:")
        if len(missing_in_test) <= 10:
            print(f"    {missing_in_test}")
        else:
            print(f"    {missing_in_test[:10]}... (and {len(missing_in_test)-10} more)")
        print(f"    These features will be set to 0 in test data")
    
    if extra_in_test:
        print(f"\n  INFO: {len(extra_in_test)} features present in test but not in training:")
        if len(extra_in_test) <= 10:
            print(f"    {extra_in_test}")
        else:
            print(f"    {extra_in_test[:10]}... (and {len(extra_in_test)-10} more)")
        print(f"    These features will be ignored")
    
    print(f"\n  Using {len(common_features)} common features for evaluation")
    
    return common_features

def normalize_sequences(sequences_train, sequences_test, feature_columns_train, feature_columns_test):
    """
    Normalize sequences using scaler fitted on training data.
    
    Parameters:
    -----------
    sequences_train : list of np.ndarray
        Training sequences
    sequences_test : list of np.ndarray
        Test sequences
    feature_columns_train : list
        Feature columns from training data
    feature_columns_test : list
        Feature columns from test data
        
    Returns:
    --------
    tuple: (normalized_train_sequences, normalized_test_sequences, scaler, aligned_features)
    """
    # Align features
    aligned_features = align_features(feature_columns_train, feature_columns_test)
    
    # Get indices for aligned features in both datasets
    train_indices = [feature_columns_train.index(f) for f in aligned_features]
    test_indices = [feature_columns_test.index(f) if f in feature_columns_test else None for f in aligned_features]
    
    # Extract aligned features from sequences
    train_features_aligned = []
    for seq in sequences_train:
        aligned_seq = seq[:, train_indices]
        train_features_aligned.append(aligned_seq)
    
    test_features_aligned = []
    for seq in sequences_test:
        aligned_seq = np.zeros((len(seq), len(aligned_features)), dtype=np.float32)
        for i, (train_idx, test_idx) in enumerate(zip(train_indices, test_indices)):
            if test_idx is not None:
                aligned_seq[:, i] = seq[:, test_idx]
            # else: feature missing in test, keep as 0
        test_features_aligned.append(aligned_seq)
    
    # Fit scaler on training data
    print("\n  Fitting scaler on training data...")
    train_features_flat = np.vstack(train_features_aligned)
    scaler = StandardScaler()
    scaler.fit(train_features_flat)
    
    # Transform both datasets
    print("  Normalizing sequences...")
    sequences_train_norm = [scaler.transform(seq) for seq in train_features_aligned]
    sequences_test_norm = [scaler.transform(seq) for seq in test_features_aligned]
    
    print("  Normalization complete")
    
    return sequences_train_norm, sequences_test_norm, scaler, aligned_features

def align_class_mappings(tag_to_class_train, class_names_train, num_classes_train,
                         tag_to_class_test, class_names_test, num_classes_test):
    """
    Align class mappings between train and test datasets.
    Creates a unified mapping that includes all classes from both datasets.
    
    Parameters:
    -----------
    tag_to_class_train : dict
        Tag to class mapping from training data
    class_names_train : dict
        Class names from training data
    num_classes_train : int
        Number of classes in training data
    tag_to_class_test : dict
        Tag to class mapping from test data
    class_names_test : dict
        Class names from test data
    num_classes_test : int
        Number of classes in test data
        
    Returns:
    --------
    tuple: (unified_tag_to_class, unified_class_names, unified_num_classes, 
            train_class_to_unified, test_class_to_unified)
    """
    # Get all unique tags from both datasets
    all_tags = sorted(set(list(tag_to_class_train.keys()) + list(tag_to_class_test.keys())))
    
    # Create unified mapping
    unified_tag_to_class = {tag: idx for idx, tag in enumerate(all_tags)}
    unified_class_names = {idx: f"Tag_{tag}" if tag != -1 else "Normal" 
                          for idx, tag in enumerate(all_tags)}
    unified_num_classes = len(all_tags)
    
    # Create mapping from train/test class indices to unified class indices
    train_class_to_unified = {}
    for tag, train_class in tag_to_class_train.items():
        unified_class = unified_tag_to_class[tag]
        train_class_to_unified[train_class] = unified_class
    
    test_class_to_unified = {}
    for tag, test_class in tag_to_class_test.items():
        unified_class = unified_tag_to_class[tag]
        test_class_to_unified[test_class] = unified_class
    
    print(f"\n  Unified class mapping: {unified_num_classes} classes")
    print(f"    Training classes: {num_classes_train}")
    print(f"    Test classes: {num_classes_test}")
    print(f"    Classes in both: {len(set(tag_to_class_train.keys()) & set(tag_to_class_test.keys()))}")
    
    return (unified_tag_to_class, unified_class_names, unified_num_classes,
            train_class_to_unified, test_class_to_unified)

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for sequences, labels in train_loader:
        sequences = sequences.to(device)
        labels = labels.squeeze().to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

def evaluate(model, data_loader, criterion, device, num_classes, return_probs=False):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences = sequences.to(device)
            labels = labels.squeeze().to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if return_probs:
                all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision_micro = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    recall_micro = recall_score(all_labels, all_preds, average='micro', zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    precision_weighted = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall_weighted = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    if return_probs:
        return (avg_loss, accuracy, precision_macro, recall_macro, f1_macro,
                precision_micro, recall_micro, f1_micro,
                precision_weighted, recall_weighted, f1_weighted,
                all_preds, all_labels, all_probs)
    else:
        return (avg_loss, accuracy, precision_macro, recall_macro, f1_macro,
                precision_micro, recall_micro, f1_micro,
                precision_weighted, recall_weighted, f1_weighted,
                all_preds, all_labels)

def train_model(model, train_loader, val_loader, config, num_classes):
    """Train model with early stopping."""
    device = config['device']
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None
    
    print("\n" + "="*80)
    print("TRAINING MODEL")
    print("="*80)
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | "
          f"{'Val Acc':>7} | {'Val F1':>7}")
    print("-"*80)
    
    for epoch in range(config['num_epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _, val_f1, _, _, _, _, _, _, _, _ = evaluate(
            model, val_loader, criterion, device, num_classes
        )
        scheduler.step(val_loss)
        
        print(f"{epoch+1:>6} | {train_loss:>10.4f} | {train_acc:>9.4f} | {val_loss:>8.4f} | "
              f"{val_acc:>7.4f} | {val_f1:>7.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"       ^ New best! (F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f"\nEarly stopping at epoch {epoch+1} (best F1: {best_val_f1:.4f})")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print("-"*80)
    print(f"Training complete. Best validation F1: {best_val_f1:.4f}")
    
    return model

def train_model_full_data(model, train_loader, config, num_classes):
    """
    Train model on all data for a fixed number of epochs (no validation split).
    Use this when building the final model for evaluation on new server data.
    """
    device = config['device']
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    print("\n" + "="*80)
    print("TRAINING MODEL ON ALL DATA (no validation split)")
    print("="*80)
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9}")
    print("-"*80)
    
    for epoch in range(config['num_epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step(train_loss)
        print(f"{epoch+1:>6} | {train_loss:>10.4f} | {train_acc:>9.4f}")
    
    print("-"*80)
    print(f"Training complete. Final train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
    
    return model

# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

def main():
    """Main evaluation function."""
    print("="*80)
    print("GRU MULTICLASS MODEL EVALUATION ON NEW MQTT SERVER DATA")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {BEST_HYPERPARAMETERS['device']}")
    print("="*80)
    
    # Get test data file path and model path from command line if provided
    import sys
    if CONFIG['test_data_file'] is None:
        if len(sys.argv) > 1:
            CONFIG['test_data_file'] = sys.argv[1]
            print(f"\nUsing test data file from command line: {CONFIG['test_data_file']}")
        else:
            print("\n" + "="*80)
            print("ERROR: No test data file specified!")
            print("="*80)
            print("\nPlease specify the path to the new preprocessed data file.")
            print("You can either:")
            print("  1. Edit CONFIG['test_data_file'] in this script (around line 85)")
            print("  2. Pass it as a command line argument:")
            print("     python evaluate_gru_multiclass_on_new_data.py <path_to_preprocessed_data.csv>")
            print("\nExample:")
            print("  python evaluate_gru_multiclass_on_new_data.py EMQX_preprocessed.csv")
            print("  python evaluate_gru_multiclass_on_new_data.py ../EMQX_Multi_new_server_data.csv")
            return
    else:
        print(f"\nUsing test data file: {CONFIG['test_data_file']}")
    
    # Get pre-trained model path from command line if provided
    if len(sys.argv) > 2:
        CONFIG['pretrained_model_path'] = sys.argv[2]
        print(f"Using pre-trained model from command line: {CONFIG['pretrained_model_path']}")
    
    if not os.path.exists(CONFIG['test_data_file']):
        print(f"\nERROR: Test data file not found: {CONFIG['test_data_file']}")
        return
    
    # Check if we need training data (only if no pre-trained model)
    need_training = CONFIG['pretrained_model_path'] is None or not os.path.exists(CONFIG['pretrained_model_path'])
    if need_training:
        if not os.path.exists(CONFIG['train_data_file']):
            print(f"\nERROR: Training data file not found: {CONFIG['train_data_file']}")
            print("       (Required because no pre-trained model was provided)")
            return
        print("\nNo pre-trained model found. Will train a new model with best hyperparameters.")
    else:
        print(f"\nPre-trained model found. Will load: {CONFIG['pretrained_model_path']}")
        # Still need training data to fit the scaler for normalization
        if not os.path.exists(CONFIG['train_data_file']):
            print(f"\nWARNING: Training data file not found: {CONFIG['train_data_file']}")
            print("         (Needed to fit scaler for normalization)")
            print("         Proceeding anyway, but normalization may not match training data...")
    
    set_seed(BEST_HYPERPARAMETERS['random_seed'])
    
    # Load training data
    print("\n" + "="*80)
    print("STEP 1: LOADING TRAINING DATA")
    print("="*80)
    sequences_train, labels_train, features_train, tag_to_class_train, class_names_train, num_classes_train = load_data(
        CONFIG['train_data_file'], FEATURE_COLUMNS
    )
    
    # Load test data
    print("\n" + "="*80)
    print("STEP 2: LOADING TEST DATA (NEW SERVER)")
    print("="*80)
    sequences_test, labels_test, features_test, tag_to_class_test, class_names_test, num_classes_test = load_data(
        CONFIG['test_data_file'], FEATURE_COLUMNS
    )
    
    # Align class mappings
    print("\n" + "="*80)
    print("STEP 3: ALIGNING CLASS MAPPINGS")
    print("="*80)
    (unified_tag_to_class, unified_class_names, unified_num_classes,
     train_class_to_unified, test_class_to_unified) = align_class_mappings(
        tag_to_class_train, class_names_train, num_classes_train,
        tag_to_class_test, class_names_test, num_classes_test
    )
    
    # Map labels to unified class indices
    labels_train_unified = [train_class_to_unified[label] for label in labels_train]
    labels_test_unified = [test_class_to_unified[label] for label in labels_test]
    
    # Normalize and align features
    print("\n" + "="*80)
    print("STEP 4: NORMALIZING AND ALIGNING FEATURES")
    print("="*80)
    sequences_train_norm, sequences_test_norm, scaler, aligned_features = normalize_sequences(
        sequences_train, sequences_test, features_train, features_test
    )
    
    # Use ALL training data (no validation split) for final model for new-server evaluation
    print("\n" + "="*80)
    print("STEP 5: PREPARING DATA (train on all original data)")
    print("="*80)
    print(f"  Train (all data): {len(sequences_train_norm):,} sequences")
    
    # Create data loaders (no val split)
    train_dataset = MQTTSequenceDataset(sequences_train_norm, labels_train_unified, BEST_HYPERPARAMETERS['max_seq_length'])
    test_dataset = MQTTSequenceDataset(sequences_test_norm, labels_test_unified, BEST_HYPERPARAMETERS['max_seq_length'])
    
    train_loader = DataLoader(train_dataset, batch_size=BEST_HYPERPARAMETERS['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BEST_HYPERPARAMETERS['batch_size'], shuffle=False)
    
    # Create model
    print("\n" + "="*80)
    print("STEP 6: CREATING MODEL")
    print("="*80)
    input_size = len(aligned_features)
    print(f"  Input size: {input_size} features")
    print(f"  Number of classes: {unified_num_classes}")
    print(f"  Hyperparameters:")
    print(f"    Sequence length: {BEST_HYPERPARAMETERS['max_seq_length']}")
    print(f"    GRU hidden size: {BEST_HYPERPARAMETERS['gru_hidden_size']}")
    print(f"    GRU num layers: {BEST_HYPERPARAMETERS['gru_num_layers']}")
    print(f"    GRU bidirectional: {BEST_HYPERPARAMETERS['gru_bidirectional']}")
    print(f"    Learning rate: {BEST_HYPERPARAMETERS['learning_rate']}")
    print(f"    Dropout: {BEST_HYPERPARAMETERS['dropout']}")
    
    model = GRUClassifier(
        input_size=input_size,
        hidden_size=BEST_HYPERPARAMETERS['gru_hidden_size'],
        num_layers=BEST_HYPERPARAMETERS['gru_num_layers'],
        bidirectional=BEST_HYPERPARAMETERS['gru_bidirectional'],
        num_classes=unified_num_classes,
        dropout=BEST_HYPERPARAMETERS['dropout']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Load or train model
    if CONFIG['pretrained_model_path'] and os.path.exists(CONFIG['pretrained_model_path']):
        print("\n" + "="*80)
        print("STEP 7: LOADING PRE-TRAINED MODEL")
        print("="*80)
        try:
            model.load_state_dict(torch.load(CONFIG['pretrained_model_path'], map_location=BEST_HYPERPARAMETERS['device']))
            print(f"[OK] Successfully loaded model from {CONFIG['pretrained_model_path']}")
        except Exception as e:
            print(f"[ERROR] Failed to load model from {CONFIG['pretrained_model_path']}")
            print(f"  Error: {e}")
            print("  Will train a new model instead...")
            print("\n" + "="*80)
            print("STEP 7: TRAINING MODEL (on all data)")
            print("="*80)
            model = train_model_full_data(model, train_loader, BEST_HYPERPARAMETERS, unified_num_classes)
            # Save the newly trained model
            torch.save(model.state_dict(), CONFIG['model_save_path'])
            print(f"\n[OK] Model saved to {CONFIG['model_save_path']} for future use")
    else:
        # Train model on all data
        print("\n" + "="*80)
        print("STEP 7: TRAINING MODEL (on all data)")
        print("="*80)
        model = train_model_full_data(model, train_loader, BEST_HYPERPARAMETERS, unified_num_classes)
        
        # Save the trained model
        torch.save(model.state_dict(), CONFIG['model_save_path'])
        print(f"\n[OK] Model saved to {CONFIG['model_save_path']} for future use")
    
    # Evaluate on test data
    print("\n" + "="*80)
    print("STEP 8: EVALUATING ON NEW SERVER DATA")
    print("="*80)
    criterion = nn.CrossEntropyLoss()
    (test_loss, test_acc, test_prec_macro, test_rec_macro, test_f1_macro,
     test_prec_micro, test_rec_micro, test_f1_micro,
     test_prec_weighted, test_rec_weighted, test_f1_weighted,
     test_preds, test_labels, test_probs) = evaluate(
        model, test_loader, criterion, BEST_HYPERPARAMETERS['device'], unified_num_classes, return_probs=True
    )
    
    print(f"\nTest Results:")
    print(f"  Loss:           {test_loss:.4f}")
    print(f"  Accuracy:       {test_acc:.4f}")
    print(f"\n  Macro Average:")
    print(f"    Precision:    {test_prec_macro:.4f}")
    print(f"    Recall:       {test_rec_macro:.4f}")
    print(f"    F1 Score:     {test_f1_macro:.4f}")
    print(f"\n  Micro Average:")
    print(f"    Precision:    {test_prec_micro:.4f}")
    print(f"    Recall:       {test_rec_micro:.4f}")
    print(f"    F1 Score:     {test_f1_micro:.4f}")
    print(f"\n  Weighted Average:")
    print(f"    Precision:    {test_prec_weighted:.4f}")
    print(f"    Recall:       {test_rec_weighted:.4f}")
    print(f"    F1 Score:     {test_f1_weighted:.4f}")
    
    # Confusion Matrix (force full class space even if some classes are absent in test set)
    all_class_labels = list(range(unified_num_classes))
    cm = confusion_matrix(test_labels, test_preds, labels=all_class_labels)
    print(f"\nConfusion Matrix Shape: {cm.shape}")
    
    # Classification Report
    print(f"\nClassification Report:")
    target_names = [unified_class_names[i] for i in range(unified_num_classes)]
    print(classification_report(
        test_labels,
        test_preds,
        labels=all_class_labels,
        target_names=target_names,
        zero_division=0
    ))
    
    # Calculate per-class metrics
    per_class_precision = precision_score(
        test_labels, test_preds, labels=all_class_labels, average=None, zero_division=0
    )
    per_class_recall = recall_score(
        test_labels, test_preds, labels=all_class_labels, average=None, zero_division=0
    )
    per_class_f1 = f1_score(
        test_labels, test_preds, labels=all_class_labels, average=None, zero_division=0
    )
    
    # Plot confusion matrix
    plt.figure(figsize=(max(12, unified_num_classes), max(10, unified_num_classes)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[unified_class_names[i] for i in range(unified_num_classes)],
                yticklabels=[unified_class_names[i] for i in range(unified_num_classes)],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - GRU Multiclass on New Server Data', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CONFIG['confusion_matrix_save_path'], dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to {CONFIG['confusion_matrix_save_path']}")
    plt.close()
    
    # Plot per-class metrics
    if unified_num_classes <= 20:  # Only plot if not too many classes
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        classes = [unified_class_names[i] for i in range(unified_num_classes)]
        x_pos = np.arange(len(classes))
        
        axes[0].bar(x_pos, per_class_precision, color='skyblue', alpha=0.7)
        axes[0].set_title('Per-Class Precision', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Class', fontsize=10)
        axes[0].set_ylabel('Precision', fontsize=10)
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(classes, rotation=45, ha='right')
        axes[0].set_ylim([0, 1.1])
        axes[0].grid(True, alpha=0.3, axis='y')
        
        axes[1].bar(x_pos, per_class_recall, color='lightcoral', alpha=0.7)
        axes[1].set_title('Per-Class Recall', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Class', fontsize=10)
        axes[1].set_ylabel('Recall', fontsize=10)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(classes, rotation=45, ha='right')
        axes[1].set_ylim([0, 1.1])
        axes[1].grid(True, alpha=0.3, axis='y')
        
        axes[2].bar(x_pos, per_class_f1, color='lightgreen', alpha=0.7)
        axes[2].set_title('Per-Class F1 Score', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Class', fontsize=10)
        axes[2].set_ylabel('F1 Score', fontsize=10)
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(classes, rotation=45, ha='right')
        axes[2].set_ylim([0, 1.1])
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Per-Class Metrics - GRU Multiclass on New Server Data', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(CONFIG['per_class_metrics_save_path'], dpi=300, bbox_inches='tight')
        print(f"Per-class metrics plot saved to {CONFIG['per_class_metrics_save_path']}")
        plt.close()
    else:
        print(f"\nSkipping per-class metrics plot (too many classes: {unified_num_classes})")
    
    # Save results (standardized format for reporting)
    results = {
        'evaluation_date': datetime.now().isoformat(),
        'model_type': 'GRU',
        'task': 'multiclass',
        'hyperparameters': BEST_HYPERPARAMETERS,
        'train_data_file': CONFIG['train_data_file'],
        'test_data_file': CONFIG['test_data_file'],
        'pretrained_model_path': CONFIG['pretrained_model_path'],
        'model_save_path': CONFIG['model_save_path'],
        'num_features_used': len(aligned_features),
        'features_used': aligned_features,
        'num_classes': unified_num_classes,
        'class_names': {str(i): unified_class_names[i] for i in range(unified_num_classes)},
        'tag_to_class': {str(k): int(v) for k, v in unified_tag_to_class.items()},
        'test_metrics': {
            'loss': float(test_loss),
            'accuracy': float(test_acc),
            # Macro averages (primary metrics for reporting)
            'precision': float(test_prec_macro),  # Standardized name for macro precision
            'precision_macro': float(test_prec_macro),  # Keep explicit name too
            'recall': float(test_rec_macro),  # Standardized name for macro recall
            'recall_macro': float(test_rec_macro),  # Keep explicit name too
            'f1': float(test_f1_macro),  # Standardized name for macro F1
            'f1_macro': float(test_f1_macro),  # Keep explicit name too
            # Micro averages
            'precision_micro': float(test_prec_micro),
            'recall_micro': float(test_rec_micro),
            'f1_micro': float(test_f1_micro),
            # Weighted averages
            'precision_weighted': float(test_prec_weighted),
            'recall_weighted': float(test_rec_weighted),
            'f1_weighted': float(test_f1_weighted)
        },
        'per_class_metrics': {
            'precision': [float(x) for x in per_class_precision],
            'recall': [float(x) for x in per_class_recall],
            'f1': [float(x) for x in per_class_f1],  # Standardized to 'f1'
            'f1_score': [float(x) for x in per_class_f1]  # Keep both for backward compatibility
        },
        'confusion_matrix': cm.tolist(),
        'class_distribution_test': {
            str(i): int((np.array(test_labels) == i).sum()) 
            for i in range(unified_num_classes)
        },
        'total_test_samples': len(test_labels)
    }
    
    with open(CONFIG['results_save_path'], 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {CONFIG['results_save_path']}")
    
    # Print summary in report-ready format
    print("\n" + "="*80)
    print("SUMMARY FOR REPORTING")
    print("="*80)
    print(f"Model: {results['model_type']} ({results['task']})")
    print(f"Test Dataset: {CONFIG['test_data_file']}")
    print(f"Number of Classes: {unified_num_classes}")
    print(f"\nPerformance Metrics (Macro Average):")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_prec_macro:.4f}")
    print(f"  Recall:    {test_rec_macro:.4f}")
    print(f"  F1 Score:  {test_f1_macro:.4f}")
    print(f"\nPerformance Metrics (Weighted Average):")
    print(f"  Precision: {test_prec_weighted:.4f}")
    print(f"  Recall:    {test_rec_weighted:.4f}")
    print(f"  F1 Score:  {test_f1_weighted:.4f}")
    print(f"\nTest Set Distribution:")
    print(f"  Total Samples: {results['total_test_samples']:,} sequences")
    print(f"  Classes Present: {len([k for k, v in results['class_distribution_test'].items() if v > 0])}")
    print("="*80)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()

