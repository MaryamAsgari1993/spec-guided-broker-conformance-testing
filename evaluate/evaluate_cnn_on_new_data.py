"""
Evaluate CNN Binary Model on New MQTT Server Data
==================================================
This script evaluates a CNN model (with best hyperparameters) on a new 
preprocessed dataset from another MQTT server.

Usage:
------
Option 1: Use a pre-trained model (RECOMMENDED - no training needed)
  1. Edit CONFIG['pretrained_model_path'] to point to your trained model (.pth file)
  2. Edit CONFIG['test_data_file'] to point to your new preprocessed CSV file
  3. Run: python evaluate_cnn_on_new_data.py

Option 2: Train a new model first (if no pre-trained model available)
  1. Edit CONFIG['train_data_file'] to point to training data
  2. Edit CONFIG['test_data_file'] to point to your new preprocessed CSV file
  3. Run: python evaluate_cnn_on_new_data.py

Option 3: Command line arguments
  python evaluate_cnn_on_new_data.py <test_data_file> [pretrained_model_path]
  
  Example:
  python evaluate_cnn_on_new_data.py EMQX_preprocessed.csv best_cnn_model.pth

Requirements:
-------------
- The new preprocessed data should have the same format as the training data
- Must include 'Target' column (0=Normal, 1=Violation)
- Must include 'tcp.stream' column for sequence grouping
- Should have 'seq_position' or 'frame.number' for ordering
- If using pre-trained model, it must match the hyperparameters in this script

Output:
-------
- Results: evaluation_results_new_server.json
- Confusion matrix: confusion_matrix_new_server.png
- ROC curve: roc_curve_new_server.png

Author: Maryam
Date: December 2025
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
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
warnings.filterwarnings('ignore')

from models import CNNClassifier

# =============================================================================
# CONFIGURATION - Best Hyperparameters for CNN Binary Classification
# =============================================================================

# Best hyperparameters from nested CV results
BEST_HYPERPARAMETERS = {
    'max_seq_length': 100,
    'cnn_num_filters': [64, 128, 256],
    'cnn_kernel_size': 3,
    'cnn_pooling_type': 'max',
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
    'pretrained_model_path': '../oracle/best_cnn_binary_model.pth',
    'train_data_file': '../traces/mosquitto/mqtt_preprocessed_v2.csv',
    'test_data_file': '../traces/emqx/EMQX_preprocessed.csv',
    'model_save_path': '../oracle/best_cnn_binary_model.pth',
    'results_save_path': '../results/tables/evaluation_results_new_server.json',
    'confusion_matrix_save_path': '../results/heatmaps/confusion_matrix_new_server.png'
}

# Feature columns (same as in train_cnn_binary_model.py)
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
    """PyTorch Dataset for MQTT sequences."""
    
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
    Load preprocessed data and prepare sequences.
    
    Parameters:
    -----------
    filepath : str
        Path to preprocessed CSV file
    feature_columns : list
        List of feature column names to use
        
    Returns:
    --------
    tuple: (sequences, labels, available_features)
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
    labels = []
    
    for stream_id, group in df.groupby('tcp.stream'):
        group = group.sort_values('seq_position' if 'seq_position' in group.columns else 'frame.number')
        seq_features = group[available_features].values.astype(np.float32)
        
        # Get label (should be in 'Target' column)
        if 'Target' in group.columns:
            target_value = group['Target'].iloc[0]
            # Handle NaN values
            if pd.isna(target_value):
                print("  WARNING: 'Target' column contains NaN. Using default label 0.")
                seq_label = 0
            else:
                seq_label = int(target_value)
        else:
            print("  WARNING: 'Target' column not found. Using default label 0.")
            seq_label = 0
        
        sequences.append(seq_features)
        labels.append(seq_label)
    
    print(f"  Created {len(sequences):,} sequences")
    
    # Print class distribution
    label_counts = pd.Series(labels).value_counts().sort_index()
    print(f"\n  Class distribution:")
    for label, count in label_counts.items():
        class_name = "Normal" if label == 0 else "Violation"
        print(f"    {class_name} ({label}): {count:,} sequences ({count/len(labels)*100:.2f}%)")
    
    return sequences, labels, available_features

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

def evaluate(model, data_loader, criterion, device, return_probs=False):
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
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    
    if return_probs:
        probs_positive = [prob[1] for prob in all_probs]
        return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels, probs_positive
    else:
        return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels

def train_model(model, train_loader, val_loader, config):
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
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)
        
        print(f"{epoch+1:>6} | {train_loss:>10.4f} | {train_acc:>9.4f} | {val_loss:>8.4f} | "
              f"{val_acc:>7.4f} | {val_f1:>7.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"       ↑ New best! (F1: {best_val_f1:.4f})")
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

# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

def main():
    """Main evaluation function."""
    print("="*80)
    print("CNN BINARY MODEL EVALUATION ON NEW MQTT SERVER DATA")
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
            print("     python evaluate_cnn_on_new_data.py <path_to_preprocessed_data.csv>")
            print("\nExample:")
            print("  python evaluate_cnn_on_new_data.py EMQX_preprocessed.csv")
            print("  python evaluate_cnn_on_new_data.py ../EMQX_new_server_data.csv")
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
    sequences_train, labels_train, features_train = load_data(
        CONFIG['train_data_file'], FEATURE_COLUMNS
    )
    
    # Load test data
    print("\n" + "="*80)
    print("STEP 2: LOADING TEST DATA (NEW SERVER)")
    print("="*80)
    sequences_test, labels_test, features_test = load_data(
        CONFIG['test_data_file'], FEATURE_COLUMNS
    )
    
    # Normalize and align features
    print("\n" + "="*80)
    print("STEP 3: NORMALIZING AND ALIGNING FEATURES")
    print("="*80)
    sequences_train_norm, sequences_test_norm, scaler, aligned_features = normalize_sequences(
        sequences_train, sequences_test, features_train, features_test
    )
    
    # Split training data for validation
    print("\n" + "="*80)
    print("STEP 4: SPLITTING TRAINING DATA")
    print("="*80)
    seq_train, seq_val, y_train, y_val = train_test_split(
        sequences_train_norm, labels_train, test_size=0.1,
        random_state=BEST_HYPERPARAMETERS['random_seed'], stratify=labels_train
    )
    print(f"  Train: {len(seq_train):,} sequences")
    print(f"  Val:   {len(seq_val):,} sequences")
    
    # Create data loaders
    train_dataset = MQTTSequenceDataset(seq_train, y_train, BEST_HYPERPARAMETERS['max_seq_length'])
    val_dataset = MQTTSequenceDataset(seq_val, y_val, BEST_HYPERPARAMETERS['max_seq_length'])
    test_dataset = MQTTSequenceDataset(sequences_test_norm, labels_test, BEST_HYPERPARAMETERS['max_seq_length'])
    
    train_loader = DataLoader(train_dataset, batch_size=BEST_HYPERPARAMETERS['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BEST_HYPERPARAMETERS['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BEST_HYPERPARAMETERS['batch_size'], shuffle=False)
    
    # Create model
    print("\n" + "="*80)
    print("STEP 5: CREATING MODEL")
    print("="*80)
    input_size = len(aligned_features)
    print(f"  Input size: {input_size} features")
    print(f"  Hyperparameters:")
    print(f"    Sequence length: {BEST_HYPERPARAMETERS['max_seq_length']}")
    print(f"    CNN filters: {BEST_HYPERPARAMETERS['cnn_num_filters']}")
    print(f"    Kernel size: {BEST_HYPERPARAMETERS['cnn_kernel_size']}")
    print(f"    Pooling type: {BEST_HYPERPARAMETERS['cnn_pooling_type']}")
    print(f"    Learning rate: {BEST_HYPERPARAMETERS['learning_rate']}")
    print(f"    Dropout: {BEST_HYPERPARAMETERS['dropout']}")
    
    model = CNNClassifier(
        input_size=input_size,
        cnn_num_filters=BEST_HYPERPARAMETERS['cnn_num_filters'],
        cnn_kernel_size=BEST_HYPERPARAMETERS['cnn_kernel_size'],
        cnn_pool_size=2,  # Not used but required by model
        cnn_pooling_type=BEST_HYPERPARAMETERS['cnn_pooling_type'],
        num_classes=2,
        dropout=BEST_HYPERPARAMETERS['dropout']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Load or train model
    if CONFIG['pretrained_model_path'] and os.path.exists(CONFIG['pretrained_model_path']):
        print("\n" + "="*80)
        print("STEP 6: LOADING PRE-TRAINED MODEL")
        print("="*80)
        try:
            model.load_state_dict(torch.load(CONFIG['pretrained_model_path'], map_location=BEST_HYPERPARAMETERS['device']))
            print(f"✓ Successfully loaded model from {CONFIG['pretrained_model_path']}")
        except Exception as e:
            print(f"✗ ERROR: Failed to load model from {CONFIG['pretrained_model_path']}")
            print(f"  Error: {e}")
            print("  Will train a new model instead...")
            print("\n" + "="*80)
            print("STEP 6: TRAINING MODEL")
            print("="*80)
            model = train_model(model, train_loader, val_loader, BEST_HYPERPARAMETERS)
            # Save the newly trained model
            torch.save(model.state_dict(), CONFIG['model_save_path'])
            print(f"\n✓ Model saved to {CONFIG['model_save_path']} for future use")
    else:
        # Train model
        print("\n" + "="*80)
        print("STEP 6: TRAINING MODEL")
        print("="*80)
        model = train_model(model, train_loader, val_loader, BEST_HYPERPARAMETERS)
        
        # Save the trained model
        torch.save(model.state_dict(), CONFIG['model_save_path'])
        print(f"\n✓ Model saved to {CONFIG['model_save_path']} for future use")
    
    # Evaluate on test data
    print("\n" + "="*80)
    print("STEP 7: EVALUATING ON NEW SERVER DATA")
    print("="*80)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_prec, test_rec, test_f1, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, criterion, BEST_HYPERPARAMETERS['device'], return_probs=True
    )
    
    print(f"\nTest Results:")
    print(f"  Loss:      {test_loss:.4f}")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_rec:.4f}")
    print(f"  F1 Score:  {test_f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Normal  Violation")
    print(f"Actual Normal      {cm[0,0]:4d}     {cm[0,1]:4d}")
    print(f"Actual Violation      {cm[1,0]:4d}     {cm[1,1]:4d}")
    
    # Classification Report
    print(f"\nClassification Report:")
    print(classification_report(test_labels, test_preds, 
                              target_names=['Normal', 'Violation']))
    
    # ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(test_labels, test_probs)
    roc_auc = auc(fpr, tpr)
    print(f"\nROC AUC: {roc_auc:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - CNN on New Server Data', fontsize=14, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Normal', 'Violation'])
    plt.yticks(tick_marks, ['Normal', 'Violation'])
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(CONFIG['confusion_matrix_save_path'], dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to {CONFIG['confusion_matrix_save_path']}")
    plt.close()
    
    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title('ROC Curve - CNN on New Server Data', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.text(0.6, 0.2, f'AUC = {roc_auc:.4f}', 
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    roc_save_path = CONFIG['confusion_matrix_save_path'].replace('confusion_matrix', 'roc_curve')
    plt.savefig(roc_save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {roc_save_path}")
    plt.close()
    
    # Save results
    results = {
        'evaluation_date': datetime.now().isoformat(),
        'model_type': 'CNN',
        'hyperparameters': BEST_HYPERPARAMETERS,
        'train_data_file': CONFIG['train_data_file'],
        'test_data_file': CONFIG['test_data_file'],
        'pretrained_model_path': CONFIG['pretrained_model_path'],
        'model_save_path': CONFIG['model_save_path'],
        'num_features_used': len(aligned_features),
        'features_used': aligned_features,
        'test_metrics': {
            'loss': float(test_loss),
            'accuracy': float(test_acc),
            'precision': float(test_prec),
            'recall': float(test_rec),
            'f1_score': float(test_f1),
            'roc_auc': float(roc_auc)
        },
        'confusion_matrix': cm.tolist(),
        'class_distribution_test': {
            'normal': int((np.array(test_labels) == 0).sum()),
            'violation': int((np.array(test_labels) == 1).sum())
        }
    }
    
    with open(CONFIG['results_save_path'], 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {CONFIG['results_save_path']}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()

