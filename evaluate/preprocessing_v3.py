"""
MQTT Dataset Preprocessing Pipeline
===============================================
Author: Maryam
Date: December 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')


def get_topic_depth(text):
    """Calculate topic hierarchy depth."""
    if pd.isna(text) or str(text).strip() == '':
        return 0
    return str(text).count('/') + 1

def safe_numeric_convert(series, fill_value=0):
    """Safely convert series to numeric."""
    return pd.to_numeric(series, errors='coerce').fillna(fill_value)


COLUMNS_TO_KEEP = [
    'tcp.stream', 'time.delta.prevPkt', 'frame.number',
    'mqtt.len', 'mqtt.msgtype', 'mqtt.hdr_reserved',
    'mqtt.qos', 'mqtt.dupflag', 'mqtt.msgid', 
    'mqtt.topic', 'mqtt.topic_len', 'mqtt.topic.null', 'mqtt.topic.invalidUTF', 'mqtt.topic.empty', 'mqtt.topic.wildcard',
    'mqtt.willtopic_len', 'mqtt.willtopic.null', 'mqtt.willtopic.empty', 'mqtt.willtopic.wildcard',
    'mqtt.willmsg_len', 'mqtt.msg.empty',
    'mqtt.conflag.willflag', 'mqtt.protoname', 'mqtt.kalive', 'mqtt.conflag.reserved',
    'mqtt.connack.reason_code', 'mqtt.disconnect.reason_code',
    'mqtt.subscription_options', 'mqtt.subscription_options_reserved',
    'mqtt.suback.reason_code', 'mqtt.unsuback.reason_code',
    'kalive.Violated', 'Direction',
    'ping.missed',
    'Target', 'Tag'
]

# Configuration for v2
CONFIG_V2 = {
    'input_file': '../traces/nanomq/Final-NanSET.csv',
    'output_file': '../traces/nanomq/Final-NanSET_preprocessed.csv',
    'metadata_file': '../traces/nanomq/preprocessing_metadata.json',
    'random_seed': 42,
    'version': '2.0'
}

# MQTT Message Type Constants
CONNECT = 1
CONNACK = 2
PUBLISH = 3
PUBACK = 4
PUBREC = 5
PUBREL = 6
PUBCOMP = 7
SUBSCRIBE = 8
SUBACK = 9
UNSUBSCRIBE = 10
UNSUBACK = 11
PINGREQ = 12
PINGRESP = 13
DISCONNECT = 14



def add_sequence_features_for_construction(df):
    """
    Add sequence features ONLY for sequence construction (not training features).
    
    These are kept in the dataset but will be excluded from FEATURE_COLUMNS.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset sorted by tcp.stream and frame.number
        
    Returns:
    --------
    pd.DataFrame : Dataset with sequence construction features
    """
    print("\nAdding sequence construction features (for grouping only, not training)...")
    
    df = df.sort_values(['tcp.stream', 'frame.number']).reset_index(drop=True)
    
    # Position within stream (for ordering)
    df['seq_position'] = df.groupby('tcp.stream').cumcount()
    
    # Total stream length
    stream_lengths = df.groupby('tcp.stream').size()
    df['seq_length'] = df['tcp.stream'].map(stream_lengths)
    
    # Is first packet in stream
    df['is_first_packet'] = (df['seq_position'] == 0).astype(int)
    
    # Is same frame
    df['is_same_frame'] = (
        (df['time.delta.prevPkt'] == 0) & 
        (df['seq_position'] > 0)
    ).astype(int)
    
    print(f"  Created sequence construction features (not used in model)")
    print(f"  Total streams: {df['tcp.stream'].nunique():,}")
    
    return df


def create_final_dataset_v2(df):
    """
    Create final dataset with updated features (no IP addresses, no sequence metadata in features).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Fully processed dataset
        
    Returns:
    --------
    pd.DataFrame : Final dataset ready for modeling
    """
    print("\nCreating final dataset (v2)...")
    
    # Define final feature columns (EXCLUDING sequence construction features)
    final_columns = [
        # Sequence identifier (for grouping)
        'tcp.stream',
        
        # Timing
        'time.delta.prevPkt',
        'kalive.Violated',
        
        # Direction (replaces IP addresses)
        'Direction',
        
        # Core MQTT
        'mqtt.len',
        'mqtt.msgtype',
        'mqtt.hdr_reserved',
        
        # QoS
        'mqtt.qos',
        'mqtt.dupflag',
        'msgid_value',
        'msgid_is_present',
        
        # Topic features
        # Note: mqtt.topic_len already exists in CSV, use it directly
        'mqtt.topic_len',
        'topic_depth',  # Calculated from mqtt.topic if available
        # Note: The following topic features are read from input CSV (not generated):
        'mqtt.topic.null',
        'mqtt.topic.wildcard',
        'mqtt.topic.invalidUTF',
        'mqtt.topic.empty',
        
        # Will (all from CSV, not generated)
        'mqtt.willtopic_len',
        'mqtt.willtopic.null',
        'mqtt.willtopic.wildcard',
        'mqtt.willtopic.empty',
        'mqtt.willmsg_len',
        'mqtt.msg.empty',
        'mqtt.conflag.willflag',
        
        # Connection
        'protoname_is_valid',
        'mqtt.kalive',
        'mqtt.conflag.reserved',
        'mqtt.connack.reason_code',
        'mqtt.disconnect.reason_code',
        
        # Subscription
        'mqtt.subscription_options_numeric',
        'mqtt.subscription_options_reserved_numeric',
        'mqtt.suback.reason_code_numeric',
        'mqtt.unsuback.reason_code_numeric',
        
        # Protocol violation detection
        'suback_out_of_order',
        'unsuback_out_of_order',
        'subscribe_msgid_mismatch',
        'unsubscribe_msgid_mismatch',
        'subscribe_ack_missing',
        'unsubscribe_ack_missing',
        'pending_subscribe_count',
        'pending_unsubscribe_count',
        'msgid_order_deviation',
        
        # PINGREQ keep-alive detection (Tag_2)
        'pingreq_missing',
        'time_since_last_client_control',
        'ping.missed',  # From CSV
        
        # Sequence construction features (for grouping, NOT in model)
        'seq_position',
        'seq_length',
        'is_first_packet',
        'is_same_frame',
        
        # Targets
        'Target',
        'Tag'
        # Note: 'Des' is not needed - we have 'Tag' for attack type
    ]
    
    # Select columns that exist
    existing_cols = [col for col in final_columns if col in df.columns]
    missing_cols = [col for col in final_columns if col not in df.columns]
    
    if missing_cols:
        print(f"  Warning: Missing columns: {missing_cols}")
        # Special check for willmsg_len
        if 'mqtt.willmsg_len' in missing_cols:
            print(f"    Checking for mqtt.willmsg_len variations...")
            willmsg_cols = [c for c in df.columns if 'willmsg' in c.lower()]
            if willmsg_cols:
                print(f"    Found willmsg columns: {willmsg_cols}")
                # Try to use the first one if it exists
                if len(willmsg_cols) > 0:
                    actual_col = willmsg_cols[0]
                    print(f"    Using '{actual_col}' as mqtt.willmsg_len")
                    df.rename(columns={actual_col: 'mqtt.willmsg_len'}, inplace=True)
                    if 'mqtt.willmsg_len' not in existing_cols:
                        existing_cols.append('mqtt.willmsg_len')
                    if 'mqtt.willmsg_len' in missing_cols:
                        missing_cols.remove('mqtt.willmsg_len')
    
    df_final = df[existing_cols].copy()
    
    # Verify mqtt.willmsg_len is in the final output
    if 'mqtt.willmsg_len' in final_columns:
        if 'mqtt.willmsg_len' in df_final.columns:
            print(f"  ✓ mqtt.willmsg_len is included in final dataset")
        else:
            print(f"  ✗ ERROR: mqtt.willmsg_len should be in final output but is missing!")
            will_cols = [c for c in df_final.columns if 'will' in c.lower()]
            print(f"    Available will-related columns: {will_cols}")
    
    print(f"  Final dataset: {len(df_final):,} rows, {len(existing_cols)} columns")
    print(f"  Training features (excluding seq construction): {len(existing_cols) - 8}")  # -8 for seq construction + targets
    
    return df_final


# =============================================================================
# MAIN PREPROCESSING PIPELINE V2
# =============================================================================


def load_data(filepath):
    """Load the raw MQTT dataset."""
    print(f"Loading data from {filepath}...")
    if filepath.endswith('.csv'):
        # Try different encodings to handle non-UTF-8 characters
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-8-sig']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, low_memory=False, encoding=encoding)
                print(f"  Successfully loaded with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            # Last resort: use utf-8 with error handling
            print("  Warning: Using utf-8 with errors='replace' (some characters may be lost)")
            df = pd.read_csv(filepath, low_memory=False, encoding='utf-8', errors='replace')
    else:
        df = pd.read_excel(filepath)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df

def select_columns(df, columns):
    """Select relevant columns from dataset."""
    print(f"\nSelecting {len(columns)} columns...")
    
    # First, strip whitespace from all column names in the dataframe
    df.columns = df.columns.str.strip()
    
    existing = []
    missing = []
    column_mapping = {}  # Map original column name to actual column name in CSV
    
    for col in columns:
        if col in df.columns:
            existing.append(col)
            column_mapping[col] = col
        else:
            # Check for column with trailing/leading spaces
            stripped_cols = {c.strip(): c for c in df.columns}
            if col in stripped_cols:
                actual_col = stripped_cols[col]
                existing.append(actual_col)
                column_mapping[col] = actual_col
                print(f"  Note: '{col}' found as '{actual_col}' (whitespace difference)")
            else:
                missing.append(col)
    
    if missing:
        print(f"  Warning: {len(missing)} columns not found: {missing}")
        # Check for willmsg_len specifically
        if 'mqtt.willmsg_len' in missing:
            willmsg_variations = [c for c in df.columns if 'willmsg' in c.lower()]
            if willmsg_variations:
                print(f"    'mqtt.willmsg_len' not found, but found: {willmsg_variations}")
    
    # Select columns (use actual column names from CSV)
    cols_to_select = [column_mapping.get(col, col) for col in existing if col in column_mapping or col in df.columns]
    df_selected = df[cols_to_select].copy()
    
    # Rename columns to standard names (remove any whitespace issues)
    rename_dict = {v: k for k, v in column_mapping.items()}
    df_selected.rename(columns=rename_dict, inplace=True)
    
    print(f"  Selected {len(existing)} columns")
    return df_selected

def engineer_topic_features(df):
    """Engineer topic_depth feature only (mqtt.topic_len already exists in CSV)."""
    print("\nEngineering topic features...")
    # Note: mqtt.topic_len, mqtt.topic.null, mqtt.topic.wildcard, mqtt.topic.invalidUTF, mqtt.topic.empty already exist in CSV
    # Only calculate topic_depth if mqtt.topic text column exists
    if 'mqtt.topic' in df.columns:
        df['topic_depth'] = df['mqtt.topic'].apply(get_topic_depth)
        print(f"  Created 1 topic feature (topic_depth)")
    else:
        df['topic_depth'] = 0
        print(f"  Skipped topic_depth (mqtt.topic column not found, using default 0)")
    return df

def engineer_will_features(df):
    """Engineer will features if missing from CSV."""
    print("\nEngineering Will features...")
    features_created = 0
    
    # Check if mqtt.willmsg_len exists, if not create it from mqtt.willmsg
    if 'mqtt.willmsg_len' not in df.columns:
        if 'mqtt.willmsg' in df.columns:
            # Create willmsg_len from willmsg text
            df['mqtt.willmsg_len'] = df['mqtt.willmsg'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
            print(f"  Created mqtt.willmsg_len from mqtt.willmsg")
            features_created += 1
        else:
            # If mqtt.willmsg doesn't exist either, create a default column
            df['mqtt.willmsg_len'] = 0
            print(f"  Created mqtt.willmsg_len with default value 0 (mqtt.willmsg not found)")
            features_created += 1
    else:
        print(f"  mqtt.willmsg_len already exists in CSV")
    
    if features_created > 0:
        print(f"  Created {features_created} will feature(s)")
    else:
        print(f"  All will features already exist in CSV")
    
    return df

def engineer_msgid_features(df):
    """Engineer features from mqtt.msgid field."""
    print("\nEngineering message ID features...")
    df['msgid_is_present'] = df['mqtt.msgid'].notna().astype(int)
    # Use 0 for missing msgid (since we have msgid_is_present flag to indicate presence)
    # This works better with StandardScaler normalization in sequence models
    df['msgid_value'] = safe_numeric_convert(df['mqtt.msgid'], fill_value=0)
    print(f"  Created 2 msgid features")
    return df

def engineer_protocol_features(df):
    """Engineer features from protocol name."""
    print("\nEngineering protocol features...")
    df['protoname_is_valid'] = (df['mqtt.protoname'].astype(str).str.upper() == 'MQTT').astype(int)
    # Use 0 for missing (not applicable) instead of -1 for sequence models
    df.loc[df['mqtt.protoname'].isna(), 'protoname_is_valid'] = 0
    print(f"  Created 1 protocol feature")
    return df

def engineer_protocol_violation_features(df):
    """
    Engineer features to detect MQTT protocol violations for SUBSCRIBE/SUBACK and UNSUBSCRIBE/UNSUBACK pairs.
    
    Detects:
    - Message ID mismatches
    - Out-of-order acknowledgments
    - Missing acknowledgments
    - Order violations
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset sorted by tcp.stream and frame.number, with direction column
        
    Returns:
    --------
    pd.DataFrame : Dataset with protocol violation features added
    """
    print("\nEngineering protocol violation features...")
    
    # Sort by stream and frame number
    df = df.sort_values(['tcp.stream', 'frame.number']).reset_index(drop=True)
    
    # Initialize feature columns
    df['suback_out_of_order'] = 0
    df['unsuback_out_of_order'] = 0
    df['subscribe_msgid_mismatch'] = 0
    df['unsubscribe_msgid_mismatch'] = 0
    df['subscribe_ack_missing'] = 0
    df['unsubscribe_ack_missing'] = 0
    df['pending_subscribe_count'] = 0
    df['pending_unsubscribe_count'] = 0
    df['msgid_order_deviation'] = 0
    
    # Track state per stream: queues of expected msgids in order
    stream_state = {}
    
    for idx, row in df.iterrows():
        stream_id = row['tcp.stream']
        msgtype = int(row['mqtt.msgtype']) if not pd.isna(row['mqtt.msgtype']) else -1
        direction = row['Direction']
        msgid = safe_numeric_convert(pd.Series([row['mqtt.msgid']]), fill_value=-1).iloc[0]
        frame_num = row['frame.number']
        
        # Initialize stream state if needed
        if stream_id not in stream_state:
            stream_state[stream_id] = {
                'pending_subscribe': [],  # List of (msgid, frame_num, idx) tuples in order
                'pending_unsubscribe': [],  # List of (msgid, frame_num, idx) tuples in order
            }
        
        state = stream_state[stream_id]
        
        # Update pending counts
        df.at[idx, 'pending_subscribe_count'] = len(state['pending_subscribe'])
        df.at[idx, 'pending_unsubscribe_count'] = len(state['pending_unsubscribe'])
        
        # Handle SUBSCRIBE (client→server)
        if msgtype == SUBSCRIBE and direction in [0, 2] and msgid >= 0:
            state['pending_subscribe'].append((msgid, frame_num, idx))
            # Mark as potentially missing ACK (will be cleared if ACK arrives)
            df.at[idx, 'subscribe_ack_missing'] = 1
        
        # Handle SUBACK (server→client)
        elif msgtype == SUBACK and direction in [1, 3] and msgid >= 0:
            if len(state['pending_subscribe']) > 0:
                # Check if msgid matches the first (expected) one
                expected_msgid, expected_frame, subscribe_idx = state['pending_subscribe'][0]
                
                if msgid == expected_msgid:
                    # Correct match - remove from queue
                    state['pending_subscribe'].pop(0)
                    # Clear missing flag for the original SUBSCRIBE (using stored index)
                    df.at[subscribe_idx, 'subscribe_ack_missing'] = 0
                else:
                    # Mismatch - check if it's in the queue at all
                    msgid_found = False
                    expected_position = -1
                    subscribe_idx_to_clear = None
                    for pos, (m, f, i) in enumerate(state['pending_subscribe']):
                        if m == msgid:
                            msgid_found = True
                            expected_position = pos
                            subscribe_idx_to_clear = i
                            break
                    
                    if msgid_found:
                        # Out of order - msgid exists but not at front
                        df.at[idx, 'suback_out_of_order'] = 1
                        df.at[idx, 'msgid_order_deviation'] = expected_position
                        # Clear missing flag and remove from queue
                        if subscribe_idx_to_clear is not None:
                            df.at[subscribe_idx_to_clear, 'subscribe_ack_missing'] = 0
                        state['pending_subscribe'] = [
                            (m, f, i) for m, f, i in state['pending_subscribe'] if m != msgid
                        ]
                    else:
                        # Complete mismatch - msgid not in pending queue
                        df.at[idx, 'subscribe_msgid_mismatch'] = 1
            else:
                # No pending SUBSCRIBE but SUBACK arrived - violation
                df.at[idx, 'subscribe_msgid_mismatch'] = 1
        
        # Handle UNSUBSCRIBE (client→server)
        elif msgtype == UNSUBSCRIBE and direction in [0, 2] and msgid >= 0:
            state['pending_unsubscribe'].append((msgid, frame_num, idx))
            df.at[idx, 'unsubscribe_ack_missing'] = 1
        
        # Handle UNSUBACK (server→client)
        elif msgtype == UNSUBACK and direction in [1, 3] and msgid >= 0:
            if len(state['pending_unsubscribe']) > 0:
                expected_msgid, expected_frame, unsubscribe_idx = state['pending_unsubscribe'][0]
                
                if msgid == expected_msgid:
                    # Correct match
                    state['pending_unsubscribe'].pop(0)
                    # Clear missing flag (using stored index)
                    df.at[unsubscribe_idx, 'unsubscribe_ack_missing'] = 0
                else:
                    # Check if in queue
                    msgid_found = False
                    expected_position = -1
                    unsubscribe_idx_to_clear = None
                    for pos, (m, f, i) in enumerate(state['pending_unsubscribe']):
                        if m == msgid:
                            msgid_found = True
                            expected_position = pos
                            unsubscribe_idx_to_clear = i
                            break
                    
                    if msgid_found:
                        df.at[idx, 'unsuback_out_of_order'] = 1
                        df.at[idx, 'msgid_order_deviation'] = expected_position
                        # Clear missing flag and remove from queue
                        if unsubscribe_idx_to_clear is not None:
                            df.at[unsubscribe_idx_to_clear, 'unsubscribe_ack_missing'] = 0
                        state['pending_unsubscribe'] = [
                            (m, f, i) for m, f, i in state['pending_unsubscribe'] if m != msgid
                        ]
                    else:
                        df.at[idx, 'unsubscribe_msgid_mismatch'] = 1
            else:
                df.at[idx, 'unsubscribe_msgid_mismatch'] = 1
    
    # Convert to int
    int_cols = ['suback_out_of_order', 'unsuback_out_of_order', 'subscribe_msgid_mismatch',
                'unsubscribe_msgid_mismatch', 'subscribe_ack_missing', 'unsubscribe_ack_missing',
                'pending_subscribe_count', 'pending_unsubscribe_count']
    for col in int_cols:
        df[col] = df[col].astype(int)
    
    # Print statistics
    print(f"  Created 9 protocol violation features:")
    print(f"    suback_out_of_order: {(df['suback_out_of_order'] == 1).sum():,}")
    print(f"    unsuback_out_of_order: {(df['unsuback_out_of_order'] == 1).sum():,}")
    print(f"    subscribe_msgid_mismatch: {(df['subscribe_msgid_mismatch'] == 1).sum():,}")
    print(f"    unsubscribe_msgid_mismatch: {(df['unsubscribe_msgid_mismatch'] == 1).sum():,}")
    print(f"    subscribe_ack_missing: {(df['subscribe_ack_missing'] == 1).sum():,}")
    print(f"    unsubscribe_ack_missing: {(df['unsubscribe_ack_missing'] == 1).sum():,}")
    print(f"    Max pending_subscribe_count: {df['pending_subscribe_count'].max()}")
    print(f"    Max pending_unsubscribe_count: {df['pending_unsubscribe_count'].max()}")
    print(f"    Max msgid_order_deviation: {df['msgid_order_deviation'].max()}")
    
    return df

def engineer_pingreq_keepalive_features(df):
    """
    Engineer features to detect PINGREQ keep-alive violations (Tag_2).
    
    Rule: If Keep Alive is non-zero and in the absence of sending any other 
    MQTT Control Packets, the Client MUST send a PINGREQ packet.
    
    The existing kalive.Violated only checks time.delta.prevPkt,
    but we need to track:
    - Time since last CLIENT control packet (not just previous packet)
    - Whether PINGREQ was sent when expected
    - Whether other control packets reset the timer
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset sorted by tcp.stream and frame.number, with Direction column
        
    Returns:
    --------
    pd.DataFrame : Dataset with PINGREQ keep-alive features added
    """
    print("\nEngineering PINGREQ keep-alive features...")
    
    # Sort by stream and frame number
    df = df.sort_values(['tcp.stream', 'frame.number']).reset_index(drop=True)
    
    # Initialize feature columns
    df['pingreq_missing'] = 0
    df['time_since_last_client_control'] = 0.0
    
    # Track state per stream
    stream_state = {}
    
    # All MQTT control packet types from client (these reset the keep-alive timer)
    CLIENT_CONTROL_PACKETS = [CONNECT, PUBLISH, PUBACK, PUBREC, PUBREL, 
                              SUBSCRIBE, UNSUBSCRIBE, DISCONNECT]
    
    for idx, row in df.iterrows():
        stream_id = row['tcp.stream']
        msgtype = int(row['mqtt.msgtype']) if not pd.isna(row['mqtt.msgtype']) else -1
        direction = row['Direction']
        kalive = safe_numeric_convert(pd.Series([row['mqtt.kalive']]), fill_value=-1).iloc[0]
        time_delta = pd.to_numeric(row['time.delta.prevPkt'], errors='coerce')
        if pd.isna(time_delta):
            time_delta = 0
        
        # Initialize stream state if needed
        if stream_id not in stream_state:
            stream_state[stream_id] = {
                'last_client_control_time': 0.0,
                'cumulative_time': 0.0,
                'kalive_value': -1,
                'connected': False
            }
        
        state = stream_state[stream_id]
        state['cumulative_time'] += time_delta
        
        # Update kalive value from CONNECT packet
        if msgtype == CONNECT and direction in [0, 2]:
            if kalive > 0:
                state['kalive_value'] = kalive
            state['connected'] = True
            state['last_client_control_time'] = state['cumulative_time']
        
        # Mark connection established
        if msgtype == CONNACK and direction in [1, 3]:
            state['connected'] = True
        
        # Calculate time since last client control packet
        time_since_last_control = state['cumulative_time'] - state['last_client_control_time']
        df.at[idx, 'time_since_last_client_control'] = time_since_last_control
        
        # Check if this is a client control packet (resets timer)
        is_client_control = (msgtype in CLIENT_CONTROL_PACKETS and 
                            direction in [0, 2])
        is_pingreq = (msgtype == PINGREQ and direction in [0, 2])
        
        # Update last client control time if client sent a control packet
        if is_client_control or is_pingreq:
            state['last_client_control_time'] = state['cumulative_time']
        
        # Check for PINGREQ violations
        if state['connected'] and state['kalive_value'] > 0:
            # If time since last client control >= kalive, PINGREQ should have been sent
            if time_since_last_control >= state['kalive_value']:
                # If current packet is NOT PINGREQ and NOT a client control packet,
                # then PINGREQ is missing
                if not is_pingreq and not is_client_control:
                    # PINGREQ should have been sent but wasn't
                    df.at[idx, 'pingreq_missing'] = 1
                elif is_client_control and not is_pingreq:
                    # Another control packet was sent instead of PINGREQ
                    # This is technically allowed (resets timer), but we flag it
                    # as a potential violation if time significantly exceeded
                    if time_since_last_control >= state['kalive_value'] * 1.5:
                        df.at[idx, 'pingreq_missing'] = 1
        
        # Reset on DISCONNECT
        if msgtype == DISCONNECT:
            state['connected'] = False
            state['last_client_control_time'] = state['cumulative_time']
    
    # Convert to appropriate types
    df['pingreq_missing'] = df['pingreq_missing'].astype(int)
    df['time_since_last_client_control'] = df['time_since_last_client_control'].astype(float)
    
    # Print statistics
    print(f"  Created 2 PINGREQ keep-alive features:")
    print(f"    pingreq_missing: {(df['pingreq_missing'] == 1).sum():,}")
    print(f"    Max time_since_last_client_control: {df['time_since_last_client_control'].max():.2f}s")
    
    return df

def convert_numeric_columns(df):
    """Convert remaining columns to numeric types."""
    print("\nConverting columns to numeric...")
    numeric_cols = ['mqtt.subscription_options', 'mqtt.subscription_options_reserved', 'mqtt.suback.reason_code', 'mqtt.unsuback.reason_code']
    for col in numeric_cols:
        if col in df.columns:
            # Use 0 instead of -1 for sequence models (works better with normalization)
            df[col + '_numeric'] = safe_numeric_convert(df[col], fill_value=0)
            print(f"  Converted {col}")
    return df

def handle_missing_values(df):
    """Handle missing values appropriately for sequence models."""
    print("\nHandling missing values...")
    # For sequence models with normalization, use 0 instead of -1
    # 0 is more neutral and works better with StandardScaler
    fill_zero = ['mqtt.hdr_reserved', 'mqtt.qos', 'mqtt.dupflag', 'mqtt.kalive', 
                 'mqtt.conflag.willflag', 'mqtt.conflag.reserved', 
                 'mqtt.connack.reason_code', 'mqtt.disconnect.reason_code']
    for col in fill_zero:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            df[col] = df[col].fillna(0)
            if missing_count > 0:
                print(f"  {col}: filled {missing_count:,} missing with 0")
    # Tag uses -1 for normal traffic (not missing, but a valid class label)
    if 'Tag' in df.columns:
        missing_tags = df['Tag'].isna().sum()
        # Only fill actual missing values with -1 (which represents normal traffic class)
        df['Tag'] = df['Tag'].fillna(-1)
        if missing_tags > 0:
            print(f"  Tag: filled {missing_tags:,} missing with -1 (normal traffic class)")
    return df

def convert_all_features_to_numeric(df):
    """
    Convert all feature columns to numeric types, handling boolean strings.
    
    This ensures the data is ready for training without additional preprocessing.
    Handles 'True'/'False' string conversions and ensures all features are numeric.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with features
        
    Returns:
    --------
    pd.DataFrame : Dataset with all feature columns converted to numeric
    """
    print("\nConverting all feature columns to numeric...")
    
    # Columns to exclude from conversion (identifiers and labels)
    exclude_cols = {'tcp.stream', 'Target', 'Tag', 'frame.number'}
    
    # Get all columns that need conversion
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    converted_count = 0
    bool_string_count = 0
    
    for col in feature_cols:
        if col in df.columns:
            original_dtype = df[col].dtype
            
            # Handle boolean strings first
            if df[col].dtype == 'object':
                # Check if column contains boolean strings
                unique_vals = df[col].dropna().astype(str).unique()
                if any(val in ['True', 'False', 'true', 'false'] for val in unique_vals):
                    df[col] = df[col].replace({'True': 1, 'False': 0, 'true': 1, 'false': 0})
                    bool_string_count += 1
            
            # Convert to numeric (coerce errors to NaN, then fill with 0)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            if original_dtype != df[col].dtype:
                converted_count += 1
    
    print(f"  Converted {converted_count} columns to numeric")
    if bool_string_count > 0:
        print(f"  Handled boolean strings in {bool_string_count} columns")
    
    return df

def save_outputs(df, encodings, config):
    """Save processed dataset and metadata."""
    print("\nSaving outputs...")
    df.to_csv(config['output_file'], index=False)
    print(f"  Saved data to {config['output_file']}")
    metadata = {
        'preprocessing_version': config['version'],
        'preprocessing_date': datetime.now().isoformat(),
        'input_file': config['input_file'],
        'output_file': config['output_file'],
        'random_seed': config['random_seed'],
        'total_samples': len(df),
        'total_features': len(df.columns),
        'total_streams': int(df['tcp.stream'].nunique()),
        'class_distribution': {'normal': int((df['Target'] == 0).sum()), 'attack': int((df['Target'] == 1).sum())},
        'feature_columns': list(df.columns),
        'encodings': {k: {str(kk): int(vv) for kk, vv in v.items()} for k, v in encodings.items()}
    }
    with open(config['metadata_file'], 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {config['metadata_file']}")

def print_summary(df):
    """Print preprocessing summary statistics."""
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"\nDataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nClass Distribution:")
    print(f"  Normal (0): {(df['Target'] == 0).sum():,} ({(df['Target'] == 0).mean()*100:.2f}%)")
    print(f"  Attack (1): {(df['Target'] == 1).sum():,} ({(df['Target'] == 1).mean()*100:.2f}%)")
    print(f"\nSequence Statistics:")
    print(f"  Total streams: {df['tcp.stream'].nunique():,}")
    print(f"  Min stream length: {df['seq_length'].min()}")
    print(f"  Max stream length: {df['seq_length'].max()}")
    print(f"  Mean stream length: {df['seq_length'].mean():.2f}")
    print("\n" + "="*60)


def main_v2():
    """
    Main preprocessing pipeline execution for v2.
    """
    print("="*60)
    print("MQTT DATASET PREPROCESSING PIPELINE V2")
    print("="*60)
    print(f"Version: {CONFIG_V2['version']}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Step 1: Load data
    df = load_data(CONFIG_V2['input_file'])
    
    # Step 2: Select columns
    df = select_columns(df, COLUMNS_TO_KEEP)
    
    # Step 3: Engineer features
    df = engineer_topic_features(df)
    df = engineer_will_features(df)
    df = engineer_msgid_features(df)
    df = engineer_protocol_features(df)
    # Note: kalive.Violated already exists in CSV, so keepalive feature engineering removed
    df = engineer_protocol_violation_features(df)
    df = engineer_pingreq_keepalive_features(df)
    
    # Step 4: Add sequence features (for construction only)
    df = add_sequence_features_for_construction(df)
    
    # Step 5: Convert numeric columns
    df = convert_numeric_columns(df)
    
    # Step 6: Handle missing values
    df = handle_missing_values(df)
    
    # Step 7: Convert all features to numeric (comprehensive conversion)
    df = convert_all_features_to_numeric(df)
    
    # Step 8: Create final dataset
    df_final = create_final_dataset_v2(df)
    
    # Step 9: No encoding needed - we use Tag directly for multi-class classification
    encodings = {}
    
    # Step 10: Save outputs
    save_outputs(df_final, encodings, CONFIG_V2)
    
    # Step 11: Print summary
    print_summary(df_final)
    
    print("\nPreprocessing v2 complete!")
    return df_final


if __name__ == "__main__":
    df_processed = main_v2()

