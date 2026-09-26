# -*- coding: utf-8 -*-
"""
inspect_h5.py — Kiểm tra chi tiết cấu trúc checkpoint release/94,58.h5
=====================================================================
Trích xuất model_config từ HDF5, load_model() trực tiếp và in model.summary()
"""

import os
import sys
import json

# Fix Windows console UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

WEIGHTS_PATH = os.path.join('release', '94,58.h5')
if not os.path.exists(WEIGHTS_PATH):
    WEIGHTS_PATH = os.path.join('Structure', 'Structure 5 (final)', '94,58.h5')

print("=" * 70)
print(f"=== KIỂM TRA CHECKPOINT: {WEIGHTS_PATH} ===")
print(f"File size: {os.path.getsize(WEIGHTS_PATH):,} bytes\n")

# 1. Trích xuất metadata từ HDF5
print("--- [1] ĐỌC METADATA & MODEL_CONFIG GỐC TỪ HDF5 ---")
with h5py.File(WEIGHTS_PATH, 'r') as f:
    print(f"HDF5 Root keys: {list(f.keys())}")
    print(f"HDF5 Root attrs: {list(f.attrs.keys())}")
    
    if 'keras_version' in f.attrs:
        kv = f.attrs['keras_version']
        if isinstance(kv, bytes): kv = kv.decode('utf-8')
        print(f"Keras Version khi lưu: {kv}")

    if 'model_config' in f.attrs:
        raw_cfg = f.attrs['model_config']
        if isinstance(raw_cfg, bytes):
            raw_cfg = raw_cfg.decode('utf-8')
        cfg = json.loads(raw_cfg)
        print("\n=> CẤU HÌNH TỪNG LAYER TRÍCH TỪ 'model_config':")
        layers = cfg.get('config', {}).get('layers', [])
        for i, lyr in enumerate(layers):
            c = lyr.get('config', {})
            class_name = lyr.get('class_name')
            units = c.get('units')
            activation = c.get('activation')
            ret_seq = c.get('return_sequences')
            in_shape = c.get('batch_input_shape')
            print(f"  Layer {i+1}: {class_name:<6} | units={str(units):<4} | activation={str(activation):<8} | return_sequences={str(ret_seq):<5} | input_shape={in_shape}")
    
    print("\n=> DANH SÁCH CÁC TENSOR TRỌNG SỐ TRONG FILE:")
    def visitor(name, node):
        if isinstance(node, h5py.Dataset):
            print(f"  Tensor: {name:<50} | shape={str(node.shape):<20} | dtype={node.dtype}")
    f.visititems(visitor)

# 2. Load model trực tiếp bằng tf.keras.models.load_model()
print("\n" + "=" * 70)
print("--- [2] LOAD THỰC TẾ BẰNG tf.keras.models.load_model() ---")
print("=" * 70)
try:
    model = load_model(WEIGHTS_PATH, compile=False)
    print("=> load_model() THÀNH CÔNG RỰC RỠ!\n")
    print("--- MODEL SUMMARY THẬT TỪ FILE .h5 ---")
    model.summary()
    print(f"\n=> TỔNG SỐ PARAMETERS THẬT: {model.count_params():,}")
    print(f"=> SỐ LAYERS: {len(model.layers)}")
    for idx, layer in enumerate(model.layers):
        cfg = layer.get_config()
        print(f"  [{idx+1}] {layer.name:<10} ({layer.__class__.__name__:<5}): units={cfg.get('units')}, activation={cfg.get('activation')}, return_sequences={cfg.get('return_sequences')}, output_shape={layer.output_shape}")
except Exception as e:
    print(f"=> load_model() báo lỗi: {e}")
    print("Thử load bằng Sequential + load_weights...")
    m1 = Sequential([
        LSTM(32, return_sequences=True, activation='relu', input_shape=(60, 126)),
        LSTM(128, return_sequences=True, activation='relu'),
        LSTM(64, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(60, activation='softmax')
    ])
    m1.load_weights(WEIGHTS_PATH)
    m1.summary()
