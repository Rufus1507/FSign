# -*- coding: utf-8 -*-
"""
verify_run_and_backup.py — Xác nhận độc lập tình trạng backup và đánh giá cả 2 model
=====================================================================================
1. Đọc thông tin file, kích thước, thời gian sửa đổi của:
   - Models/model_normalized_v1.h5 (Run 2 vừa train)
   - Models/model_normalized_v1_backup_run1.h5 (Run 1 trước đó)
2. Load độc lập từng file và đánh giá trực tiếp trên Holdout Test Set.
"""
import os
import sys
import time
from datetime import datetime

# Fix Windows console UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from model_def import build_model
from train_model import fast_load_dataset_normalized

print("=" * 80)
print("=== [1] XÁC NHẬN TÌNH TRẠNG FILE CHECKPOINT VÀ BACKUP TRÊN ĐĨA ===")
print("=" * 80)

m_curr = os.path.join('Models', 'model_normalized_v1.h5')
m_back = os.path.join('Models', 'model_normalized_v1_backup_run1.h5')

for path, label in [(m_back, "Run 1 (Backup)"), (m_curr, "Run 2 (Hiện tại)")]:
    if os.path.exists(path):
        sz = os.path.getsize(path)
        mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{label}]")
        print(f"  Đường dẫn:   {path}")
        print(f"  Kích thước:  {sz:,} bytes")
        print(f"  Thời gian:   {mtime}")
    else:
        print(f"[{label}] KHÔNG TÌM THẤY: {path}")
print()

print("=" * 80)
print("=== [2] ĐÁNH GIÁ ĐỘC LẬP CẢ 2 CHECKPOINT TRÊN HOLDOUT TEST SET ===")
print("=" * 80)

X, y, label_map = fast_load_dataset_normalized()
num_classes = len(label_map)
y_int = np.argmax(y, axis=1)

unique, counts = np.unique(y_int, return_counts=True)
rare_classes = set(unique[counts < 2])
if rare_classes:
    regular_mask = ~np.isin(y_int, list(rare_classes))
    X_reg, y_reg = X[regular_mask], y[regular_mask]
    X_train_reg, X_temp, y_train_reg, y_temp = train_test_split(
        X_reg, y_reg, test_size=0.30, random_state=42, stratify=np.argmax(y_reg, axis=1)
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=np.argmax(y_temp, axis=1)
    )
else:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y_int
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=np.argmax(y_temp, axis=1)
    )

print(f"=> Holdout Test Set: {X_test.shape[0]} samples\n")

eval_model = build_model(num_classes=num_classes)

# Đánh giá Run 1 Backup
if os.path.exists(m_back):
    eval_model.load_weights(m_back)
    loss1, acc1 = eval_model.evaluate(X_test, y_test, verbose=0)
    print(f">> RUN 1 (Backup: model_normalized_v1_backup_run1.h5):")
    print(f"   Test Loss:     {loss1:.4f}")
    print(f"   Test Accuracy: {acc1 * 100:.2f}% ({int(round(acc1 * len(y_test)))}/{len(y_test)} samples)\n")

# Đánh giá Run 2 Current
if os.path.exists(m_curr):
    eval_model.load_weights(m_curr)
    loss2, acc2 = eval_model.evaluate(X_test, y_test, verbose=0)
    print(f">> RUN 2 (Hiện tại: model_normalized_v1.h5):")
    print(f"   Test Loss:     {loss2:.4f}")
    print(f"   Test Accuracy: {acc2 * 100:.2f}% ({int(round(acc2 * len(y_test)))}/{len(y_test)} samples)\n")

print("=" * 80)
