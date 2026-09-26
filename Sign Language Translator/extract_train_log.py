# -*- coding: utf-8 -*-
"""
extract_train_log.py — Đọc toàn bộ log thật từng epoch từ TensorBoard event file
"""
import os
import sys

# Fix Windows console UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import tensorflow as tf

log_dir = os.path.join('Logs', 'train_normalized_20260923-092744')
train_dir = os.path.join(log_dir, 'train')
val_dir = os.path.join(log_dir, 'validation')

train_events = [f for f in os.listdir(train_dir) if f.startswith('events.out.tfevents')]
val_events = [f for f in os.listdir(val_dir) if f.startswith('events.out.tfevents')]

if not train_events or not val_events:
    print("Không tìm thấy file events!")
    sys.exit(1)

train_file = os.path.join(train_dir, train_events[0])
val_file = os.path.join(val_dir, val_events[0])

def read_scalars(filepath):
    scalars = {}
    for event in tf.compat.v1.train.summary_iterator(filepath):
        for v in event.summary.value:
            if v.tag not in scalars:
                scalars[v.tag] = []
            scalars[v.tag].append((event.step, v.simple_value))
    return scalars

train_scalars = read_scalars(train_file)
val_scalars = read_scalars(val_file)

print("=" * 85)
print(f"=== LOG CHI TIẾT NGUYÊN VĂN TỪNG EPOCH TỪ RUN 20260923-092744 ===")
print(f"Log directory: {log_dir}")
print("=" * 85)

steps = len(train_scalars.get('epoch_loss', []))
print(f"{'Epoch':<8} | {'train_loss':<12} | {'train_acc':<12} | {'val_loss':<12} | {'val_acc':<12}")
print("-" * 85)

for i in range(steps):
    t_loss = train_scalars.get('epoch_loss', [])[i][1] if i < len(train_scalars.get('epoch_loss', [])) else 0
    t_acc = train_scalars.get('epoch_categorical_accuracy', [])[i][1] if i < len(train_scalars.get('epoch_categorical_accuracy', [])) else 0
    v_loss = val_scalars.get('epoch_loss', [])[i][1] if i < len(val_scalars.get('epoch_loss', [])) else 0
    v_acc = val_scalars.get('epoch_categorical_accuracy', [])[i][1] if i < len(val_scalars.get('epoch_categorical_accuracy', [])) else 0
    print(f"Epoch {i+1:<3} | loss: {t_loss:<7.4f} | acc: {t_acc*100:<6.2f}% | val_loss: {v_loss:<7.4f} | val_acc: {v_acc*100:<6.2f}%")

print("=" * 85)
