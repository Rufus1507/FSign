# -*- coding: utf-8 -*-
"""
train_fsign159_optimized.py
===========================
Huấn luyện mô hình FSign 159 classes phiên bản TỐI ƯU HÓA:
- Sử dụng Gradient Clipping (clipnorm=1.0) triệt tiêu hoàn toàn bùng nổ gradient
- Chuẩn hóa LSTM (activation='tanh', recurrent_activation='sigmoid')
- Thêm BatchNormalization và Dropout(0.3) chống Overfitting
- Tối ưu hóa độ chính xác kỳ vọng: > 85% - 95%
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

# Fix Unicode UTF-8 output cho console Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score

import keras
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, Callback
from keras.utils import to_categorical

SEQUENCE_LENGTH = 60
FEATURE_DIM = 126


class EpochLoggingCallback(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_records = []
        self.epoch_start_time = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        dur = time.time() - self.epoch_start_time
        logs = logs or {}
        rec = {
            'epoch': epoch + 1,
            'loss': logs.get('loss', 0.0),
            'acc': logs.get('categorical_accuracy', 0.0),
            'val_loss': logs.get('val_loss', 0.0),
            'val_acc': logs.get('val_categorical_accuracy', 0.0),
            'lr': float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate)) if hasattr(self.model.optimizer, 'learning_rate') else 0.001,
            'duration_s': dur
        }
        self.epoch_records.append(rec)


def load_all_data(data_dir):
    data_path = Path(data_dir)
    labels = sorted([d.name for d in data_path.iterdir() if d.is_dir() and d.name != 'cam on'])
    num_classes = len(labels)
    label_to_id = {lbl: i for i, lbl in enumerate(labels)}
    id_to_label = {i: lbl for i, lbl in enumerate(labels)}

    print(f"=> Bắt đầu nạp dữ liệu từ {num_classes} nhãn...")

    sequences = []
    targets = []
    skipped_seqs = 0

    t0 = time.time()
    for lbl_idx, label_name in enumerate(labels):
        label_folder = data_path / label_name
        seq_folders = [d for d in label_folder.iterdir() if d.is_dir()]

        for seq_folder in seq_folders:
            frames = []
            is_valid = True
            for frame_idx in range(SEQUENCE_LENGTH):
                frame_file = seq_folder / f"{frame_idx}.npy"
                if not frame_file.exists():
                    is_valid = False
                    break
                try:
                    arr = np.load(str(frame_file))
                    if arr.shape != (FEATURE_DIM,):
                        is_valid = False
                        break
                    frames.append(arr)
                except Exception:
                    is_valid = False
                    break

            if is_valid and len(frames) == SEQUENCE_LENGTH:
                sequences.append(frames)
                targets.append(lbl_idx)
            else:
                skipped_seqs += 1

    X = np.array(sequences, dtype=np.float32)
    y = to_categorical(targets, num_classes=num_classes)

    print(f"=> Nạp xong trong {time.time() - t0:.1f}s!")
    print(f"   X shape: {X.shape} | y shape: {y.shape} (Tổng sequences: {len(X):,})")

    return X, y, labels, label_to_id, id_to_label


def build_optimized_model(input_shape=(SEQUENCE_LENGTH, FEATURE_DIM), num_classes=159):
    """
    Kiến trúc Deep LSTM chuẩn mực có Gradient Clipping & Batch Normalization
    """
    model = Sequential([
        Input(shape=input_shape, name="Input_60x126"),
        BatchNormalization(name="BatchNorm_Input"),
        
        # Tầng LSTM 1
        LSTM(64, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', name="LSTM_1_64"),
        BatchNormalization(name="BatchNorm_1"),
        Dropout(0.2, name="Dropout_1"),
        
        # Tầng LSTM 2
        LSTM(128, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', name="LSTM_2_128"),
        BatchNormalization(name="BatchNorm_2"),
        Dropout(0.2, name="Dropout_2"),
        
        # Tầng LSTM 3
        LSTM(64, return_sequences=False, activation='tanh', recurrent_activation='sigmoid', name="LSTM_3_64"),
        BatchNormalization(name="BatchNorm_3"),
        
        # Dense Layers
        Dense(128, activation='relu', name="Dense_128"),
        Dropout(0.3, name="Dropout_3"),
        Dense(64, activation='relu', name="Dense_64"),
        Dense(num_classes, activation='softmax', name="Softmax_159")
    ], name="FSign_159_Optimized")

    # Optimizer với clipnorm=1.0 chống hoàn toàn bùng nổ gradient
    optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình FSign-159 Tối Ưu")
    parser.add_argument('--epochs', type=int, default=100, help="Số epochs huấn luyện (mặc định: 100)")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (mặc định: 64)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    data_dir = script_dir / 'Data'
    release_dir = script_dir / 'release'
    release_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = script_dir / 'Logs' / 'fsign159_opt'
    logs_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = release_dir / 'fsign_159classes.h5'
    label_map_path = release_dir / 'label_map.json'

    print("=" * 65)
    print("  FSIGN - HUẤN LUYỆN MÔ HÌNH TỐI ƯU HÓA (GRADIENT CLIPPING & TANH)")
    print("=" * 65)

    X, y, labels, label_to_id, id_to_label = load_all_data(data_dir)
    num_classes = len(labels)

    # Chia tập train/val 85/15
    targets_indices = np.argmax(y, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.15,
        random_state=42,
        stratify=targets_indices
    )
    print(f"=> Phân chia tập dữ liệu: Train = {X_train.shape[0]:,} mẫu | Validation = {X_test.shape[0]:,} mẫu")

    model = build_optimized_model(input_shape=(SEQUENCE_LENGTH, FEATURE_DIM), num_classes=num_classes)
    model.summary()

    logging_cb = EpochLoggingCallback()
    callbacks = [
        logging_cb,
        ModelCheckpoint(
            filepath=str(model_save_path),
            monitor='val_categorical_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_categorical_accuracy',
            patience=25,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1
        ),
        TensorBoard(log_dir=str(logs_dir))
    ]

    print(f"\n=> Bắt đầu huấn luyện mô hình ({args.epochs} epochs)...")
    t_start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    dur = time.time() - t_start

    print(f"\n=> Huấn luyện hoàn tất trong {int(dur//60)}m {int(dur%60)}s!")
    val_loss, val_acc = model.evaluate(X_test, y_test, verbose=0)
    print("=" * 65)
    print(f"KẾT QUẢ CUỐI CÙNG:")
    print(f"  - Validation Loss:     {val_loss:.4f}")
    print(f"  - Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"  - Model lưu tại:       {model_save_path}")
    print("=" * 65)


if __name__ == '__main__':
    main()
