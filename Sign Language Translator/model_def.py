# -*- coding: utf-8 -*-
"""
model_def.py — Định nghĩa kiến trúc mô hình LSTM chuẩn cho FSign
================================================================
File này là Nguồn Sự Thật duy nhất (Single Source of Truth) cho kiến trúc mạng FSign.
Không chứa side-effect (không os.chdir, không import thư viện đồ họa hay camera).
Dùng chung cho cả training (train_model.py, ActionDetection.py) và inference (RunModel.py).

Kiến trúc:
  Input (60 frames, 126 landmark features)
  -> LSTM(32, return_sequences=True, activation='relu')
  -> LSTM(128, return_sequences=True, activation='relu')
  -> LSTM(64, return_sequences=False, activation='relu')
  -> Dense(64, activation='relu')
  -> Dense(32, activation='relu')
  -> Dense(num_classes, activation='softmax')
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def build_model(input_shape=(60, 126), num_classes=61, lr=1e-3, clipnorm=1.0) -> Sequential:
    """
    Khởi tạo mô hình FSign LSTM chuẩn hóa.

    Args:
        input_shape: tuple kích thước chuỗi đầu vào (sequence_length, feature_dim), mặc định (60, 126).
        num_classes: số lượng nhãn đầu ra, mặc định 61 (tập Data_normalized/).
        lr: learning rate cho Adam optimizer, mặc định 0.001.
        clipnorm: giới hạn chuẩn gradient để triệt tiêu gradient explosion với ReLU, mặc định 1.0.

    Returns:
        tf.keras.models.Sequential: Mô hình đã được compile sẵn sàng cho train hoặc load weights.
    """
    model = Sequential([
        LSTM(32, return_sequences=True, activation='relu', input_shape=input_shape, name='lstm_1'),
        LSTM(128, return_sequences=True, activation='relu', name='lstm_2'),
        LSTM(64, return_sequences=False, activation='relu', name='lstm_3'),
        Dense(64, activation='relu', name='dense_1'),
        Dense(32, activation='relu', name='dense_2'),
        Dense(num_classes, activation='softmax', name='dense_out')
    ], name='FSign_Unified_LSTM')

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    return model
