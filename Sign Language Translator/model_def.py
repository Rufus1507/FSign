# -*- coding: utf-8 -*-
"""
model_def.py — Định nghĩa kiến trúc mô hình LSTM chuẩn cho FSign (Chốt sau Hạng mục B)
========================================================================================
File này là Nguồn Sự Thật duy nhất (Single Source of Truth) cho kiến trúc mạng FSign.
Không chứa side-effect (không os.chdir, không import thư viện đồ họa hay camera).
Dùng chung cho cả training (train_model.py, ActionDetection.py) và inference (RunModel.py).

Kiến trúc chuẩn (Kết quả tối ưu từ Hạng mục B - Task B2 & Task B3):
  Input (60 frames, 129 landmark features: 126 normalized S_combined + 3 relative wrist)
  -> LSTM(128, return_sequences=True, activation='tanh')
  -> LSTM(64, return_sequences=True, activation='tanh')
  -> LSTM(32, return_sequences=False, activation='tanh')
  -> Dense(64, activation='relu')
  -> Dense(32, activation='relu')
  -> Dense(num_classes, activation='softmax')
  Optimizer: Adam(learning_rate=0.001, clipnorm=1.0)
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def build_model(input_shape=(60, 129), num_classes=60, lr=1e-3, clipnorm=1.0) -> Sequential:
    """
    Khởi tạo mô hình FSign LSTM chuẩn hóa (Kiến trúc giảm dần đều 128 -> 64 -> 32, Tanh).

    Args:
        input_shape: tuple kích thước chuỗi đầu vào (sequence_length, feature_dim), mặc định (60, 129).
        num_classes: số lượng nhãn đầu ra, mặc định 60 (tập Data_normalized/).
        lr: learning rate cho Adam optimizer, mặc định 0.001.
        clipnorm: giới hạn chuẩn gradient triệt tiêu gradient explosion, mặc định 1.0.

    Returns:
        tf.keras.models.Sequential: Mô hình đã được compile sẵn sàng cho train hoặc load weights.
    """
    model = Sequential([
        LSTM(128, return_sequences=True, activation='tanh', input_shape=input_shape, name='lstm_1'),
        LSTM(64, return_sequences=True, activation='tanh', name='lstm_2'),
        LSTM(32, return_sequences=False, activation='tanh', name='lstm_3'),
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
