# -*- coding: utf-8 -*-
"""
model_def_v3_presence.py — Định nghĩa kiến trúc mô hình FSign v3 với Presence Flag (Task E2)
=============================================================================================
Kiến trúc chuẩn từ Hạng mục B (128 -> 64 -> 32, Tanh, Clipnorm 1.0)
với đầu vào mở rộng 131 chiều:
  - 126 chiều: Tọa độ 2 bàn tay đã chuẩn hóa S_combined (63 LH + 63 RH)
  - 3 chiều: Vector tương đối 2 cổ tay (Wrist_RH - Wrist_LH)
  - 2 chiều: Presence Flag [LH_present, RH_present] (Task E2)
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def build_model(input_shape=(60, 131), num_classes=60, lr=1e-3, clipnorm=1.0) -> Sequential:
    """
    Khởi tạo mô hình FSign LSTM v3 với Presence Flag (Input shape: (60, 131)).
    """
    model = Sequential([
        LSTM(128, return_sequences=True, activation='tanh', input_shape=input_shape, name='lstm_1'),
        LSTM(64, return_sequences=True, activation='tanh', name='lstm_2'),
        LSTM(32, return_sequences=False, activation='tanh', name='lstm_3'),
        Dense(64, activation='relu', name='dense_1'),
        Dense(32, activation='relu', name='dense_2'),
        Dense(num_classes, activation='softmax', name='dense_out')
    ], name='FSign_v3_Presence_LSTM')

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    return model
