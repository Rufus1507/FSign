# -*- coding: utf-8 -*-
"""
archive/model_def_relu_baseline.py — Bản lưu trữ kiến trúc Baseline cũ (ReLU 32 -> 128 -> 64)
=============================================================================================
Lưu trữ phiên bản gốc trước khi tối ưu hóa ở Hạng mục B (đã được thay thế bởi kiến trúc
giảm dần đều 128 -> 64 -> 32 với Tanh trong Task B3).
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def build_model(input_shape=(60, 129), num_classes=60, lr=1e-3, clipnorm=1.0) -> Sequential:
    model = Sequential([
        LSTM(32, return_sequences=True, activation='relu', input_shape=input_shape, name='lstm_1'),
        LSTM(128, return_sequences=True, activation='relu', name='lstm_2'),
        LSTM(64, return_sequences=False, activation='relu', name='lstm_3'),
        Dense(64, activation='relu', name='dense_1'),
        Dense(32, activation='relu', name='dense_2'),
        Dense(num_classes, activation='softmax', name='dense_out')
    ], name='FSign_Unified_LSTM_ReLU_Legacy')

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    return model
