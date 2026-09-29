# -*- coding: utf-8 -*-
"""
model_def_merged119.py — Định nghĩa kiến trúc Baseline LSTM cho tập dữ liệu hợp nhất 119 nhãn
=============================================================================================
Dùng ĐÚNG kiến trúc đã chốt ở Hạng mục B (Task B2 & B3):
  Input (60 frames, 129 landmark features: 126 normalized S_combined + 3 relative wrist)
  -> LSTM(128, return_sequences=True, activation='tanh')
  -> LSTM(64, return_sequences=True, activation='tanh')
  -> LSTM(32, return_sequences=False, activation='tanh')
  -> Dense(64, activation='relu')
  -> Dense(32, activation='relu')
  -> Dense(119, activation='softmax')
  Optimizer: Adam(learning_rate=0.001, clipnorm=1.0)

TUYỆT ĐỐI KHÔNG thêm Bi-LSTM, KHÔNG thêm Attention, KHÔNG thêm BatchNorm/Dropout
theo đúng yêu cầu Baseline Apples-to-Apples của Task MERGE-5.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def build_model(input_shape=(60, 129), num_classes=119, lr=1e-3, clipnorm=1.0) -> Sequential:
    """
    Khởi tạo mô hình FSign Baseline LSTM 119 nhãn (128 -> 64 -> 32, Tanh, clipnorm=1.0).

    Args:
        input_shape: tuple kích thước chuỗi đầu vào (sequence_length, feature_dim), mặc định (60, 129).
        num_classes: số lượng nhãn đầu ra, mặc định 119 (tập Data_normalized_merged119/).
        lr: learning rate cho Adam optimizer, mặc định 0.001.
        clipnorm: giới hạn chuẩn gradient triệt tiêu gradient explosion, mặc định 1.0.

    Returns:
        tf.keras.models.Sequential: Mô hình đã được compile sẵn sàng cho training.
    """
    model = Sequential([
        LSTM(128, return_sequences=True, activation='tanh', input_shape=input_shape, name='lstm_1'),
        LSTM(64, return_sequences=True, activation='tanh', name='lstm_2'),
        LSTM(32, return_sequences=False, activation='tanh', name='lstm_3'),
        Dense(64, activation='relu', name='dense_1'),
        Dense(32, activation='relu', name='dense_2'),
        Dense(num_classes, activation='softmax', name='dense_out')
    ], name='FSign_Baseline_119_LSTM')

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'categorical_accuracy',
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_acc'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_acc')
        ]
    )
    return model


if __name__ == '__main__':
    m = build_model()
    m.summary()
