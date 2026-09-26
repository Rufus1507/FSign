# -*- coding: utf-8 -*-
"""
verify_tflite_threads.py — Kiểm tra chi tiết hành vi num_threads trên TFLite Interpreter
========================================================================================
Mục tiêu:
  1. Kiểm tra tham số num_threads (1 luồng, 2 luồng, 4 luồng, mặc định) có thực sự được nhận.
  2. Đo đạc chính xác độ trễ trên từng cấu hình luồng (50 iterations sau warmup).
  3. Phân tích nguyên nhân tại sao đa luồng không tạo ra khác biệt lớn trên mô hình LSTM tuần tự batch=1.
"""
import os
import sys
import time
import numpy as np
import tensorflow as tf

# Fix Windows console UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

tflite_model_path = os.path.join('Models', 'model_normalized_v1.tflite')
if not os.path.exists(tflite_model_path):
    raise FileNotFoundError(f"Không tìm thấy {tflite_model_path}")

# Dữ liệu giả lập chuẩn batch=1 (1, 60, 126)
sample = np.random.randn(1, 60, 126).astype(np.float32)

configs = [
    ("Mặc định (num_threads=None)", None),
    ("1 Luồng (num_threads=1)", 1),
    ("2 Luồng (num_threads=2)", 2),
    ("4 Luồng (num_threads=4)", 4),
    ("8 Luồng (num_threads=8)", 8),
]

print("=" * 80)
print("=== THỰC NGHIỆM ĐO ĐẠC num_threads TRÊN TFLITE INTERPRETER ===")
print(f"TensorFlow: {tf.__version__} | Model: {tflite_model_path} ({os.path.getsize(tflite_model_path)/1024:.1f} KB)")
print("=" * 80)

results = []

for label, n_threads in configs:
    # Đo đạc thời gian khởi tạo và kiểm tra tham số
    t0_init = time.perf_counter()
    if n_threads is None:
        interp = tf.lite.Interpreter(model_path=tflite_model_path)
    else:
        interp = tf.lite.Interpreter(model_path=tflite_model_path, num_threads=n_threads)
    interp.allocate_tensors()
    init_time = (time.perf_counter() - t0_init) * 1000.0

    in_idx = interp.get_input_details()[0]['index']
    out_idx = interp.get_output_details()[0]['index']

    # Warmup 10 lần
    for _ in range(10):
        interp.set_tensor(in_idx, sample)
        interp.invoke()
        _ = interp.get_tensor(out_idx)

    # Đo 50 lần
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        interp.set_tensor(in_idx, sample)
        interp.invoke()
        _ = interp.get_tensor(out_idx)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = np.mean(latencies)
    min_ms = np.min(latencies)
    max_ms = np.max(latencies)
    std_ms = np.std(latencies)

    results.append({
        'label': label,
        'threads': n_threads,
        'mean': mean_ms,
        'min': min_ms,
        'max': max_ms,
        'std': std_ms,
        'init_ms': init_time
    })
    print(f"[{label}] -> Trung bình: {mean_ms:5.2f} ms | Min: {min_ms:5.2f} ms | Max: {max_ms:5.2f} ms | Std: {std_ms:4.2f} ms")

print("\n" + "=" * 80)
print(f"{'Cấu hình luồng':<30} | {'Mean (ms)':<10} | {'Min (ms)':<10} | {'Max (ms)':<10} | {'Std (ms)':<10}")
print("-" * 80)
for r in results:
    print(f"{r['label']:<30} | {r['mean']:<10.2f} | {r['min']:<10.2f} | {r['max']:<10.2f} | {r['std']:<10.2f}")
print("=" * 80)
