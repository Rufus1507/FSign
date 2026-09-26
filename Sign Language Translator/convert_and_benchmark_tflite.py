# -*- coding: utf-8 -*-
"""
convert_and_benchmark_tflite.py — Xuất mô hình sang TFLite & Benchmark toàn diện (Mục C2)
========================================================================================
Nhiệm vụ:
  1. Convert Models/model_normalized_v1.h5 sang TFLite (Models/model_normalized_v1.tflite).
  2. Kiểm tra xem có bị dính Flex ops / Select TF Ops hay dùng thuần TFLite Builtin ops.
  3. Đánh giá Accuracy trên đúng 541 mẫu holdout test set (so sánh đối đầu trực diện H5 vs TFLite).
  4. Đo latency suy luận thực tế (50 lần chạy trên CPU) để xác định tốc độ tăng tốc thật.
"""
import os
import sys
import time

# Fix Windows console UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from model_def import build_model
from RunModel import load_actions

# Chuẩn bị file log ghi nhận toàn bộ kết quả C2
log_file_path = "benchmark_c2_tflite.log"
log_file = open(log_file_path, "w", encoding="utf-8")

def log_print(msg=""):
    print(msg)
    log_file.write(str(msg) + "\n")
    log_file.flush()

log_print("=" * 88)
log_print("=== [MỤC C2] XUẤT MÔ HÌNH SANG TFLITE & ĐÁNH GIÁ ACCURACY / LATENCY THỰC TẾ ===")
log_print(f"TensorFlow Version: {tf.__version__}")
log_print(f"Thời gian bắt đầu:   {time.strftime('%Y-%m-%d %H:%M:%S')}")
log_print("=" * 88)

# ─── BƯỚC 1: NẠP MODEL H5 VÀ CONVERT SANG TFLITE ────────────────────────────
h5_model_path = os.path.join('Models', 'model_normalized_v1.h5')
tflite_model_path = os.path.join('Models', 'model_normalized_v1.tflite')

actions = load_actions()
num_classes = len(actions)
log_print(f"\n[1/4] Nạp mô hình Keras gốc từ: {h5_model_path} ({num_classes} nhãn)...")
keras_model = build_model(input_shape=(60, 126), num_classes=num_classes)
keras_model.load_weights(h5_model_path)
h5_size_mb = os.path.getsize(h5_model_path) / (1024 * 1024)
log_print(f"=> Nạp thành công! Kích thước file .h5: {h5_size_mb:.2f} MB")

log_print("\n[2/4] Tiến hành Convert sang TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

# Cấu hình kiểm tra Builtin Ops: Yêu cầu chuẩn TFLite Builtin
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

t0_conv = time.perf_counter()
pure_builtin = False
try:
    log_print("-> Đang convert với target_spec = [TFLITE_BUILTINS] (Float32 tiêu chuẩn)...")
    tflite_model_content = converter.convert()
    convert_time = time.perf_counter() - t0_conv
    log_print(f"=> Convert THÀNH CÔNG trong {convert_time:.2f}s với TFLITE_BUILTINS thuần túy!")
    pure_builtin = True
except Exception as e:
    log_print(f"[CẢNH BÁO] Lỗi khi ép dùng thuần TFLITE_BUILTINS: {e}")
    log_print("=> Thử nghiệm fallback với Select TF Ops (Flex Delegate)...")
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    t0_conv = time.perf_counter()
    tflite_model_content = converter.convert()
    convert_time = time.perf_counter() - t0_conv
    log_print(f"=> Convert hoàn thành với Select TF Ops (Flex) trong {convert_time:.2f}s!")
    pure_builtin = False

# Lưu file .tflite
os.makedirs(os.path.dirname(tflite_model_path), exist_ok=True)
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model_content)

tflite_size_mb = os.path.getsize(tflite_model_path) / (1024 * 1024)
log_print(f"=> Đã lưu file TFLite tại: {tflite_model_path}")
log_print(f"   Dung lượng .h5:     {h5_size_mb:.2f} MB")
log_print(f"   Dung lượng .tflite: {tflite_size_mb:.2f} MB (Chênh lệch: {(tflite_size_mb - h5_size_mb):+.2f} MB)")

# Kiểm tra sâu các Operator và Tensor bên trong file TFLite
log_print("\n=> Kiểm tra cấu trúc Tensors và Operators bên trong mô hình TFLite:")
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
tensor_details = interpreter.get_tensor_details()

log_print(f"   Input shape:         {input_details[0]['shape']} | Type: {input_details[0]['dtype']}")
log_print(f"   Output shape:        {output_details[0]['shape']} | Type: {output_details[0]['dtype']}")
log_print(f"   Tổng số tensors:     {len(tensor_details)}")

# Quét tìm Flex ops trong tensor names
flex_tensors = [t['name'] for t in tensor_details if 'flex' in t['name'].lower() or 'select' in t['name'].lower()]
log_print(f"   Số tensors dính Flex ops: {len(flex_tensors)}")
if flex_tensors:
    log_print(f"   [CẢNH BÁO] Danh sách Flex tensors: {flex_tensors}")

# Liệt kê các Ops trong mô hình
ops_used = set()
if hasattr(interpreter, '_get_ops_details'):
    try:
        for op in interpreter._get_ops_details():
            ops_used.add(op.get('op_name', 'UNKNOWN'))
    except Exception:
        pass

flex_ops = [op for op in ops_used if 'flex' in op.lower() or 'select' in op.lower()]
log_print(f"   Tổng số distinct operator types: {len(ops_used)}")
if ops_used:
    log_print(f"   Danh sách Ops sử dụng: {sorted(list(ops_used))}")

if len(flex_ops) == 0 and len(flex_tensors) == 0 and pure_builtin:
    log_print("   [XÁC NHẬN] Mô hình 100% sử dụng THUẦN TFLite Builtin Ops! KHÔNG BỊ RƠI VÀO FLEX DELEGATE!")
else:
    log_print(f"   [CẢNH BÁO] Phát hiện Flex Ops hoặc Flex Tensors!")

# ─── BƯỚC 2: XÁC MINH ACCURACY TRÊN 541 MẪU HOLDOUT TEST SET ────────────────
log_print("\n" + "=" * 88)
log_print("[3/4] XÁC MINH ACCURACY TRÊN TẬP HOLDOUT TEST SET (541 MẪU)")
log_print("=" * 88)

cache_path = os.path.join('Data_normalized', 'dataset_cache.npz')
if not os.path.exists(cache_path):
    raise FileNotFoundError(f"Không tìm thấy file cache tại {cache_path}")

log_print(f"=> Nạp dataset từ: {cache_path}")
c = np.load(cache_path, allow_pickle=True)
X_all = c['X']
y_all = c['y']
log_print(f"=> Tổng dataset: {X_all.shape[0]} mẫu ({len(actions)} nhãn)")

# Tái lập chính xác tập phân chia Train / Val / Test (Holdout 15%) giống Phase A
y_int = np.argmax(y_all, axis=1)
unique, counts = np.unique(y_int, return_counts=True)
rare_classes = set(unique[counts < 2])

if rare_classes:
    regular_mask = ~np.isin(y_int, list(rare_classes))
    X_reg, y_reg = X_all[regular_mask], y_all[regular_mask]
    
    X_train_reg, X_temp, y_train_reg, y_temp = train_test_split(
        X_reg, y_reg, test_size=0.30, random_state=42, stratify=np.argmax(y_reg, axis=1)
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=np.argmax(y_temp, axis=1)
    )
else:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.30, random_state=42, stratify=y_int
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=np.argmax(y_temp, axis=1)
    )

log_print(f"=> Số lượng mẫu tập Holdout Test: {X_test.shape[0]} mẫu (Đúng chuẩn 541 mẫu)")
assert X_test.shape[0] == 541, f"Lỗi: Số mẫu tập test không khớp 541 (thực tế: {X_test.shape[0]})"
y_test_indices = np.argmax(y_test, axis=1)

# 1. Chạy đánh giá Keras .h5
log_print("\n--- [1] Đánh giá mô hình gốc Keras (.h5) ---")
t0_h5_eval = time.perf_counter()
preds_h5 = keras_model.predict(X_test, batch_size=32, verbose=0)
time_h5_eval = time.perf_counter() - t0_h5_eval
pred_indices_h5 = np.argmax(preds_h5, axis=1)
acc_h5 = accuracy_score(y_test_indices, pred_indices_h5)
correct_h5 = np.sum(pred_indices_h5 == y_test_indices)
log_print(f"  Accuracy .h5:       {acc_h5 * 100:.2f}% ({correct_h5}/{len(y_test_indices)} mẫu đúng)")
log_print(f"  Thời gian test h5:  {time_h5_eval:.2f}s")

# 2. Chạy đánh giá TFLite
log_print("\n--- [2] Đánh giá mô hình TFLite (.tflite) ---")
preds_tflite = np.zeros((len(X_test), num_classes), dtype=np.float32)
in_idx = input_details[0]['index']
out_idx = output_details[0]['index']

t0_tflite_eval = time.perf_counter()
for i in range(len(X_test)):
    sample = np.expand_dims(X_test[i], axis=0).astype(np.float32)
    interpreter.set_tensor(in_idx, sample)
    interpreter.invoke()
    preds_tflite[i] = interpreter.get_tensor(out_idx)[0]
time_tflite_eval = time.perf_counter() - t0_tflite_eval

pred_indices_tflite = np.argmax(preds_tflite, axis=1)
acc_tflite = accuracy_score(y_test_indices, pred_indices_tflite)
correct_tflite = np.sum(pred_indices_tflite == y_test_indices)
log_print(f"  Accuracy .tflite:   {acc_tflite * 100:.2f}% ({correct_tflite}/{len(y_test_indices)} mẫu đúng)")
log_print(f"  Thời gian test:     {time_tflite_eval:.2f}s")

# So sánh độ tương đồng xác suất giữa H5 và TFLite
max_diff = np.max(np.abs(preds_h5 - preds_tflite))
mean_diff = np.mean(np.abs(preds_h5 - preds_tflite))
matching_predictions = np.sum(pred_indices_h5 == pred_indices_tflite)
match_rate = matching_predictions / len(y_test_indices) * 100

log_print(f"\n=> ĐỐI CHỨNG ĐỘ SAI LỆCH SỐ HỌC GIỮA .H5 VÀ .TFLITE:")
log_print(f"   Số nhãn trùng khớp 100% giữa H5 và TFLite: {matching_predictions}/{len(y_test_indices)} ({match_rate:.2f}%)")
log_print(f"   Độ lệch xác suất lớn nhất (Max Diff):     {max_diff:.8f}")
log_print(f"   Độ lệch xác suất trung bình (Mean Diff):   {mean_diff:.8f}")
log_print(f"   Chênh lệch Accuracy (TFLite - H5):        {(acc_tflite - acc_h5)*100:+.2f}%")

# ─── BƯỚC 3: BENCHMARK TỐC ĐỘ THỰC TẾ (SINGLE INFERENCE BATCH=1 TRÊN CPU) ───
log_print("\n" + "=" * 88)
log_print("[4/4] BENCHMARK ĐỘ TRỄ SUY LUẬN ĐƠN (BATCH=1) TRÊN CPU NÀY (50 ITERATIONS)")
log_print("=" * 88)

sample_single = np.expand_dims(X_test[0], axis=0).astype(np.float32)

# Warmup 10 lần
for _ in range(10):
    _ = keras_model.predict(sample_single, verbose=0)
    interpreter.set_tensor(in_idx, sample_single)
    interpreter.invoke()
    _ = interpreter.get_tensor(out_idx)

# 1. Đo độ trễ Keras .h5
h5_latencies = []
for _ in range(50):
    t0 = time.perf_counter()
    _ = keras_model.predict(sample_single, verbose=0)
    h5_latencies.append((time.perf_counter() - t0) * 1000.0)

mean_h5_ms = np.mean(h5_latencies)
min_h5_ms = np.min(h5_latencies)
max_h5_ms = np.max(h5_latencies)
std_h5_ms = np.std(h5_latencies)

# 2. Đo độ trễ TFLite (default threads / XNNPACK)
tflite_latencies = []
for _ in range(50):
    t0 = time.perf_counter()
    interpreter.set_tensor(in_idx, sample_single)
    interpreter.invoke()
    _ = interpreter.get_tensor(out_idx)
    tflite_latencies.append((time.perf_counter() - t0) * 1000.0)

mean_tflite_ms = np.mean(tflite_latencies)
min_tflite_ms = np.min(tflite_latencies)
max_tflite_ms = np.max(tflite_latencies)
std_tflite_ms = np.std(tflite_latencies)

# 3. Đo độ trễ TFLite với num_threads=4
interpreter_4t = tf.lite.Interpreter(model_path=tflite_model_path, num_threads=4)
interpreter_4t.allocate_tensors()
tflite_4t_latencies = []
for _ in range(50):
    t0 = time.perf_counter()
    interpreter_4t.set_tensor(in_idx, sample_single)
    interpreter_4t.invoke()
    _ = interpreter_4t.get_tensor(out_idx)
    tflite_4t_latencies.append((time.perf_counter() - t0) * 1000.0)

mean_tflite_4t_ms = np.mean(tflite_4t_latencies)

speedup = mean_h5_ms / mean_tflite_ms if mean_tflite_ms > 0 else 0

log_print(f"\n{'Chỉ số đo đạc':<32} | {'Keras .h5 (Baseline)':<22} | {'TFLite (CPU/XNNPACK)':<22}")
log_print("-" * 80)
log_print(f"{'Thời gian trung bình (Mean)':<32} | {mean_h5_ms:6.2f} ms              | {mean_tflite_ms:6.2f} ms")
log_print(f"{'Thời gian nhỏ nhất (Min)':<32} | {min_h5_ms:6.2f} ms              | {min_tflite_ms:6.2f} ms")
log_print(f"{'Thời gian lớn nhất (Max)':<32} | {max_h5_ms:6.2f} ms              | {max_tflite_ms:6.2f} ms")
log_print(f"{'Độ lệch chuẩn (Std Dev)':<32} | {std_h5_ms:6.2f} ms              | {std_tflite_ms:6.2f} ms")
log_print(f"{'TFLite (4 Threads)':<32} | {'-':<22} | {mean_tflite_4t_ms:6.2f} ms")
log_print(f"{'TỐC ĐỘ TĂNG TỐC (Speedup)':<32} | {'1.00x':<22} | {speedup:6.2f}x NHANH HƠN")
log_print("-" * 80)

log_print("\n" + "=" * 88)
log_print("=== TỔNG KẾT SO SÁNH CUỐI CÙNG: KERAS .H5 VS TFLITE ===")
log_print("=" * 88)
log_print(f"1. Kích thước file: .h5 ({h5_size_mb:.2f} MB) -> .tflite ({tflite_size_mb:.2f} MB)")
log_print(f"2. Loại Ops:        {'Thuần TFLite Builtin (KHÔNG Flex Ops)' if pure_builtin else 'Có Flex Ops'}")
log_print(f"3. Accuracy Test:   .h5: {acc_h5*100:.2f}%  vs  .tflite: {acc_tflite*100:.2f}%  (Khớp {match_rate:.2f}%)")
log_print(f"4. Latency Predict: .h5: {mean_h5_ms:.2f} ms  ->  .tflite: {mean_tflite_ms:.2f} ms  (Tăng tốc {speedup:.2f}x)")
log_print(f"5. Tác động tới FPS trần: Từ ~10.1 FPS lên ~{1000.0/(55.0 + mean_tflite_ms):.1f} FPS (chưa cần multi-threading)")
log_print(f"6. Chi tiết log đã lưu tại: {log_file_path}")
log_print("=" * 88)

log_file.close()
