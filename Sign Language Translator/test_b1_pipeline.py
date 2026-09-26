# -*- coding: utf-8 -*-
"""
test_b1_pipeline.py — Test độc lập cho Mục B1 (Không dùng webcam)
===================================================================
Mục tiêu kiểm tra:
  [Test 1] Danh sách nhãn actions: Đảm bảo nạp đúng 61 nhãn chuẩn từ dataset_cache.npz.
  [Test 2] Hàm normalize_keypoints: Kiểm tra tính đúng đắn khi đưa cổ tay về gốc (0,0,0)
           và co giãn bất biến tỷ lệ.
  [Test 3] Model & Weights: Load Models/model_normalized_v1.h5 qua build_model(model_def.py).
  [Test 4] Inference thực tế: Load 61 sequence thật (1 mẫu từ mỗi nhãn trong Data_normalized/)
           và tính Accuracy dự đoán thực tế.
  [Test 5] Mô phỏng Buffer Frame-by-frame: Mô phỏng cơ chế sliding window 60 frame
           như trong vòng lặp thời gian thực của RunModel.py.
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
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from model_def import build_model
from preprocessing import normalize_keypoints
from RunModel import load_actions, TFLiteModelWrapper

print("=" * 80)
print("=== [TEST B1.1] KIỂM TRA DANH SÁCH 61 NHÃN TỪ DATASET CHUẨN ===")
print("=" * 80)
actions = load_actions()
print(f"=> Tổng số nhãn: {len(actions)}")
assert len(actions) == 61, f"LỖI: Mong đợi 61 nhãn, nhưng nhận được {len(actions)}"
print(f"   Nhãn đầu tiên [0]:  '{actions[0]}'")
print(f"   Nhãn giữa    [30]:  '{actions[30]}'")
print(f"   Nhãn cuối    [60]:  '{actions[60]}'")
print("=> [PASS] Đã nạp chính xác 61 nhãn chuẩn theo đúng index!\n")

print("=" * 80)
print("=== [TEST B1.2] KIỂM TRA HÀM CHUẨN HÓA normalize_keypoints() ===")
print("=" * 80)
# Giả lập 1 bàn tay raw: cổ tay tại (0.6, 0.4, 0.1), đầu ngón giữa tại (0.6, 0.6, 0.1) -> scale = 0.2
raw_dummy = np.zeros(126, dtype=np.float32)
# Left hand (0:63): wrist (idx 0)
raw_dummy[0] = 0.6; raw_dummy[1] = 0.4; raw_dummy[2] = 0.1
# middle_tip (idx 12 -> 12*3 = 36)
raw_dummy[36] = 0.6; raw_dummy[37] = 0.6; raw_dummy[38] = 0.1

norm_dummy = normalize_keypoints(raw_dummy)
# Cổ tay sau khi normalize phải về chính xác (0, 0, 0)
assert np.allclose(norm_dummy[:3], [0.0, 0.0, 0.0]), "Lỗi: Cổ tay chưa được centering về (0,0,0)"
# Middle tip sau khi chia cho scale 0.2 phải có khoảng cách đúng bằng 1.0
dist = np.linalg.norm(norm_dummy[36:39] - norm_dummy[:3])
assert np.isclose(dist, 1.0), f"Lỗi: Khoảng cách scale sau normalize không bằng 1.0 (nhận được: {dist})"
print(f"   Raw wrist:       {raw_dummy[:3]} -> Normalized wrist: {norm_dummy[:3]}")
print(f"   Raw middle tip:  {raw_dummy[36:39]} -> Normalized middle tip: {norm_dummy[36:39]}")
print(f"   Độ dài xương tay sau chuẩn hóa: {dist:.4f} (chuẩn hóa tỷ lệ thành công)")
print("=> [PASS] Hàm normalize_keypoints() hoạt động hoàn hảo!\n")

print("=" * 80)
print("=== [TEST B1.3] KHỞI TẠO VÀ LOAD TRỌNG SỐ MODEL TFLITE MỚI ===")
print("=" * 80)
tflite_path = os.path.join('Models', 'model_normalized_v1.tflite')
assert os.path.exists(tflite_path), f"Không tìm thấy model tại {tflite_path}"

model = TFLiteModelWrapper(tflite_path, num_threads=4)
print(f"=> Load TFLite model thành công từ: {tflite_path}")
print(f"=> Dung lượng file TFLite: {os.path.getsize(tflite_path)/1024:.1f} KB\n")

print("=" * 80)
print("=== [TEST B1.4] ĐÁNH GIÁ TRỰC TIẾP TRÊN 61 SEQUENCE THẬT TỪ DATA_NORMALIZED ===")
print("=" * 80)
data_dir = 'Data_normalized'
correct = 0
total = 0
details = []

for idx, label in enumerate(actions):
    label_dir = os.path.join(data_dir, label)
    if not os.path.isdir(label_dir):
        continue
    seq_dirs = sorted([d for d in os.listdir(label_dir) if os.path.isdir(os.path.join(label_dir, d))])
    if not seq_dirs:
        continue
    # Chọn 1 sequence thực tế để test
    seq_choice = seq_dirs[0]
    seq_path = os.path.join(label_dir, seq_choice)
    
    # Đọc 60 frames
    frames = []
    for f in range(60):
        fpath = os.path.join(seq_path, f"{f}.npy")
        if os.path.exists(fpath):
            frames.append(np.load(fpath))
    
    if len(frames) == 60:
        seq_arr = np.array(frames, dtype=np.float32)
        # Dự đoán
        preds = model.predict(np.expand_dims(seq_arr, axis=0), verbose=0)[0]
        pred_idx = np.argmax(preds)
        conf = preds[pred_idx]
        is_match = (pred_idx == idx)
        if is_match:
            correct += 1
        total += 1
        details.append((label, actions[pred_idx], conf, is_match))

print(f"{'STT':<4} | {'Nhãn gốc (Ground Truth)':<35} | {'Dự đoán (Prediction)':<35} | {'Conf':<7} | {'Kết quả'}")
print("-" * 95)
# In đại diện 15 nhãn mẫu
for i, (gt, pred, conf, match) in enumerate(details[:15]):
    status = "ĐÚNG [OK]" if match else "SAI [FAIL]"
    print(f"{i+1:<4} | {gt:<35} | {pred:<35} | {conf*100:>5.1f}% | {status}")
print(f"... (đã kiểm tra toàn bộ {total} nhãn)")

acc = correct / total if total > 0 else 0
print("-" * 95)
print(f"=> KẾT QUẢ INFERENCE: {correct}/{total} mẫu chính xác -> Accuracy: {acc*100:.2f}%\n")
assert acc > 0.90, f"LỖI: Accuracy trên các sequence thật quá thấp: {acc*100:.2f}%"

print("=" * 80)
print("=== [TEST B1.5] MÔ PHỎNG VÒNG LẶP SLIDING WINDOW FRAME-BY-FRAME ===")
print("=" * 80)
# Chọn 1 chuỗi ngẫu nhiên (ví dụ nhãn 'cam on')
sample_label = 'cam on'
sample_dir = os.path.join(data_dir, sample_label, sorted(os.listdir(os.path.join(data_dir, sample_label)))[0])
test_frames = [np.load(os.path.join(sample_dir, f"{i}.npy")) for i in range(60)]

buffer = []
predicted_result = None

# Mô phỏng đẩy từng frame vào như webcam stream
for f_idx, frame in enumerate(test_frames):
    buffer.append(frame)
    buffer = buffer[-60:]
    if len(buffer) == 60:
        res = model.predict(np.expand_dims(buffer, axis=0), verbose=0)[0]
        pred_idx = np.argmax(res)
        predicted_result = actions[pred_idx]
        conf = res[pred_idx]

print(f"Cử chỉ nạp vào stream:      '{sample_label}'")
print(f"Kết quả nhận diện khi đủ 60 frame: '{predicted_result}' (Confidence: {conf*100:.2f}%)")
assert predicted_result == sample_label, f"Lỗi: Dự đoán sai: {predicted_result} != {sample_label}"
print("=> [PASS] Cơ chế sliding window buffer 60 frames hoạt động chuẩn xác!\n")

print("=" * 80)
print("=== TỔNG KẾT: TOÀN BỘ 5/5 TEST CASES CHO BƯỚC B1 ĐỀU THÀNH CÔNG RỰC RỠ! ===")
print("=" * 80)
