# -*- coding: utf-8 -*-
"""
test_b2_idle.py — Test độc lập cho Mục B2 (Idle Rule-based theo thời gian thực)
================================================================================
Kiểm tra toàn bộ logic thuần của FSignRealtimeProcessor (không cần mở webcam):
  [Test B2.0] Sequence toàn zero: 60 frames 0 -> Chuyển sang Idle, predict_call_count = 0.
  [Test B2.1] Hoạt động Active bình thường: Khi có tay, tích luỹ 60 frame và thực hiện predict.
  [Test B2.2] Ngắt quãng ngắn (< 0.5s): Tay mất dấu trong 0.2s -> Chưa kích hoạt Idle, không predict.
  [Test B2.3] Kích hoạt IDLE (>= 0.5s): Mất tay đủ 0.55s -> Chuyển sang Idle, reset buffer về 0,
              chặn tuyệt đối model.predict().
  [Test B2.4] Idle kéo dài (5 giây): Mô phỏng người dùng nghỉ 5s -> predict_call_count không tăng.
  [Test B2.5] Phục hồi sạch sau Idle: Có tay lại -> Thoát Idle, buffer tích lũy từ frame 1,
              không bị lẫn lộn cử chỉ cũ.
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

from RunModel import load_actions, FSignRealtimeProcessor, IDLE_TIME_THRESHOLD_SEC, TFLiteModelWrapper

print("=" * 80)
print(f"=== [TEST B2] KIỂM THỬ TRẠNG THÁI NGHỈ (IDLE RULE-BASED) THEO THỜI GIAN THỰC ===")
print(f"Ngưỡng kích hoạt Idle: {IDLE_TIME_THRESHOLD_SEC} giây")
print("=" * 80)

actions = load_actions()
tflite_path = os.path.join('Models', 'model_normalized_v1.tflite')
model = TFLiteModelWrapper(tflite_path, num_threads=4)
print(f"=> Load TFLite model thành công từ: {tflite_path}")

empty_frame = np.zeros(126, dtype=np.float32)

# ─── TEST B2.0: Mảng sequence toàn zero (Yêu cầu đề bài) ───────────────────
print("\n--- [TEST B2.0] Mảng sequence toàn zero (60 frames 0) ---")
p_zero = FSignRealtimeProcessor(actions, model=model, idle_threshold_sec=IDLE_TIME_THRESHOLD_SEC)
t_zero = 2000.0
for i in range(60):
    t_zero += 1.0 / 30.0
    res_zero = p_zero.process_keypoints(empty_frame, current_time=t_zero)

print(f"  Trạng thái is_idle: {res_zero['is_idle']} (Mong đợi: True)")
print(f"  Số lần gọi predict: {p_zero.predict_call_count} (Mong đợi: 0 - Tuyệt đối không gọi)")
print(f"  Độ dài buffer:      {res_zero['buffer_len']} (Mong đợi: 0 - Đã reset sạch)")
print(f"  Status text:        '{res_zero['status_text']}'")
assert res_zero['is_idle'] == True, "Lỗi: Sequence toàn zero nhưng không chuyển sang Idle!"
assert p_zero.predict_call_count == 0, "Lỗi: Sequence toàn zero nhưng vẫn gọi model.predict()!"
assert res_zero['buffer_len'] == 0, "Lỗi: Sequence toàn zero nhưng buffer chưa reset về 0!"
print("=> [PASS] Test B2.0 thành công!\n")

# ─── Chuẩn bị dữ liệu cử chỉ hoạt động (active frames) ──────────────────────
# Tìm 1 sequence từ Data_normalized có đầy đủ các frame hoạt động
candidate_frames = None
cam_on_dir = os.path.join('Data_normalized', 'cam on')
for s in sorted(os.listdir(cam_on_dir)):
    s_path = os.path.join(cam_on_dir, s)
    if os.path.isdir(s_path):
        f_list = [np.load(os.path.join(s_path, f"{i}.npy")) for i in range(60) if os.path.exists(os.path.join(s_path, f"{i}.npy"))]
        if len(f_list) == 60:
            if all(not np.all(f == 0) for f in f_list):
                candidate_frames = f_list
                break
            elif candidate_frames is None:
                candidate_frames = f_list

# Đảm bảo 60 frames test là các frame cử chỉ có tay thực sự
active_only = [f for f in candidate_frames if not np.all(f == 0)]
if len(active_only) >= 60:
    active_60_frames = active_only[:60]
else:
    # Lặp lại các frame có tay để đảm bảo đủ 60 frame hoạt động
    active_60_frames = [active_only[i % len(active_only)] for i in range(60)]

# ─── Khởi tạo Processor chính để test chuỗi hành vi liên tục ───────────────
processor = FSignRealtimeProcessor(actions, model=model, idle_threshold_sec=IDLE_TIME_THRESHOLD_SEC)
sim_time = 1000.0

# ─── TEST B2.1: Hoạt động Active bình thường ──────────────────────────────
print("--- [TEST B2.1] Nạp 60 frames cử chỉ thật ('cam on') với FPS giả định 30 FPS ---")
for i in range(60):
    sim_time += 1.0 / 30.0
    res = processor.process_keypoints(active_60_frames[i], current_time=sim_time)

print(f"  Trạng thái is_idle: {res['is_idle']} (Mong đợi: False)")
print(f"  Độ dài buffer:      {res['buffer_len']} (Mong đợi: 60)")
print(f"  Số lần gọi predict: {processor.predict_call_count} (Mong đợi: 1)")
print(f"  Dự đoán:            '{res['predicted_action']}'")
assert not res['is_idle'], "Lỗi: Đang có tay mà lại báo Idle!"
assert res['buffer_len'] == 60, f"Lỗi: Buffer chưa đủ 60 frames (nhận được {res['buffer_len']})!"
assert processor.predict_call_count == 1, "Lỗi: Chưa gọi model.predict()!"
print("=> [PASS] Test B2.1 thành công!\n")

# ─── TEST B2.2: Ngắt quãng ngắn (< 0.5s, ví dụ 0.2s) ──────────────────────
print("--- [TEST B2.2] Mất dấu tay trong 0.2s (6 frames rỗng ở 30 FPS) ---")
prev_predict_count = processor.predict_call_count
for i in range(6):
    sim_time += 1.0 / 30.0
    res = processor.process_keypoints(empty_frame, current_time=sim_time)

print(f"  Thời gian mất tay:  0.20s (< {IDLE_TIME_THRESHOLD_SEC}s)")
print(f"  Trạng thái is_idle: {res['is_idle']} (Mong đợi: False - Chưa đủ ngưỡng để coi là nghỉ)")
print(f"  Số lần gọi predict: {processor.predict_call_count} (Mong đợi: {prev_predict_count} - Không tăng)")
print(f"  Status text:        '{res['status_text']}'")
assert not res['is_idle'], "Lỗi: Mất tay 0.2s (< 0.5s) không được kích hoạt Idle sớm!"
assert processor.predict_call_count == prev_predict_count, "Lỗi: Mất tay ngắn nhưng vẫn bị gọi predict!"
print("=> [PASS] Test B2.2 thành công!\n")

# ─── TEST B2.3: Kích hoạt IDLE (>= 0.5s) ──────────────────────────────────
print(f"--- [TEST B2.3] Tiếp tục mất dấu tay thêm 0.35s (Tổng cộng 0.55s >= {IDLE_TIME_THRESHOLD_SEC}s) ---")
for i in range(11):
    sim_time += 1.0 / 30.0
    res = processor.process_keypoints(empty_frame, current_time=sim_time)

print(f"  Thời gian mất tay:  0.55s (>= {IDLE_TIME_THRESHOLD_SEC}s)")
print(f"  Trạng thái is_idle: {res['is_idle']} (Mong đợi: True)")
print(f"  Độ dài buffer:      {res['buffer_len']} (Mong đợi: 0 - Đã reset sạch)")
print(f"  Số lần gọi predict: {processor.predict_call_count} (Mong đợi: {prev_predict_count} - Không tăng)")
print(f"  Status text:        '{res['status_text']}'")
assert res['is_idle'], "Lỗi: Mất tay quá 0.5s nhưng chưa chuyển sang Idle!"
assert res['buffer_len'] == 0, f"Lỗi: Buffer chưa được reset về 0 (nhận được: {res['buffer_len']})!"
assert processor.predict_call_count == prev_predict_count, "Lỗi: Khi Idle vẫn bị gọi predict lãng phí!"
print("=> [PASS] Test B2.3 thành công: Đã kích hoạt Idle và reset buffer triệt để!\n")

# ─── TEST B2.4: Buông tay kéo dài (Idle 5 giây) ───────────────────────────
print("--- [TEST B2.4] Buông tay kéo dài 5.0 giây (Mô phỏng người dùng nghỉ nói) ---")
idle_freeze_predict_count = processor.predict_call_count
for i in range(150):  # 150 frames ở 30 FPS = 5 giây
    sim_time += 1.0 / 30.0
    res = processor.process_keypoints(empty_frame, current_time=sim_time)

print(f"  Thời gian nghỉ:     ~5.5s")
print(f"  Trạng thái is_idle: {res['is_idle']} (Mong đợi: True)")
print(f"  Số lần gọi predict: {processor.predict_call_count} (Mong đợi: {idle_freeze_predict_count} - Giữ nguyên tuyệt đối)")
print(f"  Status text:        '{res['status_text']}'")
assert res['is_idle'], "Lỗi: Không duy trì trạng thái Idle!"
assert processor.predict_call_count == idle_freeze_predict_count, "Lỗi: Có lệnh predict bị lọt trong thời gian Idle!"
print("=> [PASS] Test B2.4 thành công: Khóa chặt predict trong suốt thời gian nghỉ!\n")

# ─── TEST B2.5: Phục hồi sạch sau Idle ────────────────────────────────────
print("--- [TEST B2.5] Đưa tay trở lại thực hiện cử chỉ mới (Phục hồi sau Idle) ---")
# Nạp 10 frames đầu của cử chỉ
for i in range(10):
    sim_time += 1.0 / 30.0
    res = processor.process_keypoints(active_60_frames[i], current_time=sim_time)

print(f"  Sau 10 frame có tay trở lại:")
print(f"  Trạng thái is_idle: {res['is_idle']} (Mong đợi: False - Đã thoát Idle ngay khi có tay)")
print(f"  Độ dài buffer:      {res['buffer_len']} (Mong đợi: 10 - Bắt đầu tích lũy sạch từ 1)")
print(f"  Status text:        '{res['status_text']}'")
assert not res['is_idle'], "Lỗi: Có tay trở lại nhưng vẫn bị kẹt ở Idle!"
assert res['buffer_len'] == 10, f"Lỗi: Buffer không tích lũy từ đầu, bị dính dữ liệu cũ! ({res['buffer_len']})"

# Nạp tiếp 50 frames còn lại để đủ 60 frame
for i in range(10, 60):
    sim_time += 1.0 / 30.0
    res = processor.process_keypoints(active_60_frames[i], current_time=sim_time)

print(f"  Khi đủ 60 frames cử chỉ mới:")
print(f"  Độ dài buffer:      {res['buffer_len']} (Mong đợi: 60)")
print(f"  Dự đoán mới:        '{res['predicted_action']}'")
print(f"  Số lần gọi predict: {processor.predict_call_count} (Mong đợi: {idle_freeze_predict_count + 1})")
assert res['buffer_len'] == 60
assert processor.predict_call_count == idle_freeze_predict_count + 1
print("=> [PASS] Test B2.5 thành công: Cử chỉ mới được nhận diện sạch sẽ, không bị lẫn lộn!\n")

print("=" * 80)
print("=== TỔNG KẾT: TOÀN BỘ 6/6 TEST CASES CHO BƯỚC B2 ĐỀU ĐẠT CHUẨN 100%! ===")
print("=" * 80)
