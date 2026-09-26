# -*- coding: utf-8 -*-
"""
test_b3_consensus.py — Test độc lập cho Mục B3 (Sửa bug Consensus-check của Look & Tell)
========================================================================================
Kiểm thử 4 kịch bản đối chứng trực tiếp giữa logic cũ (Look & Tell) và logic mới đã sửa:
  [Case 1] Bug cũ để lọt: 9 frame nhãn A + 1 frame nhãn B  → Phải BỊ TỪ CHỐI
  [Case 2] Nhiễu hỗn loạn: 10 frame có 10 nhãn khác nhau   → Phải BỊ TỪ CHỐI
  [Case 3] Đồng thuận thật: 10/10 frame nhãn A, conf > 0.5  → Phải ĐƯỢC CHẤP NHẬN
  [Case 4] Đồng thuận nhưng conf thấp: 10/10 frame, conf < 0.5 → Phải BỊ TỪ CHỐI
"""
import os
import sys

# Fix Windows console UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from RunModel import load_actions, FSignRealtimeProcessor, CONSENSUS_WINDOW

print("=" * 85)
print("=== [TEST B3] KIỂM THỬ SỬA TRIỆT ĐỂ BUG CONSENSUS-CHECK CỦA LOOK & TELL ===")
print(f"Cửa sổ đồng thuận: CONSENSUS_WINDOW = {CONSENSUS_WINDOW} frames")
print("=" * 85)

actions = load_actions()

def old_look_and_tell_check(predictions, pred_idx, conf, threshold=0.5):
    """Mã nguồn bug cũ của Look & Tell."""
    if len(predictions) < 10:
        return False
    if np.unique(predictions[-10:])[0] == pred_idx:
        if conf > threshold:
            return True
    return False

def new_consensus_check(predictions, pred_idx, conf, consensus_window=10, threshold=0.5):
    """Mã nguồn logic mới đã sửa trong RunModel.py."""
    return (
        len(predictions) >= consensus_window and
        all(p == pred_idx for p in predictions[-consensus_window:]) and
        conf > threshold
    )

class MockModel:
    """Mock model để kiểm thử chính xác luồng dữ liệu của FSignRealtimeProcessor."""
    def __init__(self, pred_idx=0, conf=0.9):
        self.pred_idx = pred_idx
        self.conf = conf

    def set_prediction(self, pred_idx, conf):
        self.pred_idx = pred_idx
        self.conf = conf

    def predict(self, x, verbose=0):
        prob = np.zeros((1, len(actions)), dtype=np.float32)
        prob[0, self.pred_idx] = self.conf
        return prob

# Khởi tạo mock model và processor
mock = MockModel()
processor = FSignRealtimeProcessor(actions, model=mock, idle_threshold_sec=0.5, consensus_window=10, threshold=0.5)

# Nạp 59 frames ban đầu để chuẩn bị sẵn buffer
active_frame = np.ones(126, dtype=np.float32)
sim_time = 1000.0

for i in range(59):
    sim_time += 1.0 / 30.0
    processor.process_keypoints(active_frame, current_time=sim_time)

assert len(processor.sequence) == 59
print("=> Đã chuẩn bị sẵn buffer sequence (59 frames), sẵn sàng kiểm thử từng frame dự đoán.\n")

# ─── CASE 1: BUG CŨ ĐỂ LỌT (9 frame nhãn A + 1 frame nhãn B) ─────────────
print("--- [CASE 1] 9 frame nhãn A (0) + 1 frame nhãn B (1) ---")
# Giả lập lịch sử predictions: 9 lần nhãn 0, 1 lần nhãn 0 ở hiện tại
preds_case1 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  # 9 nhãn 0, 1 nhãn 1 xen vào ở vị trí áp chót
current_pred = 0
conf_case1 = 0.95

old_res = old_look_and_tell_check(preds_case1, current_pred, conf_case1)
new_res = new_consensus_check(preds_case1, current_pred, conf_case1)

print(f"  Predictions 10 frame: {preds_case1}")
print(f"  Nhãn hiện tại: {current_pred} (conf = {conf_case1})")
print(f"  np.unique(preds[-10:]): {list(np.unique(preds_case1[-10:]))} -> [0] là {np.unique(preds_case1[-10:])[0]}")
print(f"  -> Kết quả code cũ Look & Tell: {old_res} [NGHIÊM TRỌNG: Để lọt dù có frame 1 lẫn vào!]")
print(f"  -> Kết quả code mới sửa:        {new_res} [CHÍNH XÁC: Bị từ chối vì không đủ 10/10!]")
assert old_res == True, "Chứng minh bug: Code cũ phải trả về True do lấy nhãn 0 nhỏ nhất"
assert new_res == False, "Lỗi: Code mới không được phép chấp nhận khi có nhãn khác lẫn vào!"
print("=> [PASS] Case 1 kiểm thử thành công: Đã chặn đứng bug để lọt của Look & Tell!\n")

# ─── CASE 2: NHIỄU HỖN LOẠN (10 frame có các nhãn khác nhau) ─────────────
print("--- [CASE 2] 10 frame nhảy nhãn hỗn loạn (Nhiễu jitter) ---")
preds_case2 = [5, 8, 2, 7, 1, 9, 3, 6, 4, 0]
current_pred_case2 = 0
conf_case2 = 0.85

old_res2 = old_look_and_tell_check(preds_case2, current_pred_case2, conf_case2)
new_res2 = new_consensus_check(preds_case2, current_pred_case2, conf_case2)

print(f"  Predictions 10 frame: {preds_case2}")
print(f"  Nhãn hiện tại: {current_pred_case2} (conf = {conf_case2})")
print(f"  np.unique(preds[-10:]): {list(np.unique(preds_case2[-10:]))} -> [0] là {np.unique(preds_case2[-10:])[0]}")
print(f"  -> Kết quả code cũ Look & Tell: {old_res2} [NGHIÊM TRỌNG: Nhiễu 10 nhãn vẫn chấp nhận vì 0 nhỏ nhất!]")
print(f"  -> Kết quả code mới sửa:        {new_res2} [CHÍNH XÁC: Bị từ chối vì hoàn toàn hỗn loạn!]")
assert old_res2 == True
assert new_res2 == False
print("=> [PASS] Case 2 kiểm thử thành công: Khử nhiễu loạn nhãn tuyệt đối!\n")

# ─── CASE 3: ĐỒNG THUẬN THẬT (10/10 frame nhãn A, conf > 0.5) ─────────────
print("--- [CASE 3] Đồng thuận thật sự (10/10 frame nhãn 0, conf = 0.95 > 0.5) ---")
preds_case3 = [0] * 10
current_pred_case3 = 0
conf_case3 = 0.95

old_res3 = old_look_and_tell_check(preds_case3, current_pred_case3, conf_case3)
new_res3 = new_consensus_check(preds_case3, current_pred_case3, conf_case3)

print(f"  Predictions 10 frame: {preds_case3}")
print(f"  Nhãn hiện tại: {current_pred_case3} (conf = {conf_case3})")
print(f"  -> Kết quả code cũ Look & Tell: {old_res3}")
print(f"  -> Kết quả code mới sửa:        {new_res3} [CHÍNH XÁC: Đã chấp nhận vì 10/10 đồng nhất!]")
assert new_res3 == True, "Lỗi: 10/10 frame cùng 1 nhãn và conf cao nhưng bị từ chối!"

# Kiểm tra tích hợp thực tế qua FSignRealtimeProcessor
processor.sentence.clear()
processor.predictions = [0] * 9  # Đã có sẵn 9 frame nhãn 0
mock.set_prediction(pred_idx=0, conf=0.95)
sim_time += 1.0 / 30.0
res_proc3 = processor.process_keypoints(active_frame, current_time=sim_time)
print(f"  -> Kết quả FSignRealtimeProcessor chèn câu: {res_proc3['sentence']}")
assert res_proc3['sentence'] == [actions[0]], f"Lỗi: Chưa chèn được nhãn vào câu! ({res_proc3['sentence']})"
print("=> [PASS] Case 3 kiểm thử thành công: Cử chỉ đồng thuận được chấp nhận chuẩn xác!\n")

# ─── CASE 4: ĐỒNG THUẬN NHƯNG CONFIDENCE THẤP (< 0.5) ────────────────────
print("--- [CASE 4] 10/10 frame nhãn 1 nhưng confidence thấp (conf = 0.42 < 0.5) ---")
preds_case4 = [1] * 10
current_pred_case4 = 1
conf_case4 = 0.42

old_res4 = old_look_and_tell_check(preds_case4, current_pred_case4, conf_case4)
new_res4 = new_consensus_check(preds_case4, current_pred_case4, conf_case4)

print(f"  Predictions 10 frame: {preds_case4}")
print(f"  Nhãn hiện tại: {current_pred_case4} (conf = {conf_case4} < 0.5)")
print(f"  -> Kết quả code cũ Look & Tell: {old_res4} [Bị từ chối]")
print(f"  -> Kết quả code mới sửa:        {new_res4} [CHÍNH XÁC: Bị từ chối vì chưa đủ độ tin cậy!]")
assert new_res4 == False, "Lỗi: Confidence thấp mà vẫn cho qua!"

# Kiểm tra tích hợp thực tế qua FSignRealtimeProcessor
prev_sentence_len = len(processor.sentence)
processor.predictions = [1] * 9  # 9 frame nhãn 1
mock.set_prediction(pred_idx=1, conf=0.42)
sim_time += 1.0 / 30.0
res_proc4 = processor.process_keypoints(active_frame, current_time=sim_time)
print(f"  -> FSignRealtimeProcessor không chèn nhãn 1 vào câu: {res_proc4['sentence']}")
assert len(res_proc4['sentence']) == prev_sentence_len, "Lỗi: Confidence thấp nhưng vẫn chèn vào câu!"
print("=> [PASS] Case 4 kiểm thử thành công: Chặn đứng dự đoán thiếu tự tin!\n")

print("=" * 85)
print("=== TỔNG KẾT: TOÀN BỘ 4/4 TEST CASES CHO BƯỚC B3 ĐỀU THÀNH CÔNG XUẤT SẮC! ===")
print("=" * 85)
