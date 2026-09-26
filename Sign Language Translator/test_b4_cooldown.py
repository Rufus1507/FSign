# -*- coding: utf-8 -*-
"""
test_b4_cooldown.py — Test độc lập cho Mục B4 (Cơ chế Cooldown theo thời gian thực)
===================================================================================
Kiểm thử 3 kịch bản theo đúng thiết kế:
  [Case 1] Chèn A tại t=0s, consensus B tại t=0.5s (< 1.2s) → BỊ CHẶN (Chống spam nhãn)
  [Case 2] Consensus B tại t=1.3s (>= 1.2s)               → ĐƯỢC CHẤP NHẬN (câu = [A, B])
  [Case 3] Consensus B lặp lại tại t=2.6s (>= 1.2s)       → ĐƯỢC CHẤP NHẬN (câu = [A, B, B], khắc phục hạn chế của Look & Tell)
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

from RunModel import load_actions, FSignRealtimeProcessor, COOLDOWN_TIME_SEC, CONSENSUS_WINDOW

print("=" * 85)
print("=== [TEST B4] KIỂM THỬ CƠ CHẾ COOLDOWN THEO THỜI GIAN THỰC (DEBOUNCE CHÈN CÂU) ===")
print(f"Cấu hình: COOLDOWN_TIME_SEC = {COOLDOWN_TIME_SEC}s, CONSENSUS_WINDOW = {CONSENSUS_WINDOW} frames")
print("=" * 85)

actions = load_actions()
action_A = actions[0]  # Ví dụ: 'ban dang lam gi'
action_B = actions[1]  # Ví dụ: 'ban di dau the'
print(f"=> Nhãn thử nghiệm A (idx 0): '{action_A}'")
print(f"=> Nhãn thử nghiệm B (idx 1): '{action_B}'\n")

class MockModel:
    """Mock model mô phỏng output dự đoán ổn định của mô hình."""
    def __init__(self, pred_idx=0, conf=0.95):
        self.pred_idx = pred_idx
        self.conf = conf

    def set_prediction(self, pred_idx, conf):
        self.pred_idx = pred_idx
        self.conf = conf

    def predict(self, x, verbose=0):
        prob = np.zeros((1, len(actions)), dtype=np.float32)
        prob[0, self.pred_idx] = self.conf
        return prob

# Khởi tạo mock model và processor FSign với cooldown 1.2s
mock = MockModel()
processor = FSignRealtimeProcessor(
    actions,
    model=mock,
    idle_threshold_sec=0.5,
    consensus_window=CONSENSUS_WINDOW,
    cooldown_sec=COOLDOWN_TIME_SEC,
    sequence_length=60,
    threshold=0.5
)

# Chuẩn bị sẵn buffer sequence (59 frames) với bàn tay hợp lệ
active_frame = np.ones(126, dtype=np.float32)
for _ in range(59):
    processor.process_keypoints(active_frame, current_time=-1.0)
assert len(processor.sequence) == 59
print("=> Đã chuẩn bị sẵn buffer sequence (59 frames), sẵn sàng đưa frame thứ 60 để kích hoạt predict.\n")

# ─── CASE 1: CHÈN A TẠI t=0s, CONSENSUS B TẠI t=0.5s (< 1.2s) → BỊ CHẶN ───
print("--- [CASE 1] Chèn A tại t=0.0s, consensus B tại t=0.5s (< 1.2s) ---")
# Giả lập từ A đã được chèn vào câu tại t=0.0s
processor.sentence = [action_A]
processor.last_action_time = 0.0

print(f"  Thời điểm ban đầu: t = 0.0s")
print(f"  Câu hiện tại:      {processor.sentence}")
print(f"  last_action_time:  {processor.last_action_time:.1f}s")

# Tại t=0.5s, mô hình đạt consensus 10 frame nhãn B (idx 1), conf = 0.95
t_case1 = 0.5
elapsed_1 = t_case1 - processor.last_action_time
mock.set_prediction(pred_idx=1, conf=0.95)
processor.predictions = [1] * (CONSENSUS_WINDOW - 1)  # 9 frame trước là 1

res1 = processor.process_keypoints(active_frame, current_time=t_case1)

print(f"  Tại t = {t_case1}s:")
print(f"    Elapsed:         {elapsed_1:.2f}s (< {COOLDOWN_TIME_SEC}s)")
print(f"    Consensus nhãn:  B ('{action_B}') với conf = 0.95")
print(f"    Trạng thái UI:   '{res1['status_text']}'")
print(f"    Câu sau xử lý:   {res1['sentence']}")

# Kiểm chứng:
assert res1['sentence'] == [action_A], f"Lỗi: Nhãn B bị chèn khi chưa hết cooldown! ({res1['sentence']})"
assert "Cho cooldown" in res1['status_text'], f"Lỗi: Trạng thái UI không hiển thị chờ cooldown! ({res1['status_text']})"
assert processor.last_action_time == 0.0, "Lỗi: last_action_time bị cập nhật sai khi hành động bị chặn!"
print(f"  -> Kết quả code cũ Look & Tell: SẼ CHÈN NGAY LẬP TỨC [NGHIÊM TRỌNG: Gây spam từ liên tục khi frame nhảy]")
print(f"  -> Kết quả code mới FSign:      BỊ CHẶN CHÍNH XÁC [Tránh spam nhãn thành công!]")
print("=> [PASS] Case 1 kiểm thử thành công: Cơ chế cooldown đã chặn đứng việc chèn nhãn quá sớm!\n")

# ─── CASE 2: CONSENSUS B TẠI t=1.3s (>= 1.2s) → ĐƯỢC CHẤP NHẬN, CÂU = [A, B] ───
print("--- [CASE 2] Consensus B tại t=1.3s (>= 1.2s) ---")
t_case2 = 1.3
elapsed_2 = t_case2 - processor.last_action_time

mock.set_prediction(pred_idx=1, conf=0.95)
processor.predictions = [1] * (CONSENSUS_WINDOW - 1)

res2 = processor.process_keypoints(active_frame, current_time=t_case2)

print(f"  Tại t = {t_case2}s:")
print(f"    Elapsed:         {elapsed_2:.2f}s (>= {COOLDOWN_TIME_SEC}s)")
print(f"    Consensus nhãn:  B ('{action_B}') với conf = 0.95")
print(f"    Trạng thái UI:   '{res2['status_text']}'")
print(f"    last_action_time:{processor.last_action_time:.1f}s")
print(f"    Câu sau xử lý:   {res2['sentence']}")

# Kiểm chứng:
assert res2['sentence'] == [action_A, action_B], f"Lỗi: Câu không đúng [A, B]! ({res2['sentence']})"
assert processor.last_action_time == t_case2, f"Lỗi: last_action_time không được cập nhật lên {t_case2}s!"
print(f"  -> Kết quả FSign: ĐÃ CHẤP NHẬN CHÈN B [Hợp lệ vì đã trôi qua {elapsed_2:.2f}s >= {COOLDOWN_TIME_SEC}s]")
print("=> [PASS] Case 2 kiểm thử thành công: Chèn từ mới chuẩn xác sau khi hết cooldown!\n")

# ─── CASE 3: CONSENSUS B LẶP LẠI TẠI t=2.6s (>= 1.2s) → CÂU = [A, B, B] ─────
print("--- [CASE 3] Consensus B lặp lại tại t=2.6s (>= 1.2s sau B trước) ---")
t_case3 = 2.6
elapsed_3 = t_case3 - processor.last_action_time

mock.set_prediction(pred_idx=1, conf=0.95)
processor.predictions = [1] * (CONSENSUS_WINDOW - 1)

# So sánh hành vi với code cũ Look & Tell:
look_and_tell_repeat_blocked = (action_B == res2['sentence'][-1])
print(f"  Đặc điểm Look & Tell cũ: if predicted_action != sentence[-1]")
print(f"  -> Look & Tell cũ kiểm tra: ('{action_B}' != '{res2['sentence'][-1]}') -> {not look_and_tell_repeat_blocked}")
print(f"  -> Hệ quả Look & Tell cũ: TUYỆT ĐỐI KHÔNG THỂ NÓI 2 TỪ GIỐNG NHAU LIÊN TIẾP (ví dụ: 'cam on ... cam on')!")

res3 = processor.process_keypoints(active_frame, current_time=t_case3)

print(f"\n  Tại t = {t_case3}s:")
print(f"    Elapsed:         {elapsed_3:.2f}s (>= {COOLDOWN_TIME_SEC}s)")
print(f"    Consensus nhãn:  B lặp lại ('{action_B}') với conf = 0.95")
print(f"    Trạng thái UI:   '{res3['status_text']}'")
print(f"    last_action_time:{processor.last_action_time:.1f}s")
print(f"    Câu sau xử lý:   {res3['sentence']}")

# Kiểm chứng:
assert res3['sentence'] == [action_A, action_B, action_B], f"Lỗi: Câu không đúng [A, B, B]! ({res3['sentence']})"
assert processor.last_action_time == t_case3, f"Lỗi: last_action_time không được cập nhật lên {t_case3}s!"
print(f"  -> Kết quả FSign: ĐÃ CHẤP NHẬN CHÈN LẶP LẠI B [Cho phép lặp từ hợp lệ sau {elapsed_3:.2f}s cooldown!]")
print("=> [PASS] Case 3 kiểm thử thành công: Khắc phục triệt để điểm nghẽn không thể lặp từ của Look & Tell!\n")

print("=" * 85)
print("=== TỔNG KẾT: TOÀN BỘ 3/3 TEST CASES CHO BƯỚC B4 ĐỀU THÀNH CÔNG XUẤT SẮC! ===")
print("=" * 85)
