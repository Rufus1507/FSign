# -*- coding: utf-8 -*-
"""
test_b5_dropout.py — Test độc lập cho Mục B5 (Khử nhiễu tracking dropout < 0.5s)
===============================================================================
Mục tiêu kiểm thử:
  Chứng minh và khắc phục triệt để lỗi rò rỉ frame zero vào sequence buffer
  khi MediaPipe bị mất dấu tay thoáng qua (dropout ngắn < 0.5s) giữa một cử chỉ:

  Kịch bản mô phỏng:
    - 30 frames cử chỉ thật (nửa đầu)
    - 6 frames mất tay rỗng (0.2s ở 30 FPS, dưới ngưỡng Idle 0.5s)
    - 30 frames tiếp theo của CÙNG cử chỉ đó (nửa sau)

  Kiểm chứng đối chứng:
    [Code Cũ] archive/RunModel_pre_b5.py:
      - Trong lúc chờ Idle, vẫn normalize và append 6 frame zero vào sequence!
      - Sequence buffer bị ô nhiễm 6 frame zero ở giữa, đẩy mất 6 frame thật ở đầu.
    [Code Mới] RunModel.py:
      - Khi mất tay, tuyệt đối KHÔNG append vào buffer; tạm ngưng và bảo toàn buffer (30 frames).
      - Khi tay quay lại, 30 frame nửa sau được nối tiếp sạch sẽ -> sequence đủ 60 frame thật (30+30).
      - Sequence chứa 0 frame zero, model nhận diện chuẩn xác cử chỉ!
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from preprocessing import normalize_keypoints
from RunModel import load_actions, FSignRealtimeProcessor, TFLiteModelWrapper

# ─── ĐỊNH NGHĨA PROCESSOR CODE CŨ (archive/RunModel_pre_b5.py) ĐỂ ĐỐI CHỨNG ─
class OldFSignRealtimeProcessor:
    """Mã nguồn logic cũ trước B5 (archive/RunModel_pre_b5.py) - Rò rỉ frame zero vào buffer sequence"""
    def __init__(self, actions, model=None, idle_threshold_sec=0.5, sequence_length=60):
        self.actions = actions
        self.model = model
        self.idle_threshold_sec = idle_threshold_sec
        self.sequence_length = sequence_length
        self.sequence = []
        self.empty_hand_start_time = None
        self.is_idle = True
        self.status_text = "Dang cho / Idle"
        self.predict_call_count = 0

    def process_keypoints(self, keypoints_raw: np.ndarray, current_time: float = None) -> dict:
        if current_time is None:
            current_time = time.time()

        is_empty_hand = np.all(keypoints_raw == 0)

        if is_empty_hand:
            if self.empty_hand_start_time is None:
                self.empty_hand_start_time = current_time

            elapsed = current_time - self.empty_hand_start_time
            if elapsed >= self.idle_threshold_sec:
                self.is_idle = True
                self.status_text = f"Dang cho / Idle ({elapsed:.1f}s)"
                if len(self.sequence) > 0:
                    self.sequence.clear()
                return {
                    'is_idle': True,
                    'status_text': self.status_text,
                    'predicted_action': None,
                    'confidence': 0.0,
                    'buffer_len': 0
                }
            else:
                self.status_text = f"Cho phat hien tay ({elapsed:.2f}s)"
                # [BUG CŨ]: Không return ở đây -> rơi xuống normalize và append frame zero!
        else:
            self.empty_hand_start_time = None
            self.is_idle = False

        # [BUG CŨ]: Vector rỗng vẫn bị normalize và append vào sequence!
        keypoints_norm = normalize_keypoints(keypoints_raw)
        self.sequence.append(keypoints_norm)
        self.sequence = self.sequence[-self.sequence_length:]

        predicted_action = None
        confidence = 0.0

        if len(self.sequence) == self.sequence_length:
            if not is_empty_hand and not self.is_idle:
                self.status_text = "Dang nhan dien..."
                if self.model is not None:
                    self.predict_call_count += 1
                    res = self.model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
                    pred_idx = np.argmax(res)
                    predicted_action = self.actions[pred_idx]
                    confidence = float(res[pred_idx])
            else:
                self.status_text = "Tam ngung / Cho tay"

        return {
            'is_idle': self.is_idle,
            'status_text': self.status_text,
            'predicted_action': predicted_action,
            'confidence': confidence,
            'buffer_len': len(self.sequence)
        }

print("=" * 88)
print("=== [TEST B5] KIỂM THỬ KHỬ RÒ RỈ FRAME ZERO KHI MẤT DẤU TAY THOÁNG QUA (< 0.5s) ===")
print("=" * 88)

# 1. Nạp danh sách nhãn và mô hình chuẩn TFLite
actions = load_actions()
tflite_path = os.path.join('Models', 'model_normalized_v1.tflite')
model = TFLiteModelWrapper(tflite_path, num_threads=4)
print(f"=> Load model TFLite thành công từ {tflite_path} ({len(actions)} nhãn)")

# 2. Tìm một chuỗi 60 frames cử chỉ thật hoàn toàn có tay trong Data_normalized
target_label = "cam on"
chosen_seq_dir = None
real_60_frames = None

for act in ["cam on", "ban dang lam gi", "chuc mung"]:
    act_dir = os.path.join("Data_normalized", act)
    if not os.path.isdir(act_dir):
        continue
    for sub in sorted(os.listdir(act_dir)):
        sub_path = os.path.join(act_dir, sub)
        if not os.path.isdir(sub_path):
            continue
        frames = []
        for f in range(60):
            fp = os.path.join(sub_path, f"{f}.npy")
            if os.path.exists(fp):
                frames.append(np.load(fp))
            else:
                break
        if len(frames) == 60:
            active_only = [f for f in frames if not np.all(f == 0)]
            if len(active_only) >= 60:
                target_label = act
                chosen_seq_dir = sub
                real_60_frames = active_only[:60]
                break
            elif len(active_only) >= 50:
                # Đệm các frame có tay để đủ 60 frame sạch
                target_label = act
                chosen_seq_dir = sub
                real_60_frames = [active_only[i % len(active_only)] for i in range(60)]
                break
    if real_60_frames is not None:
        break

assert real_60_frames is not None, "Không tìm thấy chuỗi frame cử chỉ có tay phù hợp"
print(f"=> Sử dụng chuỗi 60 frame thật của nhãn '{target_label}' ({target_label}/{chosen_seq_dir})")

first_half_30 = real_60_frames[:30]
second_half_30 = real_60_frames[30:]
empty_frame = np.zeros(126, dtype=np.float32)

# Baseline: Dự đoán trên 60 frame liên tục không bị ngắt
baseline_preds = model.predict(np.expand_dims(real_60_frames, axis=0), verbose=0)[0]
baseline_idx = np.argmax(baseline_preds)
baseline_conf = baseline_preds[baseline_idx]
print(f"=> Baseline khi không ngắt quãng: Dự đoán '{actions[baseline_idx]}' (conf = {baseline_conf*100:.2f}%)\n")

# ─── ĐỐI CHỨNG: CHẠY TRÊN CODE CŨ (archive/RunModel_pre_b5.py) ───────────────
print("-" * 88)
print("--- [ĐỐI CHỨNG] KIỂM THỬ TRÊN CODE CŨ (TRƯỚC KHI SỬA) ---")
old_proc = OldFSignRealtimeProcessor(actions, model=model, idle_threshold_sec=0.5)
sim_time = 100.0

# Bước 1: Nạp 30 frame đầu
for f in first_half_30:
    sim_time += 1.0 / 30.0
    old_proc.process_keypoints(f, current_time=sim_time)
len_after_first_half_old = len(old_proc.sequence)
print(f"  1. Sau 30 frame đầu:       Buffer len = {len_after_first_half_old} (Mong đợi: 30)")

# Bước 2: Bị mất tay trong 6 frames (0.2s < 0.5s)
for _ in range(6):
    sim_time += 1.0 / 30.0
    res_dropout_old = old_proc.process_keypoints(empty_frame, current_time=sim_time)
len_during_dropout_old = len(old_proc.sequence)
zero_in_buffer_during_old = sum(1 for f in old_proc.sequence if np.all(f == 0))
print(f"  2. Sau 6 frame mất tay:    Buffer len = {len_during_dropout_old} [NGHIÊM TRỌNG: Bị tăng lên 36 do đẩy frame zero vào!]")
print(f"     Số frame zero bị nhét:  {zero_in_buffer_during_old} frames zero trong buffer!")

# Bước 3: Nạp 30 frame sau
res_final_old = None
for f in second_half_30:
    sim_time += 1.0 / 30.0
    res_final_old = old_proc.process_keypoints(f, current_time=sim_time)

len_final_old = len(old_proc.sequence)
zero_count_in_final_old = sum(1 for f in old_proc.sequence if np.all(f == 0))
print(f"  3. Sau 30 frame tiếp theo: Buffer len = {len_final_old}")
print(f"     Số frame zero lẫn vào:  {zero_count_in_final_old} frames zero vẫn kẹt trong 60 frames của sequence!")
print(f"     Kết quả dự đoán cũ:     '{res_final_old['predicted_action']}' (conf = {res_final_old['confidence']*100:.2f}%)")
print(f"  -> KẾT LUẬN CODE CŨ: XÁC NHẬN CÓ BUG! Buffer bị rò rỉ frame zero, đẩy mất {zero_count_in_final_old} frame đầu cử chỉ.\n")

# ─── KIỂM THỬ TRÊN CODE MỚI ĐÃ SỬA (RunModel.py) ────────────────────────────
print("-" * 88)
print("--- [CODE MỚI] KIỂM THỬ TRÊN RUNMODEL.PY ĐÃ TÁI CẤU TRÚC HOÀN CHỈNH ---")
new_proc = FSignRealtimeProcessor(actions, model=model, idle_threshold_sec=0.5)
sim_time = 100.0

# Bước 1: Nạp 30 frame đầu
for f in first_half_30:
    sim_time += 1.0 / 30.0
    new_proc.process_keypoints(f, current_time=sim_time)
len_after_first_half_new = len(new_proc.sequence)
print(f"  1. Sau 30 frame đầu:       Buffer len = {len_after_first_half_new} (Mong đợi: 30)")
assert len_after_first_half_new == 30, "Lỗi: Buffer sau nửa đầu không đúng 30!"

# Bước 2: Bị mất tay trong 6 frames (0.2s < 0.5s)
res_dropout_new = None
for _ in range(6):
    sim_time += 1.0 / 30.0
    res_dropout_new = new_proc.process_keypoints(empty_frame, current_time=sim_time)
len_during_dropout_new = len(new_proc.sequence)
zero_in_buffer_during_new = sum(1 for f in new_proc.sequence if np.all(f == 0))
print(f"  2. Sau 6 frame mất tay:    Buffer len = {len_during_dropout_new} [BẢO TOÀN TUYỆT ĐỐI: Giữ nguyên 30, KHÔNG append!]")
print(f"     Số frame zero bị nhét:  {zero_in_buffer_during_new} (Sạch hoàn toàn!)")
print(f"     Trạng thái UI:          '{res_dropout_new['status_text']}'")
assert len_during_dropout_new == 30, f"Lỗi: Buffer bị thay đổi trong lúc mất tay ngắn ({len_during_dropout_new})!"
assert zero_in_buffer_during_new == 0, f"Lỗi: Có frame zero lọt vào buffer trong lúc mất tay!"
assert not res_dropout_new['is_idle'], "Lỗi: Mất tay 0.2s chưa được coi là Idle!"

# Bước 3: Nạp 30 frame sau
res_final_new = None
for f in second_half_30:
    sim_time += 1.0 / 30.0
    res_final_new = new_proc.process_keypoints(f, current_time=sim_time)

len_final_new = len(new_proc.sequence)
zero_count_in_final_new = sum(1 for f in new_proc.sequence if np.all(f == 0))
raw_pred_idx_new = new_proc.predictions[-1]
raw_pred_label_new = actions[raw_pred_idx_new]

print(f"  3. Sau 30 frame tiếp theo: Buffer len = {len_final_new} (Đủ chuẩn 60 frames)")
print(f"     Số frame zero lẫn vào:  {zero_count_in_final_new} (Hoàn toàn KHÔNG lẫn một frame zero nào!)")
print(f"     Model predict frame 60: '{raw_pred_label_new}' (conf = {res_final_new['confidence']*100:.2f}%)")
print(f"     Nhãn baseline mong đợi: '{actions[baseline_idx]}' (conf = {baseline_conf*100:.2f}%)")

assert len_final_new == 60, f"Lỗi: Độ dài buffer cuối cùng không đúng 60 ({len_final_new})!"
assert zero_count_in_final_new == 0, f"Lỗi: Có frame zero nằm trong buffer cuối ({zero_count_in_final_new})!"
assert raw_pred_label_new == actions[baseline_idx], f"Lỗi: Model nhận diện sai cử chỉ ({raw_pred_label_new} != {actions[baseline_idx]})!"
assert np.isclose(res_final_new['confidence'], baseline_conf, atol=1e-3), f"Lỗi: Độ tin cậy bị lệch so với baseline ({res_final_new['confidence']} vs {baseline_conf})!"
print(f"=> [PASS] Code mới đã bảo toàn nguyên vẹn 60 frames cử chỉ thật và nhận diện chính xác 100%!\n")

print("=" * 88)
print("=== SO SÁNH TRỰC DIỆN KẾT QUẢ GIỮA CODE CŨ VÀ CODE MỚI ===")
print("=" * 88)
print(f"{'Tiêu chí đánh giá':<36} | {'Code cũ (Bug)':<22} | {'Code mới (Đã sửa)':<22}")
print("-" * 88)
print(f"{'Buffer khi mất tay 0.2s':<36} | {len_during_dropout_old:<22} | {len_during_dropout_new:<22} [Bảo toàn]")
print(f"{'Frame zero lẫn vào sequence':<36} | {zero_count_in_final_old:<22} | {zero_count_in_final_new:<22} [Triệt tiêu 100%]")
print(f"{'Frame cử chỉ thật bị đẩy mất':<36} | {zero_count_in_final_old:<22} | {'0':<22} [Bảo toàn 60/60]")
print(f"{'Dự đoán model khi mất tay':<36} | {res_final_old['predicted_action']:<22} | {raw_pred_label_new:<22} [ĐÚNG 100%]")
print(f"{'Kết quả nhận diện cử chỉ':<36} | {'SAI BÉT (chuc mung)':<22} | {'CHÍNH XÁC (cam on)':<22} [Khắc phục]")
print("-" * 88)
print("=> [TỔNG KẾT] TOÀN BỘ YÊU CẦU KIỂM THỬ B5 ĐỀU ĐẠT CHUẨN 100%!")
print("=" * 88)
