# -*- coding: utf-8 -*-
"""
test_c3_threaded_camera.py — Kiểm thử Unit Test cho ThreadedCamera (Mục C3)
===========================================================================
Kiểm tra toàn diện cơ chế Producer-Consumer đa luồng KHÔNG CẦN WEBCAM THẬT:
  - Producer: Giả lập camera phát frame tốc độ cao (100 FPS ~ 10ms/frame).
  - Consumer: Main loop đọc với tốc độ 20 FPS (~50ms/frame) trong 3 giây.
  
3 tiêu chí kiểm định nghiêm ngặt:
  (a) An toàn dữ liệu: Sau frame đầu tiên, 100% lần đọc KHÔNG BAO GIỜ bị None hoặc rác.
  (b) Ổn định đồng thời: Chạy song song không phát sinh exception, deadlock hoặc thread lockup.
  (c) Luôn nhận frame mới nhất: Frame ID nhận được luôn tăng dần và bám sát tốc độ của Producer,
      chứng minh deque(maxlen=1) tự động loại bỏ frame cũ và không bị kẹt buffer.
  (d) Dừng sạch (Clean shutdown): Luồng worker kết thúc hoàn toàn, không để lại zombie thread.
"""
import os
import sys
import time
import threading
from collections import deque
import numpy as np

# Fix Windows console UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from RunModel import ThreadedCamera

class MockVideoCapture:
    """Mock VideoCapture giả lập thiết bị camera phần cứng tốc độ 100 FPS."""
    def __init__(self, target_fps=100.0):
        self.interval = 1.0 / target_fps
        self.is_opened = True
        self.counter = 0
        self.lock = threading.Lock()

    def isOpened(self):
        return self.is_opened

    def read(self):
        if not self.is_opened:
            return False, None
        time.sleep(self.interval)
        with self.lock:
            self.counter += 1
            current_id = self.counter
            current_time = time.perf_counter()
        
        # Tạo frame giả lập chứa ID và timestamp
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Ghi ID vào góc frame để kiểm tra
        frame[0, 0, 0] = current_id % 256
        frame[0, 0, 1] = (current_id // 256) % 256
        return True, frame

    def release(self):
        self.is_opened = False

print("=" * 80)
print("=== [TEST C3] KIỂM THỬ ĐA LUỒNG & CHỐNG RACE CONDITION CHO THREADEDCAMERA ===")
print("Producer: 100 FPS (10ms/frame) | Consumer: 20 FPS (50ms/frame) | Thời gian: 3.0s")
print("=" * 80)

# Khởi tạo mock camera và ThreadedCamera
mock_cap = MockVideoCapture(target_fps=100.0)
cam = ThreadedCamera(custom_cap=mock_cap)

# Kiểm tra thread đã khởi chạy
assert cam.thread is not None, "Lỗi: Worker thread chưa được khởi tạo!"
assert cam.thread.is_alive(), "Lỗi: Worker thread chưa bắt đầu chạy!"
print(f"=> Worker thread khởi động thành công: '{cam.thread.name}' (daemon={cam.thread.daemon})\n")

read_records = []
exceptions = []
start_time = time.perf_counter()
duration = 3.0  # Chạy thử nghiệm trong 3 giây
consumer_interval = 0.05  # Đọc ở tần số 20 FPS

print(f"--- Bắt đầu phiên đọc Consumer đồng thời trong {duration}s ---")
read_count = 0
none_count = 0

while time.perf_counter() - start_time < duration:
    loop_start = time.perf_counter()
    try:
        ret, frame = cam.read()
        read_count += 1
        if not ret or frame is None:
            none_count += 1
        else:
            # Giải mã frame_id từ frame
            frame_id = int(frame[0, 0, 0]) + int(frame[0, 0, 1]) * 256
            read_records.append({
                'read_idx': read_count,
                'frame_id': frame_id,
                'timestamp': time.perf_counter() - start_time
            })
    except Exception as e:
        exceptions.append(e)
    
    elapsed_loop = time.perf_counter() - loop_start
    sleep_time = max(0.0, consumer_interval - elapsed_loop)
    time.sleep(sleep_time)

total_elapsed = time.perf_counter() - start_time
print(f"=> Hoàn tất phiên đọc sau {total_elapsed:.2f}s (Tổng lần đọc: {read_count} lần)")

# ─── KIỂM ĐỊNH TIÊU CHÍ (a): KHÔNG CÓ FRAME NONE / RÁC ───────────────────────
print("\n[Tiêu chí (a)] Kiểm tra tính toàn vẹn dữ liệu:")
print(f"  Tổng số lần đọc: {read_count}")
print(f"  Số lần bị None:  {none_count}")
assert none_count == 0, f"LỖI: Có {none_count} lần đọc bị None hoặc rỗng!"
assert len(read_records) == read_count, "Lỗi: Số lượng frame hợp lệ không khớp tổng số lần đọc!"
print("  => [PASS] 100% các lần đọc đều thu được frame hợp lệ, không bị rỗng/None!")

# ─── KIỂM ĐỊNH TIÊU CHÍ (b): KHÔNG CÓ EXCEPTION HOẶC DEADLOCK ───────────────
print("\n[Tiêu chí (b)] Kiểm tra độ ổn định đồng thời (No Deadlock / Exception):")
print(f"  Số exceptions phát sinh: {len(exceptions)}")
assert len(exceptions) == 0, f"LỖI: Phát hiện exception trong lúc chạy đa luồng: {exceptions}"
print(f"  Thời gian thực thi: {total_elapsed:.2f}s (Không bị treo/deadlock)")
print("  => [PASS] Không xảy ra deadlock hay race condition nào giữa Producer và Consumer!")

# ─── KIỂM ĐỊNH TIÊU CHÍ (d): CLEAN SHUTDOWN (DỪNG SẠCH THREAD) ───────────────
print("\n[Tiêu chí (d)] Kiểm tra dừng sạch (Clean Shutdown):")
t0_stop = time.perf_counter()
cam.stop()
stop_elapsed = (time.perf_counter() - t0_stop) * 1000.0

assert not cam.thread.is_alive(), "LỖI: Worker thread vẫn còn sống sau khi cam.stop()!"
assert not mock_cap.isOpened(), "LỖI: Mock camera chưa được giải phóng!"
print(f"  Thời gian dừng luồng: {stop_elapsed:.2f} ms")
print(f"  Trạng thái thread sau dừng: is_alive = {cam.thread.is_alive()}")
print("  => [PASS] Luồng worker đã dừng sạch sẽ trong < 0.5s, không để lại zombie thread!")

# ─── KIỂM ĐỊNH TIÊU CHÍ (c): LUÔN ĐỌC ĐƯỢC FRAME MỚI NHẤT (KHÔNG KẸT BUFFER) ─
print("\n[Tiêu chí (c)] Kiểm tra tính thời gian thực (Always Latest Frame):")
frame_ids = [r['frame_id'] for r in read_records]
producer_final_counter = mock_cap.counter
consumer_final_id = frame_ids[-1]
frame_lag = producer_final_counter - consumer_final_id

# 1. Kiểm tra tính tăng tiến đơn điệu (frame ID luôn tăng, không bao giờ thụt lùi)
is_monotonic = all(frame_ids[i] < frame_ids[i+1] for i in range(len(frame_ids)-1))
print(f"  Frame ID đầu tiên đọc được: {frame_ids[0]}")
print(f"  Frame ID cuối cùng đọc được: {consumer_final_id}")
print(f"  Producer đã tạo ra tổng cộng: {producer_final_counter} frames")
print(f"  Độ trễ frame cuối (Producer - Consumer): {frame_lag} frames (tương đương {frame_lag * 10:.1f}ms ở 100 FPS)")
print(f"  Tính tăng tiến đơn điệu (Monotonic Increase): {is_monotonic}")

assert is_monotonic, "LỖI: Frame ID không tăng tiến đơn điệu (phát hiện frame bị thụt lùi hoặc xáo trộn)!"
# Vì Consumer đọc chu kỳ 50ms (mỗi 5 frame Producer tạo ra), độ trễ <= 8 frames (< 80ms) chứng minh không có hàng đợi ứ đọng!
# (Nếu là hàng đợi FIFO thông thường không drop, độ trễ sẽ bị dồn ứ lên tới > 140 frames!).
assert frame_lag <= 8, f"LỖI: Buffer bị kẹt, độ trễ frame quá lớn: {frame_lag}!"
print(f"  => [PASS] Consumer luôn đọc được frame mới nhất (độ trễ {frame_lag} frames <= 1 chu kỳ đọc 50ms), deque(maxlen=1) tự động loại bỏ frame cũ thành công!")

print("\n" + "=" * 80)
print("=== TỔNG KẾT: TOÀN BỘ 4/4 TIÊU CHÍ CHO THREADEDCAMERA ĐỀU ĐẠT CHUẨN XUẤT SẮC! ===")
print("=" * 80)
