# -*- coding: utf-8 -*-
"""
archive/RunModel_pre_c3.py — Bản lưu trữ RunModel.py trước khi cài đặt Multi-threading (Mục C3)
=============================================================================================
Bản này đã merge hoàn tất TFLite Engine (Models/model_normalized_v1.tflite) với TFLiteModelWrapper,
vượt qua 100% 5 bài test GĐ B (B1-B5), tốc độ suy luận ~4.1ms lúc webcam:
  - B1: Chuẩn hóa per-hand (normalize_keypoints).
  - B2: Idle Detection 0.5s theo thời gian thực.
  - B3: Consensus-check 10 frame liên tiếp cùng nhãn.
  - B4: Cooldown 1.2s theo thời gian thực.
  - B5: Chống rò rỉ frame zero khi mất dấu tay ngắn (< 0.5s).
  - C2: TFLite Engine (XNNPACK 4 threads).
"""
import cv2
import numpy as np
import os
import time
from collections import deque
from matplotlib import pyplot as plt
import mediapipe as mp
import tensorflow as tf
from model_def import build_model
from preprocessing import normalize_keypoints

# Cấu hình thời gian thực
IDLE_TIME_THRESHOLD_SEC = 0.5  # Ngưỡng thời gian không thấy tay (giây) để coi là Idle
CONSENSUS_WINDOW = 10          # Số frame liên tiếp cần đồng thuận cùng 1 nhãn (Mục B3)
COOLDOWN_TIME_SEC = 1.2        # Thời gian cooldown giữa 2 lần chèn từ vào câu (Mục B4)

# Chuyển working directory về thư mục chứa file script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# Khởi tạo mô hình MediaPipe Holistic
mp_hands = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Chuyển đổi màu BGR sang RGB
    image.flags.writeable = False                  # Tối ưu xử lý ảnh
    results = model.process(image)                 # Dự đoán landmark
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Chuyển lại màu BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)

def draw_styled_landmarks(image, results):
    # Vẽ bàn tay trái
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    )
    # Vẽ bàn tay phải
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

def load_actions(cache_path=os.path.join('Data_normalized', 'dataset_cache.npz'), data_dir='Data_normalized'):
    """
    Lấy danh sách 61 nhãn chuẩn đúng thứ tự index lúc train:
    1. Đọc trực tiếp từ dataset_cache.npz (label_map) để đảm bảo 100% khớp index.
    2. Fallback duyệt Data_normalized nếu chưa có file cache.
    """
    if os.path.exists(cache_path):
        c = np.load(cache_path, allow_pickle=True)
        label_map = c['label_map'].item()
        return np.array([k for k, v in sorted(label_map.items(), key=lambda x: x[1])])
    elif os.path.exists(data_dir):
        return np.array(sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]))
    else:
        raise FileNotFoundError(f"Không tìm thấy nguồn dữ liệu nhãn tại {cache_path} hoặc {data_dir}")

# Danh sách 61 câu cử chỉ nhận diện chuẩn từ dataset
actions = load_actions()

class FSignRealtimeProcessor:
    """
    Bộ xử lý suy luận thời gian thực cho FSign:
    - Quản lý trạng thái nghỉ (Idle Detection) theo thời gian thực (time.time()).
    - Chuẩn hóa tọa độ per-hand trước khi đẩy vào sequence buffer.
    - Reset buffer sequence và predictions khi rơi vào Idle để tránh lẫn lộn cử chỉ.
    - Quản lý dự đoán và danh sách câu kết quả.
    """
    def __init__(self, actions, model=None, idle_threshold_sec=IDLE_TIME_THRESHOLD_SEC, consensus_window=CONSENSUS_WINDOW, cooldown_sec=COOLDOWN_TIME_SEC, sequence_length=60, threshold=0.5):
        self.actions = actions
        self.model = model
        self.idle_threshold_sec = idle_threshold_sec
        self.consensus_window = consensus_window
        self.cooldown_sec = cooldown_sec
        self.sequence_length = sequence_length
        self.threshold = threshold

        self.sequence = []
        self.sentence = []
        self.predictions = []

        self.empty_hand_start_time = None
        self.is_idle = True
        self.last_action_time = 0.0
        self.status_text = "Dang cho / Idle"
        self.predict_call_count = 0

    def process_keypoints(self, keypoints_raw: np.ndarray, current_time: float = None) -> dict:
        """
        Xử lý 1 vector keypoints thô (126,) tại thời điểm current_time.
        Bổ sung telemetry đo thời gian chuẩn hóa (t_norm) và suy luận (t_predict) - Mục C1.
        """
        if current_time is None:
            current_time = time.time()

        t_norm = 0.0
        t_predict = 0.0
        predicted_this_frame = False

        is_empty_hand = np.all(keypoints_raw == 0)

        # ─── 1. XỬ LÝ TRƯỜNG HỢP MẤT DẤU TAY (EMPTY HAND) ───
        # Tuyệt đối KHÔNG normalize hay append frame rỗng vào sequence buffer!
        if is_empty_hand:
            if self.empty_hand_start_time is None:
                self.empty_hand_start_time = current_time

            elapsed = current_time - self.empty_hand_start_time
            if elapsed >= self.idle_threshold_sec:
                # Quá ngưỡng thời gian -> Kích hoạt Idle thật sự, dọn sạch buffer
                self.is_idle = True
                self.status_text = f"Dang cho / Idle ({elapsed:.1f}s)"
                if len(self.sequence) > 0:
                    self.sequence.clear()
                if len(self.predictions) > 0:
                    self.predictions.clear()
            else:
                # Dưới ngưỡng thời gian -> Tạm ngưng (Dropout ngắn), bảo toàn buffer, KHÔNG append frame rỗng
                self.status_text = f"Tam ngung / Cho tay ({elapsed:.2f}s)"

            return {
                'is_idle': self.is_idle,
                'status_text': self.status_text,
                'predicted_action': None,
                'confidence': 0.0,
                'sentence': list(self.sentence),
                'buffer_len': len(self.sequence),
                't_norm': t_norm,
                't_predict': t_predict,
                'predicted_this_frame': predicted_this_frame
            }

        # ─── 2. KHI CÓ TAY (NOT EMPTY HAND) ───
        # Thoát Idle và đặt lại đồng hồ theo dõi rỗng tay
        self.empty_hand_start_time = None
        self.is_idle = False

        # CHỈ chuẩn hóa và đưa vào buffer sequence khi THẬT SỰ CÓ TAY (đo lường t_norm)
        t0_norm = time.perf_counter()
        keypoints_norm = normalize_keypoints(keypoints_raw)
        t_norm = time.perf_counter() - t0_norm

        self.sequence.append(keypoints_norm)
        self.sequence = self.sequence[-self.sequence_length:]

        predicted_action = None
        confidence = 0.0

        if len(self.sequence) == self.sequence_length:
            self.status_text = "Dang nhan dien..."
            if self.model is not None:
                self.predict_call_count += 1
                t0_pred = time.perf_counter()
                res = self.model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
                t_predict = time.perf_counter() - t0_pred
                predicted_this_frame = True

                pred_idx = np.argmax(res)
                self.predictions.append(pred_idx)
                confidence = float(res[pred_idx])

                # Sửa triệt để bug consensus-check cũ của Look & Tell (Mục B3):
                # Yêu cầu TOÀN BỘ consensus_window frame gần nhất cùng dự đoán ra pred_idx VÀ confidence > threshold
                is_consensus = (
                    len(self.predictions) >= self.consensus_window and
                    all(p == pred_idx for p in self.predictions[-self.consensus_window:]) and
                    confidence > self.threshold
                )

                # Cơ chế Cooldown theo thời gian thực (Mục B4):
                # Chặn đứng spam nhãn liên tục và cho phép lặp lại từ một cách hợp lệ sau khi hết cooldown
                if is_consensus:
                    predicted_action = self.actions[pred_idx]
                    elapsed_cooldown = current_time - self.last_action_time
                    if elapsed_cooldown >= self.cooldown_sec:
                        self.sentence.append(predicted_action)
                        self.last_action_time = current_time
                    else:
                        self.status_text = f"Cho cooldown ({self.cooldown_sec - elapsed_cooldown:.1f}s)"

                if len(self.sentence) > 5:
                    self.sentence = self.sentence[-5:]
        else:
            self.status_text = f"Thu thap cu chi: {len(self.sequence)}/{self.sequence_length}"

        return {
            'is_idle': self.is_idle,
            'status_text': self.status_text,
            'predicted_action': predicted_action,
            'confidence': confidence,
            'sentence': list(self.sentence),
            'buffer_len': len(self.sequence),
            't_norm': t_norm,
            't_predict': t_predict,
            'predicted_this_frame': predicted_this_frame
        }

def run_realtime_detection(model):
    processor = FSignRealtimeProcessor(
        actions,
        model=model,
        idle_threshold_sec=IDLE_TIME_THRESHOLD_SEC,
        consensus_window=CONSENSUS_WINDOW,
        cooldown_sec=COOLDOWN_TIME_SEC,
        sequence_length=60,
        threshold=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Lỗi] Không thể mở webcam!")
        return

    # Khởi tạo các buffer đo đếm hiệu năng thời gian thực (Mục C1)
    window_size = 30
    fps_history = deque(maxlen=window_size)
    cap_times = deque(maxlen=window_size)
    mp_times = deque(maxlen=window_size)
    norm_times = deque(maxlen=window_size)
    predict_times = deque(maxlen=window_size)
    predict_called_history = deque(maxlen=window_size)
    draw_times = deque(maxlen=window_size)
    compute_times = deque(maxlen=window_size)

    log_file_path = "benchmark_c1_fps.log"
    log_file = open(log_file_path, "w", encoding="utf-8")
    header_info = (
        "=================================================================\n"
        "=== FSIGN REAL-TIME FPS & LATENCY BENCHMARK (MỤC C1) ===\n"
        f"Thời gian bắt đầu: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        "================================================================="
    )
    print(header_info)
    log_file.write(header_info + "\n")
    log_file.flush()

    frame_count = 0
    session_start_time = time.perf_counter()

    print("=> Bắt đầu nhận diện qua webcam. Nhấn 'q' trên cửa sổ video để dừng và xem tổng kết.")
    with mp_hands.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            t_frame_start = time.perf_counter()

            # 1. Đo thời gian đọc frame từ webcam (cap.read)
            t0 = time.perf_counter()
            ret, frame = cap.read()
            t_cap = time.perf_counter() - t0
            if not ret:
                break

            # 2. Đo thời gian MediaPipe Holistic
            t0 = time.perf_counter()
            image, results = mediapipe_detection(frame, holistic)
            t_mp = time.perf_counter() - t0

            # 3. Trích xuất landmark và xử lý qua FSignRealtimeProcessor (đo normalize & predict)
            keypoints_raw = extract_keypoints(results)
            res_dict = processor.process_keypoints(keypoints_raw, current_time=time.time())
            t_norm = res_dict['t_norm']
            t_predict = res_dict['t_predict']
            predicted_this_frame = res_dict['predicted_this_frame']

            # 4. Đo thời gian vẽ landmark và render UI OpenCV
            t0_draw = time.perf_counter()
            draw_styled_landmarks(image, results)

            # 4.1. Thanh hiển thị câu nhận diện (top)
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(
                image, ' '.join(res_dict['sentence']), (3, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
            )

            # 4.2. Thanh trạng thái Idle / Cooldown / Nhận diện (ngay dưới banner top)
            status_color = (0, 165, 255) if res_dict['is_idle'] else (0, 255, 0)
            cv2.putText(
                image, res_dict['status_text'], (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA
            )

            # 4.3. Badge hiển thị FPS thời gian thực lên màn hình (góc phải trên)
            current_fps_display = np.mean(fps_history) if len(fps_history) > 0 else 0.0
            cv2.putText(
                image, f"FPS: {current_fps_display:4.1f}", (480, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA
            )

            cv2.imshow('OpenCV Feed - FSign', image)
            t_draw_ui = time.perf_counter() - t0_draw

            # Tổng thời gian tính toán thuần
            t_compute = t_cap + t_mp + t_norm + t_predict + t_draw_ui

            # Lắng nghe phím 'q' để thoát (dùng waitKey(1) cho ứng dụng thời gian thực)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            t_frame_total = time.perf_counter() - t_frame_start
            instant_fps = 1.0 / max(t_frame_total, 1e-5)

            # Tích lũy vào rolling deque
            frame_count += 1
            fps_history.append(instant_fps)
            cap_times.append(t_cap * 1000.0)
            mp_times.append(t_mp * 1000.0)
            norm_times.append(t_norm * 1000.0)
            if predicted_this_frame:
                predict_times.append(t_predict * 1000.0)
            predict_called_history.append(1 if predicted_this_frame else 0)
            draw_times.append(t_draw_ui * 1000.0)
            compute_times.append(t_compute * 1000.0)

            # ─── IN BÁO CÁO TRUNG BÌNH ĐỘNG MỖI 30 FRAME ───
            if frame_count % window_size == 0:
                mean_fps = np.mean(fps_history)
                min_fps = np.min(fps_history)
                max_fps = np.max(fps_history)

                mean_cap = np.mean(cap_times)
                mean_mp = np.mean(mp_times)
                mean_norm = np.mean(norm_times)
                mean_pred = np.mean(predict_times) if len(predict_times) > 0 else 0.0
                predict_count = sum(predict_called_history)
                mean_draw = np.mean(draw_times)
                mean_compute = np.mean(compute_times)

                pct_cap = (mean_cap / mean_compute) * 100 if mean_compute > 0 else 0
                pct_mp = (mean_mp / mean_compute) * 100 if mean_compute > 0 else 0
                pct_norm = (mean_norm / mean_compute) * 100 if mean_compute > 0 else 0
                pct_pred = (mean_pred * (predict_count / window_size) / mean_compute) * 100 if mean_compute > 0 else 0
                pct_draw = (mean_draw / mean_compute) * 100 if mean_compute > 0 else 0

                log_block = (
                    f"\n[FRAME {frame_count:04d}] ─── HIỆU NĂNG THỜI GIAN THỰC (30 FRAMES GẦN NHẤT) ───\n"
                    f"  FPS Vòng lặp:   Trung bình: {mean_fps:5.1f} | Min: {min_fps:5.1f} | Max: {max_fps:5.1f}\n"
                    f"  Thời gian trung bình từng bước:\n"
                    f"    - Đọc Webcam (cap.read):            {mean_cap:6.2f} ms ({pct_cap:5.1f}%)\n"
                    f"    - MediaPipe Holistic:              {mean_mp:6.2f} ms ({pct_mp:5.1f}%)\n"
                    f"    - Chuẩn hóa (normalize_keypoints):  {mean_norm:6.2f} ms ({pct_norm:5.1f}%)\n"
                    f"    - Suy luận LSTM (model.predict):    {mean_pred:6.2f} ms ({pct_pred:5.1f}%) [gọi {predict_count}/{window_size} frames]\n"
                    f"    - Vẽ Landmark & OpenCV UI:          {mean_draw:6.2f} ms ({pct_draw:5.1f}%)\n"
                    f"    ─────────────────────────────────────────────────────────────────\n"
                    f"    Tổng thời gian xử lý thuần:         {mean_compute:6.2f} ms / frame (FPS trần lý thuyết: {1000.0/max(mean_compute, 1e-3):.1f})\n"
                    f"  Trạng thái: '{res_dict['status_text']}' | Buffer: {res_dict['buffer_len']}/60 | Idle: {res_dict['is_idle']}"
                )
                print(log_block)
                log_file.write(log_block + "\n")
                log_file.flush()

    total_session_sec = time.perf_counter() - session_start_time
    summary_block = (
        f"\n=================================================================\n"
        f"=== TỔNG KẾT PHIÊN ĐO HIỆU NĂNG REALTIME (MỤC C1) ===\n"
        f"  Tổng frames đã chạy:    {frame_count}\n"
        f"  Thời gian chạy:         {total_session_sec:.2f} giây\n"
        f"  FPS trung bình toàn bộ: {frame_count / max(total_session_sec, 1e-5):.2f} FPS\n"
        f"  Chi tiết log đã lưu tại: {log_file_path}\n"
        f"=================================================================\n"
    )
    print(summary_block)
    log_file.write(summary_block + "\n")
    log_file.close()

    cap.release()
    cv2.destroyAllWindows()

class TFLiteModelWrapper:
    """
    Wrapper đóng gói TFLite Interpreter để cung cấp giao diện .predict()
    tương thích hoàn toàn với tf.keras.Model mà không cần thay đổi bất kỳ logic nghiệp vụ nào.
    """
    def __init__(self, model_path, num_threads=4):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[Lỗi] Không tìm thấy file model TFLite tại: {model_path}")
        self.model_path = model_path
        self.num_threads = num_threads
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.in_idx = self.input_details[0]['index']
        self.out_idx = self.output_details[0]['index']

    def predict(self, input_data: np.ndarray, verbose=0) -> np.ndarray:
        data = np.asarray(input_data, dtype=np.float32)
        self.interpreter.set_tensor(self.in_idx, data)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.out_idx)

if __name__ == '__main__':
    print("Khởi tạo mô hình FSign Unified (TFLite Engine)...")
    tflite_model_path = os.path.join('Models', 'model_normalized_v1.tflite')
    if os.path.exists(tflite_model_path):
        print(f"Đang tải mô hình TFLite từ: {tflite_model_path}")
        model = TFLiteModelWrapper(tflite_model_path, num_threads=4)
        print(f"=> Tải mô hình TFLite thành công cho {len(actions)} nhãn!")
    else:
        raise FileNotFoundError(f"[Lỗi] Không tìm thấy file model TFLite tại {tflite_model_path}")

    run_realtime_detection(model)
