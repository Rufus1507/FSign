import cv2
import numpy as np
import os
import json
import time
import threading
from collections import deque, Counter
from matplotlib import pyplot as plt
import mediapipe as mp
import tensorflow as tf
from model_def import build_model
from preprocessing import normalize_keypoints

# Cấu hình thời gian thực (Mục D3: Tăng Idle Dropout Tolerance lên 0.8s)
IDLE_TIME_THRESHOLD_SEC = 0.8  # Ngưỡng thời gian không thấy tay (giây) để coi là Idle
CONSENSUS_WINDOW = 10          # Số frame liên tiếp trong cửa sổ trượt consensus
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

def load_actions(label_map_path='label_map.json', cache_path=os.path.join('Data_normalized', 'dataset_cache_129d.npz'), data_dir='Data_normalized'):
    """
    Lấy danh sách 60 nhãn chuẩn đúng thứ tự index lúc train:
    1. Đọc trực tiếp từ label_map.json (chính thức).
    2. Đọc từ dataset_cache_129d.npz (cache chuẩn 129 chiều).
    3. Fallback duyệt Data_normalized nếu không có file cache.
    """
    if os.path.exists(label_map_path):
        try:
            with open(label_map_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'classes' in data and len(data['classes']) > 0:
                return np.array(data['classes'])
        except Exception:
            pass
    if os.path.exists(cache_path):
        try:
            c = np.load(cache_path, allow_pickle=True)
            if 'actions' in c and len(c['actions']) > 0:
                return np.array(c['actions'])
        except Exception:
            pass
    if os.path.exists(data_dir):
        dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        if len(dirs) > 0:
            return np.array(dirs)
    raise FileNotFoundError(f"Không tìm thấy nguồn dữ liệu nhãn hợp lệ tại {label_map_path}, {cache_path}, hoặc {data_dir}")

# Danh sách 60 câu cử chỉ nhận diện chuẩn từ dataset
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
        self.confidences = []

        self.empty_hand_start_time = None
        self.is_idle = True
        self.last_action_time = 0.0
        self.status_text = "Dang cho / Idle"
        self.predict_call_count = 0

        # Cơ chế "Hold last valid frame" khi 1 tay bị mất landmark tạm thời (Task D3)
        self.last_valid_lh = None
        self.last_valid_rh = None
        self.last_lh_time = -100.0
        self.last_rh_time = -100.0
        self.hand_hold_timeout_sec = 0.4  # Tối đa 0.4s (12 frames) giữ tư thế tay khi che khuất ngắn hạn

    def process_keypoints(self, keypoints_raw: np.ndarray, current_time: float = None) -> dict:
        """
        Xử lý 1 vector keypoints thô (126,) tại thời điểm current_time.
        Bổ sung telemetry đo thời gian chuẩn hóa (t_norm) và suy luận (t_predict) - Mục C1.
        Tăng Idle Dropout Tolerance lên 0.8s + Hold Last Valid Hand khi che khuất - Mục D3.
        """
        if current_time is None:
            current_time = time.time()

        t_norm = 0.0
        t_predict = 0.0
        predicted_this_frame = False

        is_empty_hand = np.all(keypoints_raw == 0)

        # ─── 1. XỬ LÝ TRƯỜNG HỢP MẤT DẤU TAY HOÀN TOÀN CẢ 2 TAY (EMPTY HAND) ───
        # Tuyệt đối KHÔNG normalize hay append frame rỗng vào sequence buffer!
        if is_empty_hand:
            if self.empty_hand_start_time is None:
                self.empty_hand_start_time = current_time

            elapsed = current_time - self.empty_hand_start_time
            if elapsed >= self.idle_threshold_sec:
                # Quá ngưỡng thời gian (0.8s) -> Kích hoạt Idle thật sự, dọn sạch buffer
                self.is_idle = True
                self.status_text = f"Dang cho / Idle ({elapsed:.1f}s)"
                if len(self.sequence) > 0:
                    self.sequence.clear()
                if len(self.predictions) > 0:
                    self.predictions.clear()
                if len(self.confidences) > 0:
                    self.confidences.clear()
                self.last_valid_lh = None
                self.last_valid_rh = None
            else:
                # Dưới ngưỡng thời gian -> Tạm ngưng (Dropout ngắn), bảo toàn buffer, KHÔNG append frame rỗng
                self.status_text = f"Tam ngung / Cho tay ({elapsed:.2f}s)"

            return {
                'is_idle': self.is_idle,
                'status_text': self.status_text,
                'predicted_action': None,
                'emitted_word': None,
                'confidence': 0.0,
                'sentence': list(self.sentence),
                'buffer_len': len(self.sequence),
                't_norm': t_norm,
                't_predict': t_predict,
                'predicted_this_frame': predicted_this_frame
            }

        # ─── 2. KHI CÓ ÍT NHẤT 1 TAY HOẠT ĐỘNG (NOT EMPTY HAND) ───
        self.empty_hand_start_time = None
        self.is_idle = False

        # Cơ chế "Hold last valid frame" khi 1 tay bị mất landmark tạm thời (Task D3):
        # Giữ lại landmark tay bị che khuất ngắn hạn (<0.4s) thay vì để trống toàn 0
        lh = keypoints_raw[:63].copy()
        rh = keypoints_raw[63:].copy()
        lh_zero = np.all(lh == 0)
        rh_zero = np.all(rh == 0)

        # Cập nhật hoặc bù đắp tay trái
        if not lh_zero:
            self.last_valid_lh = lh.copy()
            self.last_lh_time = current_time
        elif self.last_valid_lh is not None and (current_time - self.last_lh_time) < self.hand_hold_timeout_sec:
            lh = self.last_valid_lh.copy()

        # Cập nhật hoặc bù đắp tay phải
        if not rh_zero:
            self.last_valid_rh = rh.copy()
            self.last_rh_time = current_time
        elif self.last_valid_rh is not None and (current_time - self.last_rh_time) < self.hand_hold_timeout_sec:
            rh = self.last_valid_rh.copy()

        effective_keypoints_raw = np.concatenate([lh, rh])

        # Chuẩn hóa và đưa vào buffer sequence
        t0_norm = time.perf_counter()
        keypoints_norm = normalize_keypoints(effective_keypoints_raw)
        t_norm = time.perf_counter() - t0_norm

        self.sequence.append(keypoints_norm)
        self.sequence = self.sequence[-self.sequence_length:]

        predicted_action = None
        confidence = 0.0
        emitted_word = None

        if len(self.sequence) == self.sequence_length:
            self.status_text = "Dang nhan dien..."
            if self.model is not None:
                self.predict_call_count += 1
                t0_pred = time.perf_counter()
                res = self.model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
                t_predict = time.perf_counter() - t0_pred
                predicted_this_frame = True

                pred_idx = np.argmax(res)
                confidence = float(res[pred_idx])
                self.predictions.append(pred_idx)
                self.confidences.append(confidence)

                # ─── TRIỂN KHAI CHÍNH THỨC BIẾN THỂ 1 (MAJORITY VOTE >= 8/10 + CONF TB > 0.5) ───
                # Thay thế triệt để consensus 100% bằng Majority Vote nhằm giải phóng "đứng hình" (Task D3)
                is_consensus = False
                candidate_idx = None

                if len(self.predictions) >= self.consensus_window:
                    window_preds = self.predictions[-self.consensus_window:]
                    window_confs = self.confidences[-self.consensus_window:]
                    c = Counter(window_preds)
                    top_idx, count = c.most_common(1)[0]
                    if count >= 8:
                        matched_confs = [window_confs[i] for i in range(self.consensus_window) if window_preds[i] == top_idx]
                        mean_conf = float(np.mean(matched_confs))
                        if mean_conf > self.threshold:
                            is_consensus = True
                            candidate_idx = top_idx

                # Cơ chế Cooldown theo thời gian thực (Mục B4):
                emitted_word = None
                if is_consensus and candidate_idx is not None:
                    predicted_action = self.actions[candidate_idx]
                    elapsed_cooldown = current_time - self.last_action_time
                    if elapsed_cooldown >= self.cooldown_sec:
                        emitted_word = predicted_action
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
            'emitted_word': emitted_word,
            'confidence': confidence,
            'sentence': list(self.sentence),
            'buffer_len': len(self.sequence),
            't_norm': t_norm,
            't_predict': t_predict,
            'predicted_this_frame': predicted_this_frame
        }

class ThreadedCamera:
    """
    Lớp đọc webcam đa luồng (Producer-Consumer pattern) - Mục C3:
    - Worker daemon thread liên tục chạy cap.read() ở tốc độ tối đa của camera.
    - Luôn lưu giữ đúng 1 frame mới nhất vào deque(maxlen=1), tự động loại bỏ frame cũ.
    - Sử dụng threading.Lock khi đọc/ghi để ngăn ngừa race condition tuyệt đối.
    - Consumer (main thread) gọi .read() lấy frame mới nhất tức thì (non-blocking, tiết kiệm ~8ms).
    - Dừng sạch (clean shutdown) khi gọi .stop() để tránh zombie thread.
    """
    def __init__(self, src=0, custom_cap=None):
        self.src = src
        if custom_cap is not None:
            self.cap = custom_cap
        else:
            self.cap = cv2.VideoCapture(self.src)

        self.lock = threading.Lock()
        self.frame_buffer = deque(maxlen=1)
        self.stopped = False
        self.thread = None

        if not self.cap.isOpened():
            raise RuntimeError(f"[Lỗi] Không thể mở thiết bị camera index {self.src}!")

        # Đọc frame khởi tạo đầu tiên để buffer luôn sẵn sàng ngay từ frame 0
        ret, initial_frame = self.cap.read()
        if ret and initial_frame is not None:
            self.frame_buffer.append(initial_frame)

        # Khởi động worker daemon thread
        self.thread = threading.Thread(target=self._capture_worker, daemon=True, name="ThreadedCameraWorker")
        self.thread.start()

    def _capture_worker(self):
        while not self.stopped:
            if not self.cap.isOpened():
                break
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.002)
                continue
            with self.lock:
                self.frame_buffer.append(frame)

    def read(self):
        with self.lock:
            if len(self.frame_buffer) > 0:
                return True, self.frame_buffer[-1].copy()
            else:
                return False, None

    def stop(self):
        self.stopped = True
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

    def isOpened(self):
        return self.cap is not None and self.cap.isOpened()

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

    try:
        cam = ThreadedCamera(src=0)
    except Exception as e:
        print(f"[Lỗi] Không thể mở webcam: {e}")
        return

    # Khởi tạo các buffer đo đếm hiệu năng thời gian thực (Mục C3)
    window_size = 30
    fps_history = deque(maxlen=window_size)
    cap_times = deque(maxlen=window_size)
    mp_times = deque(maxlen=window_size)
    norm_times = deque(maxlen=window_size)
    predict_times = deque(maxlen=window_size)
    predict_called_history = deque(maxlen=window_size)
    draw_times = deque(maxlen=window_size)
    compute_times = deque(maxlen=window_size)

    log_file_path = "benchmark_c3_threaded_fps.log"
    log_file = open(log_file_path, "w", encoding="utf-8")
    header_info = (
        "=================================================================\n"
        "=== FSIGN REAL-TIME FPS & LATENCY BENCHMARK — THREADED CAMERA (MỤC C3) ===\n"
        f"Thời gian bắt đầu: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        "================================================================="
    )
    print(header_info)
    log_file.write(header_info + "\n")
    log_file.flush()

    frame_count = 0
    session_start_time = time.perf_counter()

    print("=> Bắt đầu nhận diện qua webcam (Threaded Camera). Nhấn 'q' trên cửa sổ video để dừng.")
    with mp_hands.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cam.isOpened():
            t_frame_start = time.perf_counter()

            # 1. Đo thời gian đọc frame từ ThreadedCamera (non-blocking)
            t0 = time.perf_counter()
            ret, frame = cam.read()
            t_cap = time.perf_counter() - t0
            if not ret or frame is None:
                time.sleep(0.002)
                continue

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
        f"=== TỔNG KẾT PHIÊN ĐO HIỆU NĂNG REALTIME (MỤC C3 - THREADED CAMERA) ===\n"
        f"  Tổng frames đã chạy:    {frame_count}\n"
        f"  Thời gian chạy:         {total_session_sec:.2f} giây\n"
        f"  FPS trung bình toàn bộ: {frame_count / max(total_session_sec, 1e-5):.2f} FPS\n"
        f"  Chi tiết log đã lưu tại: {log_file_path}\n"
        f"=================================================================\n"
    )
    print(summary_block)
    log_file.write(summary_block + "\n")
    log_file.close()

    cam.stop()
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
        self.input_shape = self.input_details[0]['shape']
        self.output_shape = self.output_details[0]['shape']
        self.input_dim = int(self.input_shape[-1])
        self.output_dim = int(self.output_shape[-1])

    def predict(self, input_data: np.ndarray, verbose=0) -> np.ndarray:
        data = np.asarray(input_data, dtype=np.float32)
        self.interpreter.set_tensor(self.in_idx, data)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.out_idx)

if __name__ == '__main__':
    print("Khởi tạo mô hình FSign Unified (TFLite Engine)...")
    # Trỏ chính xác sang model V2 Tanh (128->64->32) 60 lớp 129 chiều thắng ở Hạng mục B
    tflite_model_path = os.path.join('Models', 'model_b3_v2_descending.tflite')
    if not os.path.exists(tflite_model_path):
        # Fallback thử bản v1 nếu chưa convert
        tflite_model_path = os.path.join('Models', 'model_normalized_v1.tflite')

    if os.path.exists(tflite_model_path):
        print(f"Đang tải mô hình TFLite từ: {tflite_model_path}")
        model = TFLiteModelWrapper(tflite_model_path, num_threads=4)

        # ─── ASSERT CHẶN CỨNG KIỂM TRA CHÉO (TASK C2) ────────────────────────
        model_input_dim = int(model.input_details[0]['shape'][-1])
        model_output_dim = int(model.output_details[0]['shape'][-1])
        expected_feature_dim = 129  # 126 landmark chuẩn hóa S_combined + 3 relative wrist

        assert model_input_dim == expected_feature_dim, (
            f"Input feature dimension mismatch: expected {expected_feature_dim} features "
            f"(126 normalized S_combined + 3 relative wrist), but model expects {model_input_dim}"
        )
        assert len(actions) == model_output_dim, (
            f"Label count mismatch: {len(actions)} labels vs {model_output_dim} model outputs"
        )
        print(f"=> Xác thực thành công: Input Dim = {model_input_dim} (129), Số nhãn = {len(actions)} ({model_output_dim}) - KHỚP 100%!")
        print(f"=> Tải mô hình TFLite sẵn sàng cho {len(actions)} nhãn!")
    else:
        raise FileNotFoundError(f"[Lỗi] Không tìm thấy file model TFLite tại {tflite_model_path}")

    run_realtime_detection(model)
