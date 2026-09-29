# -*- coding: utf-8 -*-
"""
RunModel_119.py — Chương trình nhận diện cử chỉ thời gian thực FSign (119 Nhãn Hợp Nhất)
========================================================================================
Tách biệt hoàn toàn với RunModel.py gốc (60 nhãn production):
  - Model suy luận: Models/baseline_119_tanh.tflite (119 classes, 129 chiều, TFLite Builtin ops)
  - Danh sách nhãn: Models/label_map_119.json (119 nhãn: 59 Webcam + 60 Video)
  - Assert chặn cứng: Kiểm tra bắt buộc model_input_dim == 129 VÀ num_classes == 119
  - Kiến trúc realtime: MediaPipe Holistic -> S_combined + Rel Wrist (129d) -> ThreadedCamera
"""

import cv2
import numpy as np
import os
import sys
import json
import time
import threading
from collections import deque, Counter
from matplotlib import pyplot as plt
import mediapipe as mp
import tensorflow as tf

# Chuyển working directory về thư mục chứa file script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from model_def_merged119 import build_model
from preprocessing import normalize_keypoints

# Cấu hình thời gian thực (Kế thừa từ chuẩn tối ưu Hạng mục D)
IDLE_TIME_THRESHOLD_SEC = 0.8  # Ngưỡng thời gian không thấy tay (giây) để coi là Idle
CONSENSUS_WINDOW = 10          # Số frame liên tiếp trong cửa sổ trượt consensus
COOLDOWN_TIME_SEC = 1.2        # Thời gian cooldown giữa 2 lần chèn từ vào câu

# Khởi tạo MediaPipe Holistic
mp_hands = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)


def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    )
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )


def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])


def load_actions(
    label_map_path=os.path.join('Models', 'label_map_119.json'),
    cache_path=os.path.join('Data_normalized_merged119', 'dataset_cache_119.npz'),
    data_dir='Data_normalized_merged119'
):
    """
    Lấy danh sách 119 nhãn hợp nhất đúng thứ tự index lúc train:
    1. Đọc trực tiếp từ Models/label_map_119.json (chính thức).
    2. Đọc từ dataset_cache_119.npz (cache nén).
    3. Fallback duyệt Data_normalized_merged119 nếu không có file cache.
    """
    if os.path.exists(label_map_path):
        try:
            with open(label_map_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'classes' in data and len(data['classes']) == 119:
                return np.array(data['classes'])
        except Exception:
            pass

    if os.path.exists(cache_path):
        try:
            c = np.load(cache_path, allow_pickle=True)
            id_to_label = c['id_to_label'].item()
            return np.array([id_to_label[i] for i in range(len(id_to_label))])
        except Exception:
            pass

    if os.path.exists(data_dir):
        dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        if len(dirs) == 119:
            return np.array(dirs)

    raise FileNotFoundError(
        f"Không tìm thấy nguồn dữ liệu 119 nhãn hợp lệ tại {label_map_path}, {cache_path}, hoặc {data_dir}"
    )


# Danh sách 119 câu cử chỉ nhận diện chuẩn
actions = load_actions()


class FSignRealtimeProcessor:
    """
    Bộ xử lý suy luận thời gian thực cho FSign 119 Nhãn:
    - Quản lý trạng thái nghỉ (Idle Detection) theo thời gian thực (0.8s).
    - Chuẩn hóa tọa độ per-hand kết hợp relative wrist (129 chiều).
    - Cửa sổ trượt đồng thuận consensus (10 frames) chống giật từ nhận diện sai.
    - Cơ chế Cooldown thời gian thực (1.2s).
    """
    def __init__(
        self,
        actions,
        model=None,
        idle_threshold_sec=IDLE_TIME_THRESHOLD_SEC,
        consensus_window=CONSENSUS_WINDOW,
        cooldown_sec=COOLDOWN_TIME_SEC,
        sequence_length=60,
        threshold=0.5
    ):
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

        # Cơ chế "Hold last valid frame" khi 1 tay bị mất landmark tạm thời
        self.last_valid_lh = None
        self.last_valid_rh = None
        self.last_lh_time = -100.0
        self.last_rh_time = -100.0
        self.hand_hold_timeout_sec = 0.4

    def process_keypoints(self, keypoints_raw: np.ndarray, current_time: float = None) -> dict:
        if current_time is None:
            current_time = time.time()

        t_norm = 0.0
        t_predict = 0.0
        predicted_this_frame = False

        is_empty_hand = np.all(keypoints_raw == 0)

        # 1. Xử lý trường hợp mất dấu cả 2 tay (Empty hand)
        if is_empty_hand:
            if self.empty_hand_start_time is None:
                self.empty_hand_start_time = current_time

            elapsed_empty = current_time - self.empty_hand_start_time
            if elapsed_empty >= self.idle_threshold_sec:
                if not self.is_idle:
                    self.is_idle = True
                    self.sequence.clear()
                    self.predictions.clear()
                    self.confidences.clear()
                    self.last_valid_lh = None
                    self.last_valid_rh = None
                self.status_text = f"Nghi / Idle ({elapsed_empty:.1f}s)"
            else:
                self.status_text = f"Cho Idle ({elapsed_empty:.1f}s / {self.idle_threshold_sec}s)"

            return {
                'is_idle': self.is_idle,
                'status_text': self.status_text,
                'predicted_action': None,
                'emitted_word': None,
                'confidence': 0.0,
                'sentence': list(self.sentence),
                'buffer_len': len(self.sequence),
                't_norm': 0.0,
                't_predict': 0.0,
                'predicted_this_frame': False
            }

        # 2. Có ít nhất một bàn tay xuất hiện
        self.empty_hand_start_time = None
        self.is_idle = False

        lh_raw = keypoints_raw[:63]
        rh_raw = keypoints_raw[63:]

        if not np.all(lh_raw == 0):
            self.last_valid_lh = lh_raw.copy()
            self.last_lh_time = current_time
        elif self.last_valid_lh is not None and (current_time - self.last_lh_time <= self.hand_hold_timeout_sec):
            lh_raw = self.last_valid_lh.copy()

        if not np.all(rh_raw == 0):
            self.last_valid_rh = rh_raw.copy()
            self.last_rh_time = current_time
        elif self.last_valid_rh is not None and (current_time - self.last_rh_time <= self.hand_hold_timeout_sec):
            rh_raw = self.last_valid_rh.copy()

        stitched_raw = np.concatenate([lh_raw, rh_raw])

        # Chuẩn hóa tọa độ 129 chiều (S_combined + Relative Wrist)
        t0_norm = time.perf_counter()
        normalized_frame = normalize_keypoints(stitched_raw, include_rel_wrist=True, include_presence=False)
        t_norm = time.perf_counter() - t0_norm

        self.sequence.append(normalized_frame)
        self.sequence = self.sequence[-self.sequence_length:]

        predicted_action = None
        emitted_word = None
        confidence = 0.0

        if len(self.sequence) == self.sequence_length and self.model is not None:
            input_tensor = np.expand_dims(self.sequence, axis=0).astype(np.float32)

            t0_pred = time.perf_counter()
            res = self.model.predict(input_tensor, verbose=0)[0]
            t_predict = time.perf_counter() - t0_pred
            predicted_this_frame = True
            self.predict_call_count += 1

            top_idx = int(np.argmax(res))
            confidence = float(res[top_idx])

            self.predictions.append(top_idx)
            self.confidences.append(confidence)
            self.predictions = self.predictions[-self.consensus_window:]
            self.confidences = self.confidences[-self.consensus_window:]

            if confidence > self.threshold:
                predicted_action = self.actions[top_idx]
                self.status_text = f"Nhan dien: {predicted_action} ({confidence*100:.1f}%)"

                is_consensus = False
                candidate_idx = None

                if len(self.predictions) >= self.consensus_window:
                    counts = Counter(self.predictions)
                    most_common_idx, freq = counts.most_common(1)[0]
                    if freq >= int(self.consensus_window * 0.7):
                        matched_confs = [c for p, c in zip(self.predictions, self.confidences) if p == most_common_idx]
                        mean_conf = float(np.mean(matched_confs))
                        if mean_conf > self.threshold:
                            is_consensus = True
                            candidate_idx = most_common_idx

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
    """Lớp đọc webcam đa luồng."""
    def __init__(self, src=0, custom_cap=None):
        self.src = src
        self.cap = custom_cap if custom_cap is not None else cv2.VideoCapture(self.src)
        self.lock = threading.Lock()
        self.frame_buffer = deque(maxlen=1)
        self.stopped = False
        self.thread = None

        if not self.cap.isOpened():
            raise RuntimeError(f"[Lỗi] Không thể mở thiết bị camera index {self.src}!")

        ret, initial_frame = self.cap.read()
        if ret and initial_frame is not None:
            self.frame_buffer.append(initial_frame)

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

    window_size = 30
    fps_history = deque(maxlen=window_size)
    frame_count = 0

    print("\n" + "=" * 80)
    print("  FSIGN REAL-TIME 119 NHÃN ĐANG HOẠT ĐỘNG")
    print("  Nhấn 'q' trên cửa sổ OpenCV Feed để kết thúc phiên.")
    print("=" * 80)

    with mp_hands.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cam.isOpened():
            t_frame_start = time.perf_counter()

            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(0.002)
                continue

            image, results = mediapipe_detection(frame, holistic)
            keypoints_raw = extract_keypoints(results)
            res_dict = processor.process_keypoints(keypoints_raw, current_time=time.time())

            draw_styled_landmarks(image, results)

            # Banner câu kết quả
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(
                image, ' '.join(res_dict['sentence']), (3, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
            )

            # Trạng thái
            status_color = (0, 165, 255) if res_dict['is_idle'] else (0, 255, 0)
            cv2.putText(
                image, res_dict['status_text'], (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA
            )

            # Badge FPS
            current_fps = np.mean(fps_history) if len(fps_history) > 0 else 0.0
            cv2.putText(
                image, f"FPS: {current_fps:4.1f} | 119 Classes", (420, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA
            )

            cv2.imshow('OpenCV Feed - FSign (119 Classes Unified)', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            t_frame_total = time.perf_counter() - t_frame_start
            instant_fps = 1.0 / max(t_frame_total, 1e-5)
            frame_count += 1
            fps_history.append(instant_fps)

    cam.stop()
    cv2.destroyAllWindows()


class TFLiteModelWrapper:
    """Wrapper đóng gói TFLite Interpreter cho pipeline 119 nhãn."""
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
    print("=" * 80)
    print("  KHỞI TẠO MÔ HÌNH FSIGN 119 NHÃN (TFLITE ENGINE - RUNMODEL_119)")
    print("=" * 80)

    tflite_model_path = os.path.join('Models', 'baseline_119_tanh.tflite')
    h5_model_path = os.path.join('Models', 'baseline_119_tanh.h5')

    if os.path.exists(tflite_model_path):
        print(f"=> Đang tải mô hình TFLite từ: {tflite_model_path}")
        model = TFLiteModelWrapper(tflite_model_path, num_threads=4)

        # ─── ASSERT CHẶN CỨNG BẢO VỆ PIPELINE 119 NHÃN (TASK MERGE-6) ────────
        model_input_dim = int(model.input_details[0]['shape'][-1])
        model_output_dim = int(model.output_details[0]['shape'][-1])
        expected_feature_dim = 129
        expected_num_classes = 119

        assert model_input_dim == expected_feature_dim, (
            f"Input feature dimension mismatch for 119 pipeline: expected {expected_feature_dim} features "
            f"(126 normalized S_combined + 3 relative wrist), but model expects {model_input_dim}"
        )
        assert model_output_dim == expected_num_classes, (
            f"Model output dimension mismatch: expected {expected_num_classes} classes, but model has {model_output_dim}"
        )
        assert len(actions) == expected_num_classes, (
            f"Label count mismatch for 119 pipeline: expected {expected_num_classes} labels, but found {len(actions)}"
        )
        print(f"=> Xác thực thành công 100%: Input Dim = {model_input_dim} (129), Số nhãn = {len(actions)} ({model_output_dim}) - KHỚP TUYỆT ĐỐI!")
        print(f"=> Mô hình 119 classes sẵn sàng hoạt động!\n")
    elif os.path.exists(h5_model_path):
        print(f"=> [FALLBACK] TFLite chưa có, tải mô hình Keras H5: {h5_model_path}")
        model = build_model(input_shape=(60, 129), num_classes=119)
        model.load_weights(h5_model_path)
        assert model.output_shape[-1] == 119, f"Kỳ vọng 119 classes, nhận {model.output_shape[-1]}"
        assert len(actions) == 119, f"Kỳ vọng 119 nhãn, nhận {len(actions)}"
        print(f"=> Xác thực H5 thành công 100%!")
    else:
        raise FileNotFoundError(f"[Lỗi] Không tìm thấy model tại {tflite_model_path} hoặc {h5_model_path}")

    run_realtime_detection(model)
