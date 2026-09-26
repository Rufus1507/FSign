import cv2
import numpy as np
import os
import time
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
        """
        if current_time is None:
            current_time = time.time()

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
                'buffer_len': len(self.sequence)
            }

        # ─── 2. KHI CÓ TAY (NOT EMPTY HAND) ───
        # Thoát Idle và đặt lại đồng hồ theo dõi rỗng tay
        self.empty_hand_start_time = None
        self.is_idle = False

        # CHỈ chuẩn hóa và đưa vào buffer sequence khi THẬT SỰ CÓ TAY
        keypoints_norm = normalize_keypoints(keypoints_raw)
        self.sequence.append(keypoints_norm)
        self.sequence = self.sequence[-self.sequence_length:]

        predicted_action = None
        confidence = 0.0

        if len(self.sequence) == self.sequence_length:
            self.status_text = "Dang nhan dien..."
            if self.model is not None:
                self.predict_call_count += 1
                res = self.model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
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
            'buffer_len': len(self.sequence)
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

    print("=> Bắt đầu nhận diện qua webcam. Nhấn 'q' để thoát.")
    with mp_hands.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Dự đoán landmark
            image, results = mediapipe_detection(frame, holistic)

            # Vẽ landmark
            draw_styled_landmarks(image, results)

            # Xử lý frame qua processor thời gian thực
            keypoints_raw = extract_keypoints(results)
            res_dict = processor.process_keypoints(keypoints_raw, current_time=time.time())

            # 1. Thanh hiển thị câu nhận diện (top)
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(
                image, ' '.join(res_dict['sentence']), (3, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
            )

            # 2. Thanh trạng thái Idle / Nhận diện (ngay dưới banner top)
            status_color = (0, 165, 255) if res_dict['is_idle'] else (0, 255, 0)
            cv2.putText(
                image, res_dict['status_text'], (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA
            )

            cv2.imshow('OpenCV Feed - FSign', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    print("Khởi tạo mô hình FSign Unified...")
    model = build_model(input_shape=(60, 126), num_classes=len(actions))

    weights_path = os.path.join('Models', 'model_normalized_v1.h5')
    if os.path.exists(weights_path):
        print(f"Đang tải trọng số từ: {weights_path}")
        model.load_weights(weights_path)
        print(f"=> Tải trọng số thành công cho {len(actions)} nhãn!")
    else:
        raise FileNotFoundError(f"[Lỗi] Không tìm thấy file model tại {weights_path}")

    run_realtime_detection(model)
