"""
preprocessing.py — FSign Shared Preprocessing Module
======================================================
Cung cấp các hàm dùng chung cho toàn bộ pipeline thu thập dữ liệu FSign:
  - Trích xuất landmark (raw và normalized)
  - Chuẩn hóa tọa độ per-hand với epsilon guard (không sinh NaN/Inf)
  - Nội suy tuyến tính để resample sequence
  - Sinh sequence ID theo nguồn (webcam_N / yt_N)

QUAN TRỌNG:
  - File này KHÔNG sửa bất kỳ logic nào trong CollectData.py hay
    ActionDetection.py — chỉ tái sử dụng và mở rộng.
  - normalize_keypoints() nhận vector (126,) thô, KHÔNG nhận MediaPipe
    results — để có thể áp dụng trực tiếp lên file .npy đã lưu.
"""

import os
import numpy as np
import cv2
import mediapipe as mp

# ─── MediaPipe setup (dùng Holistic giống CollectData.py) ───────────────────
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ─── Landmark index constants ────────────────────────────────────────────────
# MediaPipe Hands: 21 landmarks, mỗi landmark 3 chiều (x, y, z)
# Layout trong vector (126,): [left_hand(63)] + [right_hand(63)]
_WRIST_IDX = 0        # landmark 0 = WRIST
_MIDDLE_TIP_IDX = 12  # landmark 12 = MIDDLE_FINGER_TIP
_HAND_DIM = 21 * 3    # 63 chiều mỗi tay
_EPSILON = 1e-6       # ngưỡng epsilon guard tránh chia cho 0


# ─── Core detection (tái sử dụng logic từ CollectData.py) ────────────────────

def mediapipe_detection(image: np.ndarray, model) -> tuple:
    """
    Chạy MediaPipe Holistic trên 1 frame BGR.
    Tái sử dụng đúng logic từ CollectData.py (không viết lại).

    Returns:
        (image_bgr, results) — image đã được chuyển ngược về BGR
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


# ─── Keypoint extraction ──────────────────────────────────────────────────────

def extract_keypoints_raw(results) -> np.ndarray:
    """
    Trích landmark tay từ MediaPipe Holistic results, tọa độ gốc (chưa normalize).
    Giống hệt hàm extract_keypoints() trong CollectData.py và ActionDetection.py.

    Returns:
        ndarray shape (126,) — [left_hand(63), right_hand(63)]
        Tay absent → zeros(63)
    """
    lh = (np.array([[res.x, res.y, res.z]
                    for res in results.left_hand_landmarks.landmark]).flatten()
          if results.left_hand_landmarks else np.zeros(_HAND_DIM))
    rh = (np.array([[res.x, res.y, res.z]
                    for res in results.right_hand_landmarks.landmark]).flatten()
          if results.right_hand_landmarks else np.zeros(_HAND_DIM))
    return np.concatenate([lh, rh])


def extract_keypoints_normalized(results) -> np.ndarray:
    """
    Trích landmark raw rồi normalize ngay.
    Tiện dụng nhưng KHÔNG dùng trong collect_youtube_data.py
    (pipeline YouTube cần raw trước để resample, normalize sau).

    Returns:
        ndarray shape (126,) đã normalized
    """
    raw = extract_keypoints_raw(results)
    return normalize_keypoints(raw)


# ─── Normalization ────────────────────────────────────────────────────────────

def _normalize_one_hand(kp_63: np.ndarray) -> np.ndarray:
    """
    Normalize 1 tay (vector 63 chiều = 21 landmark × 3).

    Chiến lược:
      1. Toàn zeros → tay absent, trả zeros (không xử lý)
      2. scale = ||middle_tip - wrist|| < epsilon → tọa độ degenerate,
         coi invalid, trả zeros (epsilon guard tránh NaN/Inf)
      3. Ngược lại: centering quanh wrist + chia cho scale

    Args:
        kp_63: ndarray shape (63,)
    Returns:
        ndarray shape (63,), không bao giờ chứa NaN hoặc Inf
    """
    # Guard 1: tay absent
    if np.all(kp_63 == 0.0):
        return np.zeros(63, dtype=np.float64)

    wrist_slice = slice(_WRIST_IDX * 3, _WRIST_IDX * 3 + 3)
    mid_slice   = slice(_MIDDLE_TIP_IDX * 3, _MIDDLE_TIP_IDX * 3 + 3)

    wrist      = kp_63[wrist_slice].copy()
    middle_tip = kp_63[mid_slice].copy()
    scale = float(np.linalg.norm(middle_tip - wrist))

    # Guard 2: scale gần 0 (tọa độ degenerate)
    if scale < _EPSILON:
        return np.zeros(63, dtype=np.float64)

    # Centering + scaling
    result = kp_63.reshape(21, 3).copy()  # (21, 3)
    result -= wrist                        # centering: wrist → origin
    result /= scale                        # scaling: bất biến tỉ lệ
    return result.flatten()


def normalize_keypoints(kp_126: np.ndarray) -> np.ndarray:
    """
    Normalize vector (126,) gồm left_hand(0:63) và right_hand(63:126).
    Áp dụng epsilon guard độc lập cho từng tay.

    Args:
        kp_126: ndarray shape (126,) — tọa độ gốc MediaPipe
    Returns:
        ndarray shape (126,) — đã normalize, không NaN, không Inf
    """
    assert kp_126.shape == (126,), f"Expected (126,), got {kp_126.shape}"
    lh_norm = _normalize_one_hand(kp_126[:_HAND_DIM].astype(np.float64))
    rh_norm = _normalize_one_hand(kp_126[_HAND_DIM:].astype(np.float64))
    return np.concatenate([lh_norm, rh_norm])


# ─── Resampling ───────────────────────────────────────────────────────────────

def resample_sequence(frames, target_length: int) -> np.ndarray:
    """
    Nội suy tuyến tính danh sách N vector (126,) → ndarray (target_length, 126).
    Thực hiện trên tọa độ số (ndarray), KHÔNG phải trên video gốc.

    Args:
        frames: list hoặc ndarray shape (N, 126), N >= 1
        target_length: số frame đích (ví dụ 60)
    Returns:
        ndarray shape (target_length, 126), dtype float64
    """
    frames_arr = np.array(frames, dtype=np.float64)  # (N, 126)
    N, D = frames_arr.shape

    if N == target_length:
        return frames_arr

    x_old = np.linspace(0.0, 1.0, N)
    x_new = np.linspace(0.0, 1.0, target_length)

    resampled = np.zeros((target_length, D), dtype=np.float64)
    for d in range(D):
        resampled[:, d] = np.interp(x_new, x_old, frames_arr[:, d])
    return resampled


# ─── Sequence ID management ───────────────────────────────────────────────────

def get_next_sequence_id(data_path: str, label: str, source: str) -> str:
    """
    Tìm tên thư mục sequence tiếp theo chưa được dùng cho nhãn và nguồn cho trước.

    Quy ước đặt tên:
      source='webcam'  → 'webcam_0', 'webcam_1', ...
      source='youtube' → 'yt_0', 'yt_1', ...

    Args:
        data_path: thư mục gốc dataset (ví dụ 'Data_normalized')
        label:     tên nhãn (ví dụ 'xin chao')
        source:    'webcam' hoặc 'youtube'
    Returns:
        str — tên thư mục sequence kế tiếp (ví dụ 'yt_3')
    """
    if source == 'webcam':
        prefix = 'webcam_'
    elif source == 'youtube':
        prefix = 'yt_'
    else:
        raise ValueError(f"source phải là 'webcam' hoặc 'youtube', nhận được: '{source}'")

    label_dir = os.path.join(data_path, label)
    if not os.path.exists(label_dir):
        return f"{prefix}0"

    indices = []
    for d in os.listdir(label_dir):
        if os.path.isdir(os.path.join(label_dir, d)) and d.startswith(prefix):
            try:
                indices.append(int(d[len(prefix):]))
            except ValueError:
                pass  # bỏ qua thư mục đặt tên không theo chuẩn

    next_id = (max(indices) + 1) if indices else 0
    return f"{prefix}{next_id}"
