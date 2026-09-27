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
_INDEX_MCP_IDX = 5    # landmark 5 = INDEX_FINGER_MCP
_MIDDLE_MCP_IDX = 9   # landmark 9 = MIDDLE_FINGER_MCP
_MIDDLE_TIP_IDX = 12  # landmark 12 = MIDDLE_FINGER_TIP
_PINKY_MCP_IDX = 17   # landmark 17 = PINKY_MCP
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

    Chiến lược (Task A4.5 S_combined chốt):
      1. Toàn zeros → tay absent, trả zeros (không xử lý)
      2. scale = sqrt(||mcp9 - wrist||^2 + ||mcp17 - mcp5||^2) < epsilon → tọa độ degenerate,
         coi invalid, trả zeros (epsilon guard tránh NaN/Inf)
      3. Ngược lại: centering quanh wrist + chia cho scale (đường chéo mu bàn tay cố định)

    Args:
        kp_63: ndarray shape (63,)
    Returns:
        ndarray shape (63,), không bao giờ chứa NaN hoặc Inf
    """
    # Guard 1: tay absent
    if np.all(kp_63 == 0.0):
        return np.zeros(63, dtype=np.float64)

    wrist_slice = slice(_WRIST_IDX * 3, _WRIST_IDX * 3 + 3)
    mcp9_slice  = slice(_MIDDLE_MCP_IDX * 3, _MIDDLE_MCP_IDX * 3 + 3)
    mcp5_slice  = slice(_INDEX_MCP_IDX * 3, _INDEX_MCP_IDX * 3 + 3)
    mcp17_slice = slice(_PINKY_MCP_IDX * 3, _PINKY_MCP_IDX * 3 + 3)

    wrist = kp_63[wrist_slice].copy()
    mcp9  = kp_63[mcp9_slice].copy()
    mcp5  = kp_63[mcp5_slice].copy()
    mcp17 = kp_63[mcp17_slice].copy()

    d_len_sq = float(np.sum((mcp9 - wrist) ** 2))
    d_wid_sq = float(np.sum((mcp17 - mcp5) ** 2))
    scale = float(np.sqrt(d_len_sq + d_wid_sq))

    # Guard 2: scale gần 0 (tọa độ degenerate)
    if scale < _EPSILON:
        return np.zeros(63, dtype=np.float64)

    # Centering + scaling
    result = kp_63.reshape(21, 3).copy()  # (21, 3)
    result -= wrist                        # centering: wrist → origin
    result /= scale                        # scaling: bất biến tỉ lệ
    return result.flatten()


def normalize_keypoints(kp_126: np.ndarray, include_rel_wrist: bool = True, include_presence: bool = False) -> np.ndarray:
    """
    Normalize vector (126,) gồm left_hand(0:63) và right_hand(63:126).
    Áp dụng epsilon guard độc lập cho từng tay.
    Bổ sung vector tương đối 2 cổ tay (Wrist_RH - Wrist_LH, 3 chiều) -> tổng 129 chiều (Task A5.2).
    Bổ sung 2 chiều presence flag (LH present, RH present) -> tổng 131 chiều (Task E2).

    Args:
        kp_126: ndarray shape (126,) — tọa độ gốc MediaPipe
        include_rel_wrist: bool — nếu True, nối thêm 3 chiều tương đối (Wrist_RH - Wrist_LH)
        include_presence: bool — nếu True, nối thêm 2 chiều presence flag [lh_flag, rh_flag] (Task E2)
    Returns:
        ndarray shape (126,), (129,), hoặc (131,)
    """
    assert kp_126.shape == (126,), f"Expected (126,), got {kp_126.shape}"
    raw_lh = kp_126[:_HAND_DIM].astype(np.float64)
    raw_rh = kp_126[_HAND_DIM:].astype(np.float64)

    lh_norm = _normalize_one_hand(raw_lh)
    rh_norm = _normalize_one_hand(raw_rh)

    # Presence flags (Task E2): 1.0 nếu scale >= 1e-6 (tay được phát hiện), 0.0 nếu epsilon guard kích hoạt
    lh_present = not np.all(lh_norm == 0.0)
    rh_present = not np.all(rh_norm == 0.0)

    if not include_rel_wrist and not include_presence:
        return np.concatenate([lh_norm, rh_norm])

    # Tính vector tương đối giữa 2 cổ tay ở tọa độ thô: Wrist_RH - Wrist_LH
    # Chỉ tính khi cả 2 tay cùng xuất hiện; nếu 1 hoặc cả 2 tay absent -> vector 0
    if include_rel_wrist:
        if lh_present and rh_present:
            wrist_lh = raw_lh[:3]
            wrist_rh = raw_rh[:3]
            rel_wrist = wrist_rh - wrist_lh
        else:
            rel_wrist = np.zeros(3, dtype=np.float64)
        base = np.concatenate([lh_norm, rh_norm, rel_wrist])
    else:
        base = np.concatenate([lh_norm, rh_norm])

    if not include_presence:
        return base

    presence_flags = np.array([1.0 if lh_present else 0.0, 1.0 if rh_present else 0.0], dtype=np.float64)
    return np.concatenate([base, presence_flags])


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
