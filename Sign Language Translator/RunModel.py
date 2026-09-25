# -*- coding: utf-8 -*-
"""
RunModel.py — FSign Real-time Sign Language Translator (Clean & Proportional HUD Edition)
========================================================================================
Nhận diện ngôn ngữ ký hiệu tiếng Việt thời gian thực qua Webcam hoặc Video
sử dụng mô hình FSign 159 classes và MediaPipe Holistic với Giao diện Chuẩn Tỉ Lệ.

Tính năng nâng cấp UI/UX:
- Giữ nguyên tỉ lệ khung hình (Aspect Ratio Preservation): Không bị giãn/méo khi đổi kích thước hoặc Fullscreen.
- Giao diện Floating Cards góc bo tròn (Rounded Dark Panels) vô cùng tinh tế, dễ nhìn, dễ dùng.
- Hiển thị Tiếng Việt có dấu nét đẹp chuẩn HD qua Font hệ thống Windows.
- Hỗ trợ Toàn Màn Hình (Fullscreen - Phím [F] / [F11]) & Cửa sổ linh hoạt.
- Thanh Confidence progress bar mượt mà + Bộ lọc Smooth Probability Filtering triệt tiêu giật nhấp nháy.
- Toast Notifications phản hồi tức thì khi bấm phím điều khiển.
"""

import os
import sys
import json
import time
import argparse
import random
from pathlib import Path

# Fix Unicode UTF-8 output cho console Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ── Protobuf Compatibility Shim cho MediaPipe trên Windows ───────────────────
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
try:
    from google.protobuf import descriptor, symbol_database
    import google.protobuf.message_factory as mf

    if not hasattr(descriptor.FieldDescriptor, 'label'):
        descriptor.FieldDescriptor.label = property(lambda self: getattr(self, '_label', 1))

    if not hasattr(mf, 'GetMessageClass'):
        mf.GetMessageClass = lambda desc: symbol_database.Default().GetPrototype(desc)
except Exception:
    pass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp

import keras
from keras.models import load_model, Sequential
from keras.layers import Input, LSTM, Dense, Dropout

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATASET_TRAIN_DIR = BASE_DIR / 'dataset' / 'train'

SEQUENCE_LENGTH = 60
FEATURE_DIM = 126
HAND_DIM = 63

mp_hands = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


class FontManager:
    """
    Quản lý font hệ thống Windows hỗ trợ đầy đủ tiếng Việt có dấu với caching theo kích thước.
    """
    _fonts = {}

    @classmethod
    def get_font(cls, size=18, bold=False):
        size = max(10, int(size))
        key = (size, bold)
        if key in cls._fonts:
            return cls._fonts[key]

        font_candidates = [
            'arialbd.ttf' if bold else 'arial.ttf',
            'segoeuib.ttf' if bold else 'segoeui.ttf',
            'calibrib.ttf' if bold else 'calibri.ttf',
            'tahomabd.ttf' if bold else 'tahoma.ttf'
        ]

        font = None
        for font_name in font_candidates:
            try:
                font = ImageFont.truetype(font_name, size)
                break
            except Exception:
                continue

        if font is None:
            font = ImageFont.load_default()

        cls._fonts[key] = font
        return font


def get_hand_bbox(landmarks, img_w, img_h, padding=24):
    """
    Tính toán Bounding Box (x1, y1, x2, y2) cho bàn tay từ MediaPipe Landmarks.
    """
    if not landmarks:
        return None
    xs = [int(lm.x * img_w) for lm in landmarks.landmark]
    ys = [int(lm.y * img_h) for lm in landmarks.landmark]

    x1 = max(0, min(xs) - padding)
    y1 = max(0, min(ys) - padding)
    x2 = min(img_w, max(xs) + padding)
    y2 = min(img_h, max(ys) + padding)

    return (x1, y1, x2, y2)


def draw_tech_corner_box(image, bbox, label, color_rgb=(0, 240, 255), scale=1.0):
    """
    Vẽ khung vuông style góc công nghệ (Tech Corner Brackets) co giãn theo tỉ lệ màn hình.
    """
    if not bbox:
        return image

    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    corner_len = max(15, min(int(32 * scale), int(min(w, h) * 0.3)))
    thickness = max(2, int(3 * scale))
    bgr_color = (color_rgb[2], color_rgb[1], color_rgb[0])

    # Top-Left corner
    cv2.line(image, (x1, y1), (x1 + corner_len, y1), bgr_color, thickness)
    cv2.line(image, (x1, y1), (x1, y1 + corner_len), bgr_color, thickness)

    # Top-Right corner
    cv2.line(image, (x2, y1), (x2 - corner_len, y1), bgr_color, thickness)
    cv2.line(image, (x2, y1), (x2, y1 + corner_len), bgr_color, thickness)

    # Bottom-Left corner
    cv2.line(image, (x1, y2), (x1 + corner_len, y2), bgr_color, thickness)
    cv2.line(image, (x1, y2), (x1, y2 - corner_len), bgr_color, thickness)

    # Bottom-Right corner
    cv2.line(image, (x2, y2), (x2 - corner_len, y2), bgr_color, thickness)
    cv2.line(image, (x2, y2), (x2, y2 - corner_len), bgr_color, thickness)

    # Label Tag
    if label:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img, 'RGBA')
        font_size = max(12, int(14 * scale))
        font = FontManager.get_font(font_size, bold=True)

        tag_bg = (color_rgb[0], color_rgb[1], color_rgb[2], 220)
        tag_text_color = (255, 255, 255, 255)

        tag_h = int(22 * scale)
        t_box = draw.textbbox((x1, y1 - tag_h), f" {label} ", font=font)
        
        # Rounded tag box
        draw.rounded_rectangle((x1, y1 - tag_h - 2, t_box[2] + 4, y1 - 2), radius=4, fill=tag_bg)
        draw.text((x1 + 2, y1 - tag_h), f" {label} ", font=font, fill=tag_text_color)

        image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return image


def draw_styled_landmarks(image, results, scale=1.0):
    """
    Vẽ các khớp bàn tay với màu sắc neon co giãn kích thước điểm nối.
    """
    thick = max(2, int(3 * scale))
    r_node = max(3, int(4 * scale))

    # Tay trái: Neon Pink / Magenta
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(236, 72, 153), thickness=thick, circle_radius=r_node),
            mp_drawing.DrawingSpec(color=(244, 114, 182), thickness=thick, circle_radius=r_node - 1)
        )

    # Tay phải: Neon Cyan / Blue
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(6, 182, 212), thickness=thick, circle_radius=r_node),
            mp_drawing.DrawingSpec(color=(56, 189, 248), thickness=thick, circle_radius=r_node - 1)
        )


def render_fsign_hud(
    img_bgr,
    sentence,
    current_action_text,
    confidence_pct,
    fps,
    buffer_len,
    buffer_max,
    is_paused,
    is_fullscreen,
    toast_message,
    show_skeleton,
    src_name
):
    """
    Vẽ toàn bộ giao diện Floating HUD hiện đại, gọn gàng, bo tròn góc, dễ nhìn & chuẩn tỉ lệ.
    """
    h, w, _ = img_bgr.shape
    scale = max(0.70, min(h / 720.0, w / 1280.0))

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img, 'RGBA')

    font_title = FontManager.get_font(int(18 * scale), bold=True)
    font_main = FontManager.get_font(int(24 * scale), bold=True)
    font_sub = FontManager.get_font(int(16 * scale), bold=False)
    font_small = FontManager.get_font(int(13 * scale), bold=False)
    font_badge = FontManager.get_font(int(13 * scale), bold=True)

    margin = int(16 * scale)
    top_bar_h = int(48 * scale)

    # ── 1. TOP HEADER FLOATING CARDS ──────────────────────────────────────────
    # Left Pill: Logo & Status
    logo_str = "⚡ FSIGN AI"
    status_text = "PAUSED" if is_paused else ("FULLSCREEN" if is_fullscreen else "LIVE")
    status_bg = (239, 68, 68, 230) if is_paused else (34, 197, 94, 230)

    badge_w = int(210 * scale)
    draw.rounded_rectangle((margin, margin, margin + badge_w, margin + top_bar_h),
                           radius=int(10 * scale), fill=(15, 23, 42, 230), outline=(51, 65, 85, 255), width=1)

    draw.text((margin + int(14 * scale), margin + int(12 * scale)), logo_str, font=font_title, fill=(56, 189, 248, 255))

    dot_x = margin + int(125 * scale)
    dot_y = margin + int(19 * scale)
    dot_r = int(5 * scale)
    draw.ellipse((dot_x, dot_y, dot_x + dot_r * 2, dot_y + dot_r * 2), fill=status_bg)
    draw.text((dot_x + dot_r * 2 + int(6 * scale), margin + int(13 * scale)), status_text, font=font_badge, fill=(241, 245, 249, 255))

    # Right Pill: FPS & Buffer Metrics
    metrics_str = f"FPS: {fps:.0f}  |  Buffer: {buffer_len}/{buffer_max}"
    m_bbox = draw.textbbox((0, 0), metrics_str, font=font_sub)
    m_w = m_bbox[2] - m_bbox[0]
    m_badge_w = m_w + int(28 * scale)
    m_x1 = w - margin - m_badge_w

    draw.rounded_rectangle((m_x1, margin, w - margin, margin + top_bar_h),
                           radius=int(10 * scale), fill=(15, 23, 42, 230), outline=(51, 65, 85, 255), width=1)
    draw.text((m_x1 + int(14 * scale), margin + int(13 * scale)), metrics_str, font=font_sub, fill=(148, 163, 184, 255))

    # Center Pill: Translated Sentence Banner
    sent_x1 = margin + badge_w + int(12 * scale)
    sent_x2 = m_x1 - int(12 * scale)
    if sent_x2 > sent_x1 + int(180 * scale):
        draw.rounded_rectangle((sent_x1, margin, sent_x2, margin + top_bar_h),
                               radius=int(10 * scale), fill=(15, 23, 42, 230), outline=(51, 65, 85, 255), width=1)

        sentence_str = " ".join(sentence) if sentence else "(Sẵn sàng nhận diện cử chỉ...)"
        sent_color = (254, 240, 138, 255) if sentence else (148, 163, 184, 255)

        draw.text((sent_x1 + int(14 * scale), margin + int(13 * scale)), "DỊCH:", font=font_badge, fill=(56, 189, 248, 255))
        draw.text((sent_x1 + int(65 * scale), margin + int(10 * scale)), sentence_str, font=font_main, fill=sent_color)

    # ── 2. BOTTOM FLOATING CONTROL & PREDICTION CARD ────────────────────────
    card_w = w - (margin * 2)
    card_h = int(102 * scale)
    card_y1 = h - margin - card_h

    draw.rounded_rectangle((margin, card_y1, w - margin, h - margin),
                           radius=int(12 * scale), fill=(15, 23, 42, 235), outline=(51, 65, 85, 255), width=1)

    # Action Label & Result
    act_x = margin + int(20 * scale)
    act_y = card_y1 + int(12 * scale)

    draw.text((act_x, act_y + int(2 * scale)), "CỬ CHỈ DỰ ĐOÁN:", font=font_sub, fill=(148, 163, 184, 255))

    act_color = (255, 255, 255, 255)
    if confidence_pct >= 75.0:
        act_color = (74, 222, 128, 255)   # Green
    elif confidence_pct >= 50.0:
        act_color = (56, 189, 248, 255)   # Cyan
    else:
        act_color = (203, 213, 225, 255)  # Slate

    draw.text((act_x + int(150 * scale), act_y), current_action_text, font=font_main, fill=act_color)

    # Confidence Progress Bar
    bar_x = act_x
    bar_y = card_y1 + int(48 * scale)
    bar_w = card_w - int(40 * scale)
    bar_h = int(12 * scale)

    # Track Background
    draw.rounded_rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), radius=int(6 * scale), fill=(30, 41, 59, 255))

    # Fill Progress
    fill_w = int((confidence_pct / 100.0) * bar_w)
    if fill_w > int(4 * scale):
        bar_fill_color = (34, 197, 94, 255) if confidence_pct >= 75 else ((6, 182, 212, 255) if confidence_pct >= 50 else (245, 158, 11, 255))
        draw.rounded_rectangle((bar_x, bar_y, bar_x + fill_w, bar_y + bar_h), radius=int(6 * scale), fill=bar_fill_color)

    # Confidence percentage text
    conf_str = f"Độ tin cậy: {confidence_pct:.1f}%"
    draw.text((bar_x, bar_y + int(18 * scale)), conf_str, font=font_small, fill=(148, 163, 184, 255))

    # Controls shortcuts footer
    controls_str = "[F] Fullscreen  |  [C] Xóa câu  |  [S] Xương: " + ("BẬT" if show_skeleton else "TẮT") + "  |  [SPACE] Tạm dừng  |  [ESC] Thoát"
    ctrl_bbox = draw.textbbox((0, 0), controls_str, font=font_small)
    ctrl_w = ctrl_bbox[2] - ctrl_bbox[0]
    draw.text((w - margin - ctrl_w - int(20 * scale), bar_y + int(18 * scale)), controls_str, font=font_small, fill=(148, 163, 184, 255))

    # ── 3. TOAST NOTIFICATION ────────────────────────────────────────────────
    if toast_message:
        t_msg, t_expire = toast_message
        if time.time() < t_expire:
            t_font = FontManager.get_font(int(17 * scale), bold=True)
            t_bbox = draw.textbbox((0, 0), t_msg, font=t_font)
            t_w = t_bbox[2] - t_bbox[0]
            t_h = t_bbox[3] - t_bbox[1]
            t_x = (w - t_w) // 2
            t_y = margin + top_bar_h + int(16 * scale)

            pad_t_x = int(22 * scale)
            pad_t_y = int(8 * scale)
            draw.rounded_rectangle((t_x - pad_t_x, t_y - pad_t_y, t_x + t_w + pad_t_x, t_y + t_h + pad_t_y),
                                   radius=int(10 * scale), fill=(15, 23, 42, 245), outline=(56, 189, 248, 255), width=2)
            draw.text((t_x, t_y), t_msg, font=t_font, fill=(56, 189, 248, 255))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def preserve_aspect_ratio_display(win_name, frame):
    """
    Hiển thị frame trên cửa sổ OpenCV giữ nguyên tỉ lệ khung hình (Aspect Ratio),
    không bị méo hay giãn hình khi chuyển sang Toàn Màn Hình hoặc thay đổi kích thước cửa sổ.
    """
    try:
        rect = cv2.getWindowImageRect(win_name)
        if rect and rect[2] > 0 and rect[3] > 0:
            win_w, win_h = rect[2], rect[3]
            fh, fw, _ = frame.shape

            # Tính toán scale giữ nguyên tỉ lệ
            scale = min(win_w / fw, win_h / fh)
            new_w = max(1, int(fw * scale))
            new_h = max(1, int(fh * scale))

            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Tạo canvas đen vừa khít cửa sổ
            canvas = np.full((win_h, win_w, 3), (15, 23, 42), dtype=np.uint8)
            pad_x = (win_w - new_w) // 2
            pad_y = (win_h - new_h) // 2
            canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

            cv2.imshow(win_name, canvas)
            return
    except Exception:
        pass

    cv2.imshow(win_name, frame)


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def extract_keypoints(results):
    lh = (np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark], dtype=np.float32).flatten()
          if results.left_hand_landmarks else np.zeros(HAND_DIM, dtype=np.float32))
    rh = (np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark], dtype=np.float32).flatten()
          if results.right_hand_landmarks else np.zeros(HAND_DIM, dtype=np.float32))
    return np.concatenate([lh, rh])


def load_fsign_model_and_labels():
    """
    Tải mô hình Keras .h5 và từ điển nhãn tiếng Việt tương ứng.
    """
    release_dir = SCRIPT_DIR / 'release'
    model_159_path = release_dir / 'fsign_159classes.h5'
    model_legacy_path = release_dir / '94,58.h5'
    label_map_path = release_dir / 'label_map.json'

    if not label_map_path.exists():
        label_map_path = SCRIPT_DIR / 'label_map.json'

    # 1. Tải label map
    actions_map = {}
    if label_map_path.exists():
        with open(label_map_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'id_to_display' in data:
                actions_map = {int(k): v for k, v in data['id_to_display'].items()}
            elif 'id_to_label' in data:
                actions_map = {int(k): v for k, v in data['id_to_label'].items()}
            elif 'classes' in data:
                actions_map = {i: c for i, c in enumerate(data['classes'])}

    # 2. Tải model
    selected_model_path = model_159_path if model_159_path.exists() else model_legacy_path
    if not selected_model_path.exists():
        all_h5 = list(SCRIPT_DIR.glob('**/*.h5'))
        if all_h5:
            selected_model_path = all_h5[0]
        else:
            raise FileNotFoundError("Không tìm thấy bất kỳ file model .h5 nào trong dự án!")

    print(f"=> Đang tải model từ: {selected_model_path}...")
    try:
        model = load_model(str(selected_model_path), compile=False)
    except Exception as e:
        print(f"  [Cảnh báo] load_model gặp lỗi ({e}), đang dựng cấu trúc và nạp weights...")
        num_classes = len(actions_map) if actions_map else 159
        model = Sequential([
            Input(shape=(SEQUENCE_LENGTH, FEATURE_DIM)),
            LSTM(64, return_sequences=True, activation='relu'),
            LSTM(128, return_sequences=True, activation='relu'),
            LSTM(64, return_sequences=False, activation='relu'),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        model.load_weights(str(selected_model_path))

    num_out = model.output_shape[-1]
    print(f"=> Tải model thành công! Số classes output: {num_out}")
    print(f"=> Số nhãn trong từ điển: {len(actions_map)}")

    return model, actions_map


def resolve_video_path(video_arg):
    p = Path(video_arg)
    if p.exists() and p.is_file():
        return str(p.resolve())

    if DATASET_TRAIN_DIR.exists():
        direct_check = DATASET_TRAIN_DIR / video_arg
        if direct_check.exists() and direct_check.is_file():
            return str(direct_check.resolve())

        matches = list(DATASET_TRAIN_DIR.glob(f"**/{p.name}"))
        if matches:
            return str(matches[0].resolve())

    return None


def open_camera_or_video(source=0):
    if isinstance(source, str) and not source.isdigit():
        resolved = resolve_video_path(source)
        if resolved is None:
            print(f"\n[LỖI] Không tìm thấy file video: '{source}'")
            return None, source

        cap = cv2.VideoCapture(resolved)
        return cap, resolved

    idx = int(source)
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)

    if not cap.isOpened() and idx == 0:
        for alt_idx in [1, 2]:
            cap = cv2.VideoCapture(alt_idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                print(f"=> Đã tự động kết nối tới Camera {alt_idx}!")
                idx = alt_idx
                break

    return cap, f"Webcam (Port {idx})"


def run_realtime_detection(source=0, threshold=0.50, start_fullscreen=False):
    model, actions_map = load_fsign_model_and_labels()

    cap, src_name = open_camera_or_video(source)
    if cap is None or not cap.isOpened():
        print("\n" + "!" * 65)
        print(f"[LỖI] Không thể mở nguồn phát: {src_name}")
        print("!" * 65 + "\n")
        return

    is_video_file = isinstance(source, str) and not source.isdigit()

    # Cài đặt độ phân giải HD cao cho Webcam (1280x720)
    if not is_video_file:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = 'FSign - Real-time Sign Language Translator'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    is_fullscreen = start_fullscreen
    if is_fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(window_name, 1280, 720)

    print("\n" + "=" * 65)
    print("  FSIGN REAL-TIME TRANSLATOR (CLEAN & PROPORTIONAL HUD EDITION)")
    print(f"  - Nguồn phát: {src_name}")
    print("  - Phím tắt: [F] Bật/Tắt Toàn Màn Hình | [C] Xóa câu | [S] Khung xương | [SPACE] Tạm dừng | [Q] Thoát")
    print("=" * 65 + "\n")

    sequence = []
    sentence = []
    predictions = []
    prob_history = []

    fps = 30.0
    prev_frame_time = time.time()
    show_skeleton = True
    is_paused = False
    toast_message = None

    with mp_hands.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as holistic:
        while cap.isOpened():
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, curr_time - prev_frame_time))
            prev_frame_time = curr_time

            ret, frame = cap.read()
            if not ret or frame is None:
                if is_video_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.05)
                    continue
                break

            if not is_video_file:
                frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape
            scale = max(0.70, min(h / 720.0, w / 1280.0))

            if not is_paused:
                image, results = mediapipe_detection(frame, holistic)

                # Vẽ skeleton nếu bật
                if show_skeleton:
                    draw_styled_landmarks(image, results, scale=scale)

                # Bounding Box góc công nghệ cho bàn tay
                l_bbox = get_hand_bbox(results.left_hand_landmarks, w, h)
                r_bbox = get_hand_bbox(results.right_hand_landmarks, w, h)

                if l_bbox:
                    image = draw_tech_corner_box(image, l_bbox, "TAY TRÁI", color_rgb=(236, 72, 153), scale=scale)
                if r_bbox:
                    image = draw_tech_corner_box(image, r_bbox, "TAY PHẢI", color_rgb=(6, 182, 212), scale=scale)

                # Trích xuất keypoints
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-SEQUENCE_LENGTH:]

                current_action_text = "Đang nhận dạng..."
                confidence_pct = 0.0

                if len(sequence) == SEQUENCE_LENGTH:
                    input_seq = np.expand_dims(sequence, axis=0)
                    res = model.predict(input_seq, verbose=0)[0]

                    # Lọc nhiễu xác suất (Prob Smoothing)
                    prob_history.append(res)
                    prob_history = prob_history[-5:]
                    smooth_res = np.mean(prob_history, axis=0)

                    pred_idx = int(np.argmax(smooth_res))
                    confidence_pct = float(smooth_res[pred_idx]) * 100
                    predictions.append(pred_idx)

                    # Kiểm tra ổn định (10 frames gần nhất)
                    recent_preds = predictions[-10:]
                    if recent_preds.count(pred_idx) >= 6 and smooth_res[pred_idx] > threshold:
                        predicted_label = actions_map.get(pred_idx, f"Class {pred_idx}")
                        current_action_text = predicted_label

                        if len(sentence) > 0:
                            if predicted_label != sentence[-1]:
                                sentence.append(predicted_label)
                        else:
                            sentence.append(predicted_label)

                    if len(sentence) > 7:
                        sentence = sentence[-7:]
                else:
                    current_action_text = f"Đang thu thập dữ liệu ({len(sequence)}/{SEQUENCE_LENGTH})"

            else:
                image = frame.copy()
                current_action_text = "ĐÃ TẠM DỪNG"
                confidence_pct = 0.0

            # ── Draw Responsive Clean HUD Overlay ─────────────────────────────
            output_img = render_fsign_hud(
                image,
                sentence=sentence,
                current_action_text=current_action_text,
                confidence_pct=confidence_pct,
                fps=fps,
                buffer_len=len(sequence),
                buffer_max=SEQUENCE_LENGTH,
                is_paused=is_paused,
                is_fullscreen=is_fullscreen,
                toast_message=toast_message,
                show_skeleton=show_skeleton,
                src_name=src_name
            )

            # Hiển thị giữ nguyên tỉ lệ khung hình (Preserve Aspect Ratio)
            preserve_aspect_ratio_display(window_name, output_img)

            delay = 30 if is_video_file else 10
            key = cv2.waitKey(delay) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord('f') or key == 203:  # Phím 'F' hoặc F11
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    toast_message = ("🖥️ CHẾ ĐỘ TOÀN MÀN HÌNH: BẬT", time.time() + 1.8)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, 1280, 720)
                    toast_message = ("🖥️ CHẾ ĐỘ CỬA SỔ LỚN (1280x720)", time.time() + 1.8)

            elif key == ord('c'):
                sentence.clear()
                predictions.clear()
                prob_history.clear()
                toast_message = ("🧹 ĐÃ XÓA CÂU TÍCH LŨY!", time.time() + 1.8)
            elif key == ord('s'):
                show_skeleton = not show_skeleton
                state_str = "BẬT" if show_skeleton else "TẮT"
                toast_message = (f"👁️ HIỂN THỊ KHUNG XƯƠNG: {state_str}", time.time() + 1.8)
            elif key == ord(' '):
                is_paused = not is_paused
                p_str = "TẠM DỪNG" if is_paused else "TIẾP TỤC"
                toast_message = (f"⏯️ ĐÃ {p_str} NHẬN DIỆN", time.time() + 1.8)

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Chạy nhận diện ngôn ngữ ký hiệu FSign (Proportional HUD Edition)")
    parser.add_argument('--source', type=str, default='0', help="Camera index (mặc định: 0)")
    parser.add_argument('--video', type=str, default=None, help="Đường dẫn file video mp4")
    parser.add_argument('--label', type=str, default=None, help="Tên nhãn muốn test mẫu (ví dụ: 'Chào', 'Cảm ơn')")
    parser.add_argument('--random', action='store_true', help="Tự động chọn ngẫu nhiên 1 video mẫu để test")
    parser.add_argument('--fullscreen', action='store_true', help="Tự động bật chế độ Toàn Màn Hình ngay khi mở")
    parser.add_argument('--threshold', type=float, default=0.50, help="Ngưỡng tin cậy dự đoán (mặc định: 0.50)")
    args = parser.parse_args()

    src = args.source

    if args.random:
        if DATASET_TRAIN_DIR.exists():
            all_vids = list(DATASET_TRAIN_DIR.glob('*/*.mp4'))
            if all_vids:
                src = str(random.choice(all_vids))
                print(f"=> Đã chọn ngẫu nhiên video mẫu: {src}")

    elif args.label:
        if DATASET_TRAIN_DIR.exists():
            matches = list(DATASET_TRAIN_DIR.glob(f"*{args.label}*/*.mp4"))
            if matches:
                src = str(random.choice(matches))
                print(f"=> Đã tìm thấy video mẫu cho nhãn '{args.label}': {src}")
            else:
                print(f"[Cảnh báo] Không tìm thấy nhãn '{args.label}' trong dataset/train/")

    elif args.video:
        src = args.video

    run_realtime_detection(source=src, threshold=args.threshold, start_fullscreen=args.fullscreen)


if __name__ == '__main__':
    main()
