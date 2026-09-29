# -*- coding: utf-8 -*-

"""

RunModel.py ΓÇö FSign Real-time Sign Language Translator (Clean & Proportional HUD Edition)

========================================================================================

Nhß║¡n diß╗çn ng├┤n ngß╗» k├╜ hiß╗çu tiß║┐ng Viß╗çt thß╗¥i gian thß╗▒c qua Webcam hoß║╖c Video

sß╗¡ dß╗Ñng m├┤ h├¼nh FSign 159 classes v├á MediaPipe Holistic vß╗¢i Giao diß╗çn Chuß║⌐n Tß╗ë Lß╗ç.



T├¡nh n─âng n├óng cß║Ñp UI/UX:

- Giß╗» nguy├¬n tß╗ë lß╗ç khung h├¼nh (Aspect Ratio Preservation): Kh├┤ng bß╗ï gi├ún/m├⌐o khi ─æß╗òi k├¡ch th╞░ß╗¢c hoß║╖c Fullscreen.

- Giao diß╗çn Floating Cards g├│c bo tr├▓n (Rounded Dark Panels) v├┤ c├╣ng tinh tß║┐, dß╗à nh├¼n, dß╗à d├╣ng.

- Hiß╗ân thß╗ï Tiß║┐ng Viß╗çt c├│ dß║Ñu n├⌐t ─æß║╣p chuß║⌐n HD qua Font hß╗ç thß╗æng Windows.

- Hß╗ù trß╗ú To├án M├án H├¼nh (Fullscreen - Ph├¡m [F] / [F11]) & Cß╗¡a sß╗ò linh hoß║ít.

- Thanh Confidence progress bar m╞░ß╗út m├á + Bß╗Ö lß╗ìc Smooth Probability Filtering triß╗çt ti├¬u giß║¡t nhß║Ñp nh├íy.

- Toast Notifications phß║ún hß╗ôi tß╗⌐c th├¼ khi bß║Ñm ph├¡m ─æiß╗üu khiß╗ân.

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



# ── Protobuf Compatibility Shim ──────────────────────────────────────────
try:
    import google.protobuf.message_factory as mf
    if not hasattr(mf, 'GetMessageClass'):
        _factory = mf.MessageFactory()
        mf.GetMessageClass = _factory.GetPrototype
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

    Quß║ún l├╜ font hß╗ç thß╗æng Windows hß╗ù trß╗ú ─æß║ºy ─æß╗º tiß║┐ng Viß╗çt c├│ dß║Ñu vß╗¢i caching theo k├¡ch th╞░ß╗¢c.

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

    T├¡nh to├ín Bounding Box (x1, y1, x2, y2) cho b├án tay tß╗½ MediaPipe Landmarks.

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

    Vß║╜ khung vu├┤ng style g├│c c├┤ng nghß╗ç (Tech Corner Brackets) co gi├ún theo tß╗ë lß╗ç m├án h├¼nh.

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

    Vß║╜ c├íc khß╗¢p b├án tay vß╗¢i m├áu sß║»c neon co gi├ún k├¡ch th╞░ß╗¢c ─æiß╗âm nß╗æi.

    """

    thick = max(2, int(3 * scale))

    r_node = max(3, int(4 * scale))



    # Tay tr├íi: Neon Pink / Magenta

    if results.left_hand_landmarks:

        mp_drawing.draw_landmarks(

            image, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS,

            mp_drawing.DrawingSpec(color=(236, 72, 153), thickness=thick, circle_radius=r_node),

            mp_drawing.DrawingSpec(color=(244, 114, 182), thickness=thick, circle_radius=r_node - 1)

        )



    # Tay phß║úi: Neon Cyan / Blue

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

    Vß║╜ to├án bß╗Ö giao diß╗çn Floating HUD hiß╗çn ─æß║íi, gß╗ìn g├áng, bo tr├▓n g├│c, dß╗à nh├¼n & chuß║⌐n tß╗ë lß╗ç.

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



    # ΓöÇΓöÇ 1. TOP HEADER FLOATING CARDS ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

    # Left Pill: Logo & Status

    logo_str = "ΓÜí FSIGN AI"

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



        sentence_str = " ".join(sentence) if sentence else "(Sß║╡n s├áng nhß║¡n diß╗çn cß╗¡ chß╗ë...)"

        sent_color = (254, 240, 138, 255) if sentence else (148, 163, 184, 255)



        draw.text((sent_x1 + int(14 * scale), margin + int(13 * scale)), "Dß╗èCH:", font=font_badge, fill=(56, 189, 248, 255))

        draw.text((sent_x1 + int(65 * scale), margin + int(10 * scale)), sentence_str, font=font_main, fill=sent_color)



    # ΓöÇΓöÇ 2. BOTTOM FLOATING CONTROL & PREDICTION CARD ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

    card_w = w - (margin * 2)

    card_h = int(102 * scale)

    card_y1 = h - margin - card_h



    draw.rounded_rectangle((margin, card_y1, w - margin, h - margin),

                           radius=int(12 * scale), fill=(15, 23, 42, 235), outline=(51, 65, 85, 255), width=1)



    # Action Label & Result

    act_x = margin + int(20 * scale)

    act_y = card_y1 + int(12 * scale)



    draw.text((act_x, act_y + int(2 * scale)), "Cß╗¼ CHß╗ê Dß╗░ ─ÉO├üN:", font=font_sub, fill=(148, 163, 184, 255))



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

    conf_str = f"─Éß╗Ö tin cß║¡y: {confidence_pct:.1f}%"

    draw.text((bar_x, bar_y + int(18 * scale)), conf_str, font=font_small, fill=(148, 163, 184, 255))



    # Controls shortcuts footer

    controls_str = "[F] Fullscreen  |  [C] X├│a c├óu  |  [S] X╞░╞íng: " + ("Bß║¼T" if show_skeleton else "Tß║«T") + "  |  [SPACE] Tß║ím dß╗½ng  |  [ESC] Tho├ít"

    ctrl_bbox = draw.textbbox((0, 0), controls_str, font=font_small)

    ctrl_w = ctrl_bbox[2] - ctrl_bbox[0]

    draw.text((w - margin - ctrl_w - int(20 * scale), bar_y + int(18 * scale)), controls_str, font=font_small, fill=(148, 163, 184, 255))



    # ΓöÇΓöÇ 3. TOAST NOTIFICATION ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

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

    Hiß╗ân thß╗ï frame tr├¬n cß╗¡a sß╗ò OpenCV giß╗» nguy├¬n tß╗ë lß╗ç khung h├¼nh (Aspect Ratio),

    kh├┤ng bß╗ï m├⌐o hay gi├ún h├¼nh khi chuyß╗ân sang To├án M├án H├¼nh hoß║╖c thay ─æß╗òi k├¡ch th╞░ß╗¢c cß╗¡a sß╗ò.

    """

    try:

        rect = cv2.getWindowImageRect(win_name)

        if rect and rect[2] > 0 and rect[3] > 0:

            win_w, win_h = rect[2], rect[3]

            fh, fw, _ = frame.shape



            # T├¡nh to├ín scale giß╗» nguy├¬n tß╗ë lß╗ç

            scale = min(win_w / fw, win_h / fh)

            new_w = max(1, int(fw * scale))

            new_h = max(1, int(fh * scale))



            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)



            # Tß║ío canvas ─æen vß╗½a kh├¡t cß╗¡a sß╗ò

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

    Tß║úi m├┤ h├¼nh Keras .h5 v├á tß╗½ ─æiß╗ân nh├ún tiß║┐ng Viß╗çt t╞░╞íng ß╗⌐ng.

    """

    models_dir = SCRIPT_DIR / 'Models'
    release_dir = SCRIPT_DIR / 'release'

    model_159_path = models_dir / 'fsign_159classes.h5'
    if not model_159_path.exists():
        model_159_path = release_dir / 'fsign_159classes.h5'

    label_map_path = models_dir / 'label_map_159.json'
    if not label_map_path.exists():
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
    selected_model_path = model_159_path
    if not selected_model_path.exists():
        all_h5 = list(models_dir.glob('**/*.h5')) or list(SCRIPT_DIR.glob('**/*.h5'))
        if all_h5:
            selected_model_path = all_h5[0]
        else:
            raise FileNotFoundError("Không tìm thấy bất kỳ file model .h5 nào trong dự án!")

    print(f"=> Đang tải model từ: {selected_model_path}...")
    model = None
    try:
        model = load_model(str(selected_model_path), compile=False)
    except Exception as e:
        print(f"  [Thông báo] load_model trực tiếp gặp lỗi tương thích ({e})")
        print("  => Đang tự động giải mã cấu trúc model từ metadata H5 (Keras 3 -> Keras 2 adapter)...")
        import h5py
        from keras.models import model_from_json

        try:
            with h5py.File(str(selected_model_path), 'r') as f:
                if 'model_config' in f.attrs:
                    raw_cfg = f.attrs['model_config']
                    if isinstance(raw_cfg, bytes):
                        raw_cfg = raw_cfg.decode('utf-8')
                    # Tương thích ngược Keras 3 sang Keras 2 (tf.keras)
                    raw_cfg = raw_cfg.replace('"batch_shape":', '"batch_input_shape":')
                    raw_cfg = raw_cfg.replace(', "optional": false', '')
                    raw_cfg = raw_cfg.replace('"optional": false,', '')
                    raw_cfg = raw_cfg.replace('"optional": false', '')
                    model = model_from_json(raw_cfg)
                    model.load_weights(str(selected_model_path))
                    print("  [+] Đã tải cấu trúc và weights thành công qua H5 Metadata Adapter!")
        except Exception as e2:
            print(f"  [Adapter Error]: {e2}")

        if model is None:
            # Thử nạp theo kiến trúc FSign-159 Optimized (tanh + BatchNorm)
            try:
                from train_fsign159_optimized import build_optimized_model
                num_classes = len(actions_map) if actions_map else 159
                model = build_optimized_model(input_shape=(SEQUENCE_LENGTH, FEATURE_DIM), num_classes=num_classes)
                model.load_weights(str(selected_model_path))
                print("  [+] Đã nạp thành công theo kiến trúc FSign-159 Optimized (tanh + BatchNorm)!")
            except Exception as e3:
                print(f"  [Optimized architecture mismatch]: {e3}")

        if model is None:
            # Thử nạp theo kiến trúc FSign-159 DeepLSTM (relu)
            try:
                from train_fsign159 import build_fsign159_model
                num_classes = len(actions_map) if actions_map else 159
                model = build_fsign159_model(input_shape=(SEQUENCE_LENGTH, FEATURE_DIM), num_classes=num_classes)
                model.load_weights(str(selected_model_path))
                print("  [+] Đã nạp thành công theo kiến trúc FSign-159 DeepLSTM (relu)!")
            except Exception as e4:
                raise RuntimeError(f"Không thể nạp weights vào bất kỳ kiến trúc nào: {e4}")

    num_out = model.output_shape[-1]
    print(f"=> Tải model thành công! Số classes output: {num_out}")
    print(f"=> Số nhãn trong từ điển: {len(actions_map)}")

    # Assert kiểm tra chéo
    model_out_dim = getattr(model, 'output_shape', [None, None])[-1]
    if model_out_dim is not None:
        assert len(actions_map) == model_out_dim, (
            f"Label count mismatch: {len(actions_map)} labels vs {model_out_dim} model outputs"
        )
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

            print(f"\n[Lß╗ûI] Kh├┤ng t├¼m thß║Ñy file video: '{source}'")

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

                print(f"=> ─É├ú tß╗▒ ─æß╗Öng kß║┐t nß╗æi tß╗¢i Camera {alt_idx}!")

                idx = alt_idx

                break



    return cap, f"Webcam (Port {idx})"





def run_realtime_detection(source=0, threshold=0.50, start_fullscreen=False):

    model, actions_map = load_fsign_model_and_labels()



    cap, src_name = open_camera_or_video(source)

    if cap is None or not cap.isOpened():

        print("\n" + "!" * 65)

        print(f"[Lß╗ûI] Kh├┤ng thß╗â mß╗ƒ nguß╗ôn ph├ít: {src_name}")

        print("!" * 65 + "\n")

        return



    is_video_file = isinstance(source, str) and not source.isdigit()



    # C├ái ─æß║╖t ─æß╗Ö ph├ón giß║úi HD cao cho Webcam (1280x720)

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

    print(f"  - Nguß╗ôn ph├ít: {src_name}")

    print("  - Ph├¡m tß║»t: [F] Bß║¡t/Tß║»t To├án M├án H├¼nh | [C] X├│a c├óu | [S] Khung x╞░╞íng | [SPACE] Tß║ím dß╗½ng | [Q] Tho├ít")

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



                # Vß║╜ skeleton nß║┐u bß║¡t

                if show_skeleton:

                    draw_styled_landmarks(image, results, scale=scale)



                # Bounding Box g├│c c├┤ng nghß╗ç cho b├án tay

                l_bbox = get_hand_bbox(results.left_hand_landmarks, w, h)

                r_bbox = get_hand_bbox(results.right_hand_landmarks, w, h)



                if l_bbox:

                    image = draw_tech_corner_box(image, l_bbox, "TAY TR├üI", color_rgb=(236, 72, 153), scale=scale)

                if r_bbox:

                    image = draw_tech_corner_box(image, r_bbox, "TAY PHß║óI", color_rgb=(6, 182, 212), scale=scale)



                # Tr├¡ch xuß║Ñt keypoints

                keypoints = extract_keypoints(results)

                sequence.append(keypoints)

                sequence = sequence[-SEQUENCE_LENGTH:]



                current_action_text = "─Éang nhß║¡n dß║íng..."

                confidence_pct = 0.0



                if len(sequence) == SEQUENCE_LENGTH:

                    input_seq = np.expand_dims(sequence, axis=0)

                    res = model.predict(input_seq, verbose=0)[0]



                    # Lß╗ìc nhiß╗àu x├íc suß║Ñt (Prob Smoothing)

                    prob_history.append(res)

                    prob_history = prob_history[-5:]

                    smooth_res = np.mean(prob_history, axis=0)



                    pred_idx = int(np.argmax(smooth_res))

                    confidence_pct = float(smooth_res[pred_idx]) * 100

                    predictions.append(pred_idx)



                    # Kiß╗âm tra ß╗òn ─æß╗ïnh (10 frames gß║ºn nhß║Ñt)

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

                    current_action_text = f"─Éang thu thß║¡p dß╗» liß╗çu ({len(sequence)}/{SEQUENCE_LENGTH})"



            else:

                image = frame.copy()

                current_action_text = "─É├â Tß║áM Dß╗¬NG"

                confidence_pct = 0.0



            # ΓöÇΓöÇ Draw Responsive Clean HUD Overlay ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

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



            # Hiß╗ân thß╗ï giß╗» nguy├¬n tß╗ë lß╗ç khung h├¼nh (Preserve Aspect Ratio)

            preserve_aspect_ratio_display(window_name, output_img)



            delay = 30 if is_video_file else 10

            key = cv2.waitKey(delay) & 0xFF



            if key == ord('q') or key == 27:

                break

            elif key == ord('f') or key == 203:  # Ph├¡m 'F' hoß║╖c F11

                is_fullscreen = not is_fullscreen

                if is_fullscreen:

                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

                    toast_message = ("≡ƒûÑ∩╕Å CHß║╛ ─Éß╗ÿ TO├ÇN M├ÇN H├îNH: Bß║¼T", time.time() + 1.8)

                else:

                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

                    cv2.resizeWindow(window_name, 1280, 720)

                    toast_message = ("≡ƒûÑ∩╕Å CHß║╛ ─Éß╗ÿ Cß╗¼A Sß╗ö Lß╗ÜN (1280x720)", time.time() + 1.8)



            elif key == ord('c'):

                sentence.clear()

                predictions.clear()

                prob_history.clear()

                toast_message = ("≡ƒº╣ ─É├â X├ôA C├éU T├ìCH L┼¿Y!", time.time() + 1.8)

            elif key == ord('s'):

                show_skeleton = not show_skeleton

                state_str = "Bß║¼T" if show_skeleton else "Tß║«T"

                toast_message = (f"≡ƒæü∩╕Å HIß╗éN THß╗è KHUNG X╞»╞áNG: {state_str}", time.time() + 1.8)

            elif key == ord(' '):

                is_paused = not is_paused

                p_str = "Tß║áM Dß╗¬NG" if is_paused else "TIß║╛P Tß╗ñC"

                toast_message = (f"ΓÅ»∩╕Å ─É├â {p_str} NHß║¼N DIß╗åN", time.time() + 1.8)



    cap.release()

    cv2.destroyAllWindows()





def main():

    parser = argparse.ArgumentParser(description="Chß║íy nhß║¡n diß╗çn ng├┤n ngß╗» k├╜ hiß╗çu FSign (Proportional HUD Edition)")

    parser.add_argument('--source', type=str, default='0', help="Camera index (mß║╖c ─æß╗ïnh: 0)")

    parser.add_argument('--video', type=str, default=None, help="─É╞░ß╗¥ng dß║½n file video mp4")

    parser.add_argument('--label', type=str, default=None, help="T├¬n nh├ún muß╗æn test mß║½u (v├¡ dß╗Ñ: 'Ch├áo', 'Cß║úm ╞ín')")

    parser.add_argument('--random', action='store_true', help="Tß╗▒ ─æß╗Öng chß╗ìn ngß║½u nhi├¬n 1 video mß║½u ─æß╗â test")

    parser.add_argument('--fullscreen', action='store_true', help="Tß╗▒ ─æß╗Öng bß║¡t chß║┐ ─æß╗Ö To├án M├án H├¼nh ngay khi mß╗ƒ")

    parser.add_argument('--threshold', type=float, default=0.50, help="Ng╞░ß╗íng tin cß║¡y dß╗▒ ─æo├ín (mß║╖c ─æß╗ïnh: 0.50)")

    args = parser.parse_args()



    src = args.source



    if args.random:

        if DATASET_TRAIN_DIR.exists():

            all_vids = list(DATASET_TRAIN_DIR.glob('*/*.mp4'))

            if all_vids:

                src = str(random.choice(all_vids))

                print(f"=> ─É├ú chß╗ìn ngß║½u nhi├¬n video mß║½u: {src}")



    elif args.label:

        if DATASET_TRAIN_DIR.exists():

            matches = list(DATASET_TRAIN_DIR.glob(f"*{args.label}*/*.mp4"))

            if matches:

                src = str(random.choice(matches))

                print(f"=> ─É├ú t├¼m thß║Ñy video mß║½u cho nh├ún '{args.label}': {src}")

            else:

                print(f"[Cß║únh b├ío] Kh├┤ng t├¼m thß║Ñy nh├ún '{args.label}' trong dataset/train/")



    elif args.video:

        src = args.video



    run_realtime_detection(source=src, threshold=args.threshold, start_fullscreen=args.fullscreen)





if __name__ == '__main__':

    main()

