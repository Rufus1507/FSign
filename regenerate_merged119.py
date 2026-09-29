# -*- coding: utf-8 -*-
"""
regenerate_merged119.py — Pipeline Chuẩn hóa & Regenerate Dataset Hợp Nhất 119 Nhãn
==================================================================================
Thực hiện toàn diện Task MERGE-4:
1. Chuẩn hóa 59 nhãn webcam từ 'Sign Language Translator/Data/' (3,540 sequences).
2. Trích xuất MediaPipe & Chuẩn hóa 60 nhãn video từ 'dataset/train/' (3,300 sequences).
3. Áp dụng chuẩn S_combined + Relative Wrist (129 chiều, KHÔNG dùng presence flag).
4. Lưu vào thư mục riêng 'Sign Language Translator/Data_normalized_merged119/'.
5. Xác minh toàn vẹn (0 lỗi shape, 0 NaN/Inf, wrist centering 100%).
6. Tạo từ điển 'label_map_119.json' độc lập cho tập 119 nhãn.
7. Thống kê phân bố cân bằng dữ liệu (webcam vs video).
"""

import os
import sys
import time
import json
import csv
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# Fix Unicode UTF-8 output cho console Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import cv2
import numpy as np
import mediapipe as mp

# Nạp module preprocessing
BASE_DIR = Path(__file__).resolve().parent
SLT_DIR = BASE_DIR / "Sign Language Translator"
sys.path.insert(0, str(SLT_DIR))

from preprocessing import normalize_keypoints, resample_sequence

SEQUENCE_LENGTH = 60
FEATURE_DIM_RAW = 126
FEATURE_DIM_NORM = 129
HAND_DIM = 63

# ── 1. Danh sách 59 nhãn Webcam (loại trừ 'cam on' vì thay bằng 'Cảm ơn') ───
WEBCAM_CLASSES = [
    'ban dang lam gi', 'ban di dau the', 'ban hieu ngon ngu ky hieu khong',
    'ban hoc lop may', 'ban khoe khong', 'ban muon gio roi',
    'ban phai canh giac', 'ban ten la gi', 'ban tien bo day',
    'ban trong cau co the', 'bo me toi cung la nguoi Diec',
    'cai nay bao nhieu tien', 'cai nay la cai gi', 'cap cuu',
    'chuc mung', 'chung toi giao tiep voi nhau bang ngon ngu ky hieu',
    'con yeu me', 'cong viec cua ban la gi', 'hen gap lai cac ban',
    'mon nay khong ngon', 'toi bi chong mat', 'toi bi cuop',
    'toi bi dau dau', 'toi bi dau hong', 'toi bi ket xe',
    'toi bi lac', 'toi bi phan biet doi xu', 'toi cam thay rat hoi hop',
    'toi cam thay rat vui', 'toi can an sang', 'toi can di ve sinh',
    'toi can gap bac si', 'toi can phien dich', 'toi can thuoc',
    'toi dang an sang', 'toi dang buon', 'toi dang o ben xe',
    'toi dang o cong vien', 'toi dang phai cach ly', 'toi dang phan van',
    'toi di sieu thi', 'toi di toi Ha Noi', 'toi doc kem',
    'toi khoi benh roi', 'toi khong dem theo tien', 'toi khong hieu',
    'toi khong quan tam', 'toi la hoc sinh', 'toi la nguoi Diec',
    'toi la tho theu', 'toi lam viec o cua hang', 'toi nham dia chi',
    'toi song o Ha Noi', 'toi thay doi bung', 'toi thay nho ban',
    'toi thich an mi', 'toi thich phim truyen', 'toi viet kem',
    'xin chao'
]

# ── 2. Danh sách 60 nhãn Video đạt chuẩn lọc (Task MERGE-3) ──────────────────
VIDEO_CLASSES = [
    "An ủi", "Biết", "Biếu tặng", "Băn khoăn", "Bệnh viện",
    "Chiều", "Cho", "Chào", "Chân", "Chạy",
    "Chậm lại", "Con gấu", "Cá", "Cám dỗ", "Cảm ơn",
    "Cứu", "Dạy dỗ", "Dễ", "Giúp", "Hy sinh",
    "Hâm mộ", "Họ", "Khai báo", "Khóc", "Kết hôn",
    "Lây bệnh", "Mua", "Mời vào", "Nghe", "Nghỉ ngơi",
    "Nhà", "Nhìn", "Nhớ", "Nói", "Nói xấu",
    "Nôn ói", "Nặng", "Phía sau", "Phạt", "Phỏng vấn",
    "Rau", "Rẽ trái", "San sẻ", "Sử dụng", "Thức dậy",
    "Trưa", "Trường học", "Tôi", "Xa", "Xin phép",
    "Xuất viện", "Xúc động", "Áp dụng", "Ô tô", "Ăn",
    "Ăn mừng", "Đi", "Đâu", "Đầu", "Đồng ý"
]

# Mapping tên tiếng Việt có dấu cho 59 nhãn webcam
DISPLAY_MAP = {
    'ban dang lam gi': 'Bạn đang làm gì?',
    'ban di dau the': 'Bạn đi đâu thế?',
    'ban hieu ngon ngu ky hieu khong': 'Bạn hiểu ngôn ngữ ký hiệu không?',
    'ban hoc lop may': 'Bạn học lớp mấy?',
    'ban khoe khong': 'Bạn khỏe không?',
    'ban muon gio roi': 'Bạn muộn giờ rồi',
    'ban phai canh giac': 'Bạn phải cảnh giác',
    'ban ten la gi': 'Bạn tên là gì?',
    'ban tien bo day': 'Bạn tiến bộ đấy',
    'ban trong cau co the': 'Bạn trông cáu cọ thế',
    'bo me toi cung la nguoi Diec': 'Bố mẹ tôi cũng là người Điếc',
    'cai nay bao nhieu tien': 'Cái này bao nhiêu tiền?',
    'cai nay la cai gi': 'Cái này là cái gì?',
    'cap cuu': 'Cấp cứu',
    'chuc mung': 'Chúc mừng',
    'chung toi giao tiep voi nhau bang ngon ngu ky hieu': 'Chúng tôi giao tiếp với nhau bằng ngôn ngữ ký hiệu',
    'con yeu me': 'Con yêu mẹ',
    'cong viec cua ban la gi': 'Công việc của bạn là gì?',
    'hen gap lai cac ban': 'Hẹn gặp lại các bạn',
    'mon nay khong ngon': 'Món này không ngon',
    'toi bi chong mat': 'Tôi bị chóng mặt',
    'toi bi cuop': 'Tôi bị cướp',
    'toi bi dau dau': 'Tôi bị đau đầu',
    'toi bi dau hong': 'Tôi bị đau họng',
    'toi bi ket xe': 'Tôi bị kẹt xe',
    'toi bi lac': 'Tôi bị lạc',
    'toi bi phan biet doi xu': 'Tôi bị phân biệt đối xử',
    'toi cam thay rat hoi hop': 'Tôi cảm thấy rất hồi hộp',
    'toi cam thay rat vui': 'Tôi cảm thấy rất vui',
    'toi can an sang': 'Tôi cần ăn sáng',
    'toi can di ve sinh': 'Tôi cần đi vệ sinh',
    'toi can gap bac si': 'Tôi cần gặp bác sĩ',
    'toi can phien dich': 'Tôi cần phiên dịch',
    'toi can thuoc': 'Tôi cần thuốc',
    'toi dang an sang': 'Tôi đang ăn sáng',
    'toi dang buon': 'Tôi đang buồn',
    'toi dang o ben xe': 'Tôi đang ở bến xe',
    'toi dang o cong vien': 'Tôi đang ở công viên',
    'toi dang phai cach ly': 'Tôi đang phải cách ly',
    'toi dang phan van': 'Tôi đang phân vân',
    'toi di sieu thi': 'Tôi đi siêu thị',
    'toi di toi Ha Noi': 'Tôi đi tới Hà Nội',
    'toi doc kem': 'Tôi đọc kém',
    'toi khoi benh roi': 'Tôi khỏi bệnh rồi',
    'toi khong dem theo tien': 'Tôi không đem theo tiền',
    'toi khong hieu': 'Tôi không hiểu',
    'toi khong quan tam': 'Tôi không quan tam',
    'toi la hoc sinh': 'Tôi là học sinh',
    'toi la nguoi Diec': 'Tôi là người Điếc',
    'toi la tho theu': 'Tôi là thợ thêu',
    'toi lam viec o cua hang': 'Tôi làm việc ở cửa hàng',
    'toi nham dia chi': 'Tôi nhầm địa chỉ',
    'toi song o Ha Noi': 'Tôi sống ở Hà Nội',
    'toi thay doi bung': 'Tôi thấy đói bụng',
    'toi thay nho ban': 'Tôi thấy nhớ bạn',
    'toi thich an mi': 'Tôi thích ăn mì',
    'toi thich phim truyen': 'Tôi thích phim truyện',
    'toi viet kem': 'Tôi viết kém',
    'xin chao': 'Xin chào'
}


# ── 3. Hàm xử lý 1 video bằng MediaPipe ──────────────────────────────────────
def extract_single_video(video_path, holistic):
    """
    Đọc 1 video mp4, trích xuất raw keypoints (126,) cho từng frame,
    resample về đúng 60 frames, sau đó chuẩn hóa từng frame bằng normalize_keypoints.
    Returns: ndarray shape (60, 129) hoặc None nếu lỗi
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    raw_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)

        lh = (np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark], dtype=np.float32).flatten()
              if results.left_hand_landmarks else np.zeros(HAND_DIM, dtype=np.float32))
        rh = (np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark], dtype=np.float32).flatten()
              if results.right_hand_landmarks else np.zeros(HAND_DIM, dtype=np.float32))

        keypoints = np.concatenate([lh, rh])
        raw_frames.append(keypoints)

    cap.release()

    if not raw_frames:
        return None

    # Resample thành 60 frames (126 chiều raw)
    seq_60_raw = resample_sequence(raw_frames, target_length=SEQUENCE_LENGTH)

    # Chuẩn hóa từng frame thành 129 chiều (S_combined + relative wrist)
    seq_60_norm = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM_NORM), dtype=np.float32)
    for i in range(SEQUENCE_LENGTH):
        norm_frame = normalize_keypoints(seq_60_raw[i], include_rel_wrist=True, include_presence=False)
        seq_60_norm[i] = norm_frame

    return seq_60_norm


def process_single_video_label(label_name, video_dir, output_data_dir):
    """
    Xử lý tất cả video của 1 nhãn video trong dataset/train/<label_name>/
    """
    target_label_dir = output_data_dir / label_name
    target_label_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(list(video_dir.glob('*.mp4')))
    extracted_count = 0
    skipped_count = 0
    failed_count = 0

    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as holistic:
        for seq_idx, v_path in enumerate(videos):
            seq_dir = target_label_dir / f"yt_{seq_idx}"
            
            # Checkpoint nếu đã xử lý đủ 60 files .npy
            if seq_dir.exists():
                npys = list(seq_dir.glob('*.npy'))
                if len(npys) == SEQUENCE_LENGTH:
                    skipped_count += 1
                    continue

            seq_dir.mkdir(parents=True, exist_ok=True)
            norm_seq = extract_single_video(v_path, holistic)

            if norm_seq is None:
                failed_count += 1
                continue

            for f_idx in range(SEQUENCE_LENGTH):
                np.save(str(seq_dir / f"{f_idx}.npy"), norm_seq[f_idx])

            extracted_count += 1

    return {
        'label': label_name,
        'total': len(videos),
        'extracted': extracted_count,
        'skipped': skipped_count,
        'failed': failed_count
    }


# ── 4. Main Pipeline ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Regenerate Dataset 119 Nhãn (129 chiều)")
    parser.add_argument('--workers', type=int, default=4, help="Số worker song song cho trích xuất video (mặc định: 4)")
    parser.add_argument('--overwrite', action='store_true', help="Ghi đè dữ liệu nếu đã tồn tại")
    parser.add_argument('--skip_video', action='store_true', help="Chỉ xử lý webcam, bỏ qua trích xuất video")
    args = parser.parse_args()

    t_start = time.time()

    src_webcam_dir = SLT_DIR / "Data"
    src_video_dir = BASE_DIR / "dataset" / "train"
    dst_dir = SLT_DIR / "Data_normalized_merged119"
    dst_dir.mkdir(parents=True, exist_ok=True)
    models_dir = SLT_DIR / "Models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("  FSIGN - PIPELINE CHUẨN HÓA & REGENERATE DATASET HỢP NHẤT (119 NHÃN)")
    print("=" * 75)
    print(f"Thư mục nguồn Webcam:  {src_webcam_dir} (59 nhãn)")
    print(f"Thư mục nguồn Video:   {src_video_dir} (60 nhãn)")
    print(f"Thư mục đích:          {dst_dir}")
    print(f"Số worker trích xuất:  {args.workers}")
    print(f"Đặc trưng đầu ra:      {SEQUENCE_LENGTH} frames × {FEATURE_DIM_NORM} chiều (S_combined + Rel Wrist)")
    print("-" * 75)

    metadata_rows = []

    # ── GIAI ĐOẠN 1: Chuẩn hóa 59 Nhãn Webcam ────────────────────────────────
    print("\n[GIAI ĐOẠN 1/3] Đang chuẩn hóa 59 nhãn Webcam từ Data/ sang 129 chiều...")
    t1_start = time.time()
    webcam_seq_count = 0
    webcam_err_count = 0

    for idx, label_name in enumerate(WEBCAM_CLASSES, 1):
        src_label_dir = src_webcam_dir / label_name
        dst_label_dir = dst_dir / label_name
        dst_label_dir.mkdir(parents=True, exist_ok=True)

        if not src_label_dir.exists():
            print(f"  [CẢNH BÁO] Không tìm thấy thư mục webcam: {src_label_dir}")
            continue

        seq_dirs = sorted(
            [d for d in src_label_dir.iterdir() if d.is_dir()],
            key=lambda x: int(x.name) if x.name.isdigit() else 999
        )

        for seq_d in seq_dirs:
            seq_id = seq_d.name
            target_seq_d = dst_label_dir / f"webcam_{seq_id}"

            if target_seq_d.exists() and len(list(target_seq_d.glob('*.npy'))) == SEQUENCE_LENGTH and not args.overwrite:
                webcam_seq_count += 1
                metadata_rows.append({
                    'file_path': f"{label_name}/webcam_{seq_id}/0.npy",
                    'label': label_name,
                    'source': 'webcam',
                    'num_frames': SEQUENCE_LENGTH,
                    'feature_dim': FEATURE_DIM_NORM
                })
                continue

            target_seq_d.mkdir(parents=True, exist_ok=True)

            try:
                raw_frames = []
                for f_i in range(SEQUENCE_LENGTH):
                    f_p = seq_d / f"{f_i}.npy"
                    raw_frames.append(np.load(str(f_p)))

                raw_arr = np.array(raw_frames, dtype=np.float32)  # (60, 126)
                for f_i in range(SEQUENCE_LENGTH):
                    norm_k = normalize_keypoints(raw_arr[f_i], include_rel_wrist=True, include_presence=False)
                    np.save(str(target_seq_d / f"{f_i}.npy"), norm_k.astype(np.float32))

                webcam_seq_count += 1
                metadata_rows.append({
                    'file_path': f"{label_name}/webcam_{seq_id}/0.npy",
                    'label': label_name,
                    'source': 'webcam',
                    'num_frames': SEQUENCE_LENGTH,
                    'feature_dim': FEATURE_DIM_NORM
                })
            except Exception as e:
                webcam_err_count += 1
                print(f"  [LỖI] {seq_d}: {e}")

        if idx % 10 == 0 or idx == len(WEBCAM_CLASSES):
            print(f"  -> Đã xử lý {idx:2d}/{len(WEBCAM_CLASSES)} nhãn webcam ({webcam_seq_count:,} sequences)")

    t1_dur = time.time() - t1_start
    print(f"=> Hoàn tất Giai đoạn 1 trong {t1_dur:.1f}s! ({webcam_seq_count:,} sequences, {webcam_err_count} lỗi)")

    # ── GIAI ĐOẠN 2: Trích xuất & Chuẩn hóa 60 Nhãn Video ─────────────────────
    video_seq_count = 0
    video_err_count = 0

    if args.skip_video:
        print("\n[GIAI ĐOẠN 2/3] Bỏ qua xử lý video theo yêu cầu (--skip_video).")
    elif not src_video_dir.exists():
        print(f"\n[GIAI ĐOẠN 2/3] [LỖI] Không tìm thấy thư mục video: {src_video_dir}")
    else:
        print(f"\n[GIAI ĐOẠN 2/3] Đang trích xuất MediaPipe cho 60 nhãn Video từ {src_video_dir}...")
        t2_start = time.time()

        video_tasks = []
        for v_label in VIDEO_CLASSES:
            v_dir = src_video_dir / v_label
            if v_dir.exists():
                video_tasks.append((v_label, v_dir))
            else:
                print(f"  [CẢNH BÁO] Không tìm thấy thư mục video cho nhãn '{v_label}'")

        print(f"  Tìm thấy {len(video_tasks)}/60 thư mục video hợp lệ. Đang chạy song song...")

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_label = {
                executor.submit(process_single_video_label, lbl, pth, dst_dir): lbl
                for lbl, pth in video_tasks
            }

            done_count = 0
            for fut in as_completed(future_to_label):
                res = fut.result()
                done_count += 1
                v_count = res['extracted'] + res['skipped']
                video_seq_count += v_count
                video_err_count += res['failed']

                elapsed = time.time() - t2_start
                speed = done_count / elapsed if elapsed > 0 else 0
                eta_s = (len(video_tasks) - done_count) / speed if speed > 0 else 0

                print(
                    f"  [{done_count:2d}/{len(video_tasks)}] Nhãn '{res['label']:<12}' | "
                    f"Đã xử lý: {v_count:2d} seq (Mới: +{res['extracted']:2d}, Có sẵn: {res['skipped']:2d}, Lỗi: {res['failed']:2d}) | "
                    f"ETA: {int(eta_s//60):02d}m{int(eta_s%60):02d}s"
                )

        t2_dur = time.time() - t2_start
        print(f"=> Hoàn tất Giai đoạn 2 trong {int(t2_dur//60)}m {int(t2_dur%60)}s! ({video_seq_count:,} sequences, {video_err_count} lỗi)")

    # ── GIAI ĐOẠN 3: Xác minh Toàn vẹn Pipeline (0 Shape/NaN/Inf Errors) ─────
    print("\n[GIAI ĐOẠN 3/3] Đang kiểm tra xác minh toàn bộ dữ liệu (verify_pipeline)...")
    verify_start = time.time()

    all_label_dirs = sorted([d for d in dst_dir.iterdir() if d.is_dir()])
    total_checked_seqs = 0
    total_checked_frames = 0
    shape_errors = 0
    nan_inf_errors = 0
    centering_errors = 0
    seqs_per_class = {}

    for l_dir in all_label_dirs:
        l_name = l_dir.name
        seq_folders = sorted([d for d in l_dir.iterdir() if d.is_dir()])
        seqs_per_class[l_name] = len(seq_folders)

        for s_folder in seq_folders:
            total_checked_seqs += 1
            src_tag = 'webcam' if s_folder.name.startswith('webcam_') else 'video'

            # Update metadata
            if s_folder.name.startswith('yt_'):
                metadata_rows.append({
                    'file_path': f"{l_name}/{s_folder.name}/0.npy",
                    'label': l_name,
                    'source': src_tag,
                    'num_frames': SEQUENCE_LENGTH,
                    'feature_dim': FEATURE_DIM_NORM
                })

            for f_i in range(SEQUENCE_LENGTH):
                np_path = s_folder / f"{f_i}.npy"
                total_checked_frames += 1

                if not np_path.exists():
                    shape_errors += 1
                    continue

                arr = np.load(str(np_path))
                if arr.shape != (FEATURE_DIM_NORM,):
                    shape_errors += 1

                if np.isnan(arr).any() or np.isinf(arr).any():
                    nan_inf_errors += 1

                # Centering test: wrist ở kênh 0..2 (lh) và 63..65 (rh) phải bằng (0,0,0) nếu tay có mặt
                lh_wrist = arr[:3]
                rh_wrist = arr[63:66]
                if not np.all(arr[:63] == 0) and not np.allclose(lh_wrist, 0.0, atol=1e-5):
                    centering_errors += 1
                if not np.all(arr[63:126] == 0) and not np.allclose(rh_wrist, 0.0, atol=1e-5):
                    centering_errors += 1

    verify_dur = time.time() - verify_start

    # Lưu metadata.csv
    meta_csv_path = dst_dir / "metadata.csv"
    with open(meta_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['file_path', 'label', 'source', 'num_frames', 'feature_dim'])
        writer.writeheader()
        writer.writerows(metadata_rows)

    # ── GIAI ĐOẠN 4: Tạo label_map_119.json ──────────────────────────────────
    all_classes_sorted = sorted(list(seqs_per_class.keys()))
    label_map_data = {
        'num_classes': len(all_classes_sorted),
        'classes': all_classes_sorted,
        'label_to_id': {lbl: i for i, lbl in enumerate(all_classes_sorted)},
        'id_to_label': {str(i): lbl for i, lbl in enumerate(all_classes_sorted)},
        'id_to_display': {str(i): DISPLAY_MAP.get(lbl, lbl) for i, lbl in enumerate(all_classes_sorted)}
    }

    label_map_file = models_dir / "label_map_119.json"
    with open(label_map_file, 'w', encoding='utf-8') as f:
        json.dump(label_map_data, f, ensure_ascii=False, indent=2)

    with open(dst_dir / "label_map_119.json", 'w', encoding='utf-8') as f:
        json.dump(label_map_data, f, ensure_ascii=False, indent=2)

    # ── TỔNG HỢP VÀ BÁO CÁO NGHIỆM THU ─────────────────────────────────────
    total_time = time.time() - t_start
    counts = list(seqs_per_class.values()) if seqs_per_class else [0]
    min_c = min(counts) if counts else 0
    max_c = max(counts) if counts else 0
    mean_c = np.mean(counts) if counts else 0.0
    median_c = np.median(counts) if counts else 0.0

    print("\n" + "=" * 75)
    print("  KẾT QUẢ NGHIỆM THU REGENERATE DATASET HỢP NHẤT (TASK MERGE-4)")
    print("=" * 75)
    print(f"1. Tổng số nhãn hoàn thành:     {len(seqs_per_class)}/119 nhãn")
    print(f"2. Tổng số sequences đã tạo:    {total_checked_seqs:,} sequences ({total_checked_frames:,} frames)")
    print(f"   - Nguồn Webcam (59 nhãn):    {webcam_seq_count:,} sequences")
    print(f"   - Nguồn Video (60 nhãn):     {video_seq_count:,} sequences")
    print(f"3. Thống kê phân bố mẫu / nhãn:")
    print(f"   - Tối thiểu (Min sequence):  {min_c}")
    print(f"   - Tối đa (Max sequence):      {max_c}")
    print(f"   - Trung bình (Mean sequence): {mean_c:.1f}")
    print(f"   - Trung vị (Median sequence): {median_c:.1f}")
    print(f"4. Kết quả kiểm tra toàn vẹn (Verify Pipeline):")
    print(f"   - Lỗi shape (!= 129):        {shape_errors} (ĐẠT CHUẨN 100%)" if shape_errors == 0 else f"   - LỖI shape: {shape_errors}")
    print(f"   - Lỗi NaN / Inf:             {nan_inf_errors} (ĐẠT CHUẨN 100%)" if nan_inf_errors == 0 else f"   - LỖI NaN/Inf: {nan_inf_errors}")
    print(f"   - Lỗi Cổ tay Centering:      {centering_errors} (ĐẠT CHUẨN 100%)" if centering_errors == 0 else f"   - LỖI Centering: {centering_errors}")
    print(f"5. Tệp cấu hình nhãn đã xuất:   {label_map_file}")
    print(f"6. Thư mục dataset mới:         {dst_dir}")
    print(f"7. Tổng thời gian thực thi:     {int(total_time//60)}m {int(total_time%60)}s")
    print("=" * 75)


if __name__ == '__main__':
    main()
