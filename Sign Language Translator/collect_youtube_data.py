# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
from typing import Optional

# Fix Unicode output on Windows terminals (cp1252 -> utf-8)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import csv
import json
import shutil
import hashlib
import logging
import argparse
import subprocess
import tempfile
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np
import mediapipe as mp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from preprocessing import (
    mediapipe_detection,
    extract_keypoints_raw,
    normalize_keypoints,
    resample_sequence,
    get_next_sequence_id,
    mp_holistic,
)

try:
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s',
        force=True
    )
    log = logging.getLogger('collect_youtube')
    log.setLevel(logging.INFO)
except Exception:
    pass

METADATA_FIELDNAMES = [
    'file_path', 'label', 'source', 'url',
    'original_fps', 'valid_ratio', 'timestamp',
]


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Thu thập dữ liệu cử chỉ từ YouTube cho FSign'
    )
    p.add_argument('--config', required=True,
                   help='Đường dẫn file CSV cấu hình (url, start, end, label)')
    p.add_argument('--data_path', default='Data_normalized',
                   help='Thư mục đích lưu dữ liệu normalized (mặc định: Data_normalized)')
    p.add_argument('--tmp_dir', default='tmp_clips',
                   help='Thư mục tạm để lưu video/clip trung gian (mặc định: tmp_clips)')
    p.add_argument('--sequence_length', type=int, default=60,
                   help='Số frame mỗi sequence (mặc định: 60)')
    p.add_argument('--min_confidence', type=float, default=0.5,
                   help='Ngưỡng confidence MediaPipe (mặc định: 0.5)')
    p.add_argument('--min_valid_ratio', type=float, default=0.8,
                   help='Tỷ lệ frame hợp lệ tối thiểu (mặc định: 0.8)')
    p.add_argument('--min_raw_frames', type=int, default=40,
                   help='Số frame hợp lệ tối thiểu trước resample (mặc định: 40)')
    p.add_argument('--overwrite', action='store_true',
                   help='Cho phép ghi đè sequence đã tồn tại')
    p.add_argument('--dry_run', action='store_true',
                   help='Chỉ mô phỏng, không tải/ghi file nào')
    return p.parse_args()


# ─── Utility helpers ─────────────────────────────────────────────────────────

def _url_hash(url: str) -> str:
    """Tạo hash ngắn từ URL để đặt tên file tạm."""
    return hashlib.md5(url.encode()).hexdigest()[:10]


def _parse_timestamp(ts: str) -> float:
    """
    Chuyển timestamp 'mm:ss' hoặc 'mm:ss.sss' sang giây (float).
    Ví dụ: '01:23.5' → 83.5
    """
    parts = ts.strip().split(':')
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    else:
        raise ValueError(f"Định dạng timestamp không hợp lệ: '{ts}' (dùng mm:ss hoặc hh:mm:ss)")


def _check_tool(name: str) -> bool:
    """Kiểm tra tool (yt-dlp, ffmpeg) có trong PATH hoặc môi trường Python không."""
    if shutil.which(name) is not None:
        return True

    # Kiểm tra trong thư mục Scripts của Python hiện tại
    py_dir = os.path.dirname(sys.executable)
    exe_candidate = os.path.join(py_dir, f"{name}.exe")
    if os.path.isfile(exe_candidate):
        os.environ['PATH'] = py_dir + os.pathsep + os.environ.get('PATH', '')
        return True

    # Nếu là yt-dlp, kiểm tra xem có import được module yt_dlp không
    if name == 'yt-dlp':
        try:
            import yt_dlp
            return True
        except ImportError:
            pass

    print(f"\n[LỖI] Không tìm thấy công cụ '{name}' trong hệ thống / PATH!", file=sys.stderr)
    if name == 'yt-dlp':
        print(f"  -> Cài đặt bằng lệnh: pip install yt-dlp", file=sys.stderr)
    elif name == 'ffmpeg':
        print(f"  -> Cài đặt ffmpeg: tải từ https://ffmpeg.org/ hoặc chạy: winget install Gyan.FFmpeg", file=sys.stderr)
    print(file=sys.stderr)
    return False


# ─── Step 1-3: Download & cut clip ───────────────────────────────────────────

def download_and_cut(url: str, start_ts: str, end_ts: str,
                     tmp_dir: str, dry_run: bool) -> Optional[str]:
    """
    Tải video từ YouTube và cắt đoạn theo timestamp.

    Returns:
        Đường dẫn clip đã cắt (mp4), hoặc None nếu lỗi.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    uid        = _url_hash(url)
    raw_path   = os.path.join(tmp_dir, f"{uid}_raw.mp4")
    clip_path  = os.path.join(tmp_dir, f"{uid}_clip.mp4")

    if dry_run:
        log.info(f"  [DRY RUN] Sẽ tải: {url} [{start_ts} → {end_ts}]")
        return "__dry_run__"

    # Bước 1: yt-dlp tải video gốc
    print(f"  [INFO] Dang tai video: {url} ...", flush=True)
    ytdlp_cmd = [
        sys.executable, '-m', 'yt_dlp',
        '--extractor-args', 'youtube:player_client=android,web',
        '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best',
        '--merge-output-format', 'mp4',
        '-o', raw_path,
        '--no-playlist',
        url
    ]
    try:
        result = subprocess.run(ytdlp_cmd, capture_output=True, encoding='utf-8', errors='replace')
        if result.returncode != 0 or not os.path.exists(raw_path):
            err_msg = (result.stderr or result.stdout or '')[-500:].strip()
            print(f"  [ERROR] yt-dlp that bai: {err_msg}", flush=True)
            return None
    except Exception as e:
        print(f"  [ERROR] Loi khi goi yt-dlp: {e}", flush=True)
        return None

    # Bước 2: ffmpeg cắt clip
    start_sec = _parse_timestamp(start_ts)
    end_sec   = _parse_timestamp(end_ts)
    print(f"  [INFO] Cat clip [{start_ts} -> {end_ts}] ({end_sec - start_sec:.2f}s) ...", flush=True)
    ffmpeg_bin = shutil.which('ffmpeg') or 'ffmpeg'
    ffmpeg_cmd = [
        ffmpeg_bin, '-y',
        '-i', raw_path,
        '-ss', str(start_sec),
        '-to', str(end_sec),
        '-c:v', 'libx264', '-c:a', 'aac',
        '-loglevel', 'error',
        clip_path
    ]
    try:
        result2 = subprocess.run(ffmpeg_cmd, capture_output=True, encoding='utf-8', errors='replace')
        if result2.returncode != 0 or not os.path.exists(clip_path):
            err_msg = (result2.stderr or result2.stdout or '')[-500:].strip()
            print(f"  [ERROR] ffmpeg that bai: {err_msg}", flush=True)
            if os.path.exists(raw_path):
                os.remove(raw_path)
            return None
    except Exception as e:
        print(f"  [ERROR] Loi khi goi ffmpeg: {e}", flush=True)
        if os.path.exists(raw_path):
            os.remove(raw_path)
        return None

    # Bước 3: Xóa video gốc ngay sau khi cắt
    if os.path.exists(raw_path):
        os.remove(raw_path)
    print(f"  [INFO] Da luu clip tam: {clip_path}", flush=True)
    return clip_path


# ─── Step 4-10: Extract, filter, resample, normalize ─────────────────────────

def process_clip(clip_path: str, url: str, label: str,
                 args, holistic_model) -> Optional[dict]:
    """
    Xử lý 1 clip: trích landmark → lọc → resample → normalize.

    Returns:
        dict với 'frames_norm' (list of 60 ndarray (126,)), 'fps', 'valid_ratio'
        hoặc None + log lý do nếu bị loại.
    """
    if clip_path == "__dry_run__":
        log.info(f"  [DRY RUN] Bỏ qua xử lý MediaPipe")
        return None

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"  [ERROR] Khong mo duoc clip: {clip_path}", file=sys.stderr, flush=True)
        return None

    # Bước 4: Log FPS gốc
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  [INFO] FPS goc: {fps:.2f}, tong frame: {total_frame_count}", flush=True)

    # Bước 5: Trích landmark raw từng frame
    raw_frames_all = []   # tất cả frame (kể cả invalid)
    raw_frames_valid = [] # chỉ frame detect được ít nhất 1 tay

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, results = mediapipe_detection(frame, holistic_model)
        kp = extract_keypoints_raw(results)  # (126,) RAW
        raw_frames_all.append(kp)

        # Bước 6: Frame hợp lệ = ít nhất 1 tay được detect
        has_hand = (results.left_hand_landmarks is not None or
                    results.right_hand_landmarks is not None)
        if has_hand:
            raw_frames_valid.append(kp)

    cap.release()

    n_total = len(raw_frames_all)
    n_valid = len(raw_frames_valid)
    valid_ratio = n_valid / n_total if n_total > 0 else 0.0

    print(f"  [INFO] Frame hop le: {n_valid}/{n_total} ({valid_ratio:.1%})", flush=True)

    # Bước 7: Guard valid_ratio
    if valid_ratio < args.min_valid_ratio:
        print(f"  [WARN] Loai: low_valid_ratio ({valid_ratio:.1%} < {args.min_valid_ratio:.0%})", flush=True)
        return {'reject_reason': 'low_valid_ratio', 'fps': fps, 'valid_ratio': valid_ratio}

    # Bước 8: Guard min_raw_frames
    if n_valid < args.min_raw_frames:
        print(f"  [WARN] Loai: insufficient_raw_frames ({n_valid} < {args.min_raw_frames})", flush=True)
        return {'reject_reason': 'insufficient_raw_frames', 'fps': fps, 'valid_ratio': valid_ratio}

    # Bước 9: Resample trên tọa độ RAW
    resampled_raw = resample_sequence(raw_frames_valid, args.sequence_length)  # (60, 126) RAW

    # Bước 10: Normalize từng frame đã resample
    frames_norm = [normalize_keypoints(resampled_raw[i])
                   for i in range(args.sequence_length)]

    return {
        'frames_norm'  : frames_norm,
        'fps'          : fps,
        'valid_ratio'  : valid_ratio,
        'reject_reason': None,
    }


# ─── Step 11-12: Save frames + metadata ──────────────────────────────────────

def save_sequence(frames_norm: list, label: str, url: str,
                  fps: float, valid_ratio: float,
                  args) -> str:
    """
    Lưu 60 frame vào Data_normalized/<label>/yt_<id>/ và ghi metadata.

    Returns:
        Đường dẫn thư mục sequence đã lưu.
    """
    data_path = os.path.abspath(args.data_path)
    seq_name  = get_next_sequence_id(data_path, label, source='youtube')
    seq_dir   = os.path.join(data_path, label, seq_name)

    # Collision guard (Bước 11)
    if os.path.isdir(seq_dir) and os.listdir(seq_dir):
        if not args.overwrite:
            raise FileExistsError(
                f"\nThư mục sequence đã tồn tại:\n  {seq_dir}\n"
                f"Dùng --overwrite để ghi đè."
            )

    os.makedirs(seq_dir, exist_ok=True)

    for frame_idx, kp in enumerate(frames_norm):
        npy_path = os.path.join(seq_dir, f"{frame_idx}.npy")
        np.save(npy_path, kp)

    # Bước 12: Append metadata
    metadata_path = os.path.join(data_path, 'metadata.csv')
    file_exists   = os.path.isfile(metadata_path)
    first_frame   = os.path.relpath(os.path.join(seq_dir, '0.npy'), data_path)
    row = {
        'file_path'    : first_frame.replace('\\', '/'),
        'label'        : label,
        'source'       : 'youtube',
        'url'          : url,
        'original_fps' : f"{fps:.2f}",
        'valid_ratio'  : f"{valid_ratio:.4f}",
        'timestamp'    : datetime.now().isoformat(timespec='seconds'),
    }
    with open(metadata_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return seq_dir


# ─── Main pipeline ────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not args.dry_run:
        ok_ytdlp  = _check_tool('yt-dlp')
        ok_ffmpeg = _check_tool('ffmpeg')
        if not (ok_ytdlp and ok_ffmpeg):
            sys.exit(1)

    # Đọc CSV cấu hình
    config_path = os.path.abspath(args.config)
    if not os.path.isfile(config_path):
        print(f"[ERROR] Khong tim thay file config: {config_path}", file=sys.stderr, flush=True)
        sys.exit(1)

    with open(config_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"[INFO] Config: {config_path} ({len(rows)} muc)")
    print(f"[INFO] Data path: {os.path.abspath(args.data_path)}")
    print(f"[INFO] Tham so: sequence_length={args.sequence_length}, min_confidence={args.min_confidence}, min_valid_ratio={args.min_valid_ratio}, min_raw_frames={args.min_raw_frames}")
    print()

    # Thống kê kết quả
    counters = defaultdict(int)   # success / low_valid_ratio / insufficient_raw_frames / download_failed / error
    reject_urls = defaultdict(list)

    # Khởi tạo MediaPipe Holistic (dùng min_confidence làm ngưỡng)
    holistic_kwargs = dict(
        min_detection_confidence=args.min_confidence,
        min_tracking_confidence=args.min_confidence
    )

    with mp_holistic.Holistic(**holistic_kwargs) as holistic:
        for idx, row in enumerate(rows, 1):
            url      = (row.get('url_youtube') or '').strip()
            start_ts = (row.get('thoi_gian_bat_dau') or '').strip()
            end_ts   = (row.get('thoi_gian_ket_thuc') or '').strip()
            label    = (row.get('nhan') or '').strip()

            if not url or not label or not start_ts or not end_ts:
                print(f"[WARN] [{idx}/{len(rows)}] Bo qua dong thieu thong tin: {row}")
                counters['error'] += 1
                continue

            print(f"--- [{idx}/{len(rows)}] {label} | {url} [{start_ts} -> {end_ts}]")

            # Bước 1-3: Tải và cắt clip
            tmp_dir   = os.path.abspath(args.tmp_dir)
            clip_path = download_and_cut(url, start_ts, end_ts, tmp_dir, args.dry_run)

            if clip_path is None:
                counters['download_failed'] += 1
                reject_urls['download_failed'].append(url)
                continue

            # Bước 4-10: Xử lý MediaPipe
            result = process_clip(clip_path, url, label, args, holistic)

            # Xóa clip tạm sau khi xử lý xong
            if not args.dry_run and clip_path != "__dry_run__" and os.path.exists(clip_path):
                os.remove(clip_path)

            if result is None:
                # dry_run hoặc lỗi mở video
                if args.dry_run:
                    counters['dry_run_skipped'] += 1
                else:
                    counters['error'] += 1
                continue

            reason = result.get('reject_reason')
            if reason:
                counters[reason] += 1
                reject_urls[reason].append(url)
                continue

            # Bước 11-12: Lưu sequence
            if not args.dry_run:
                try:
                    seq_dir = save_sequence(
                        frames_norm  = result['frames_norm'],
                        label        = label,
                        url          = url,
                        fps          = result['fps'],
                        valid_ratio  = result['valid_ratio'],
                        args         = args,
                    )
                    print(f"  [OK] Da luu: {seq_dir}", flush=True)
                    counters['success'] += 1
                except FileExistsError as e:
                    print(f"  [ERROR] {e}", file=sys.stderr, flush=True)
                    counters['error'] += 1
                except Exception as e:
                    print(f"  [ERROR] Loi khi luu: {e}", file=sys.stderr, flush=True)
                    counters['error'] += 1
            else:
                print(f"  [DRY RUN] Se luu sequence cho nhan '{label}'", flush=True)
                counters['dry_run_skipped'] += 1

            print()

    # ── Summary log ───────────────────────────────────────────────────────
    print()
    print('=' * 55)
    print('KET QUA PIPELINE YOUTUBE')
    print('=' * 55)
    print(f"  Thanh cong                 : {counters['success']}")
    print(f"  low_valid_ratio            : {counters['low_valid_ratio']}")
    print(f"  insufficient_raw_frames    : {counters['insufficient_raw_frames']}")
    print(f"  download_failed            : {counters['download_failed']}")
    print(f"  Loi khac                  : {counters['error']}")
    if args.dry_run:
        print(f"  (dry run, bo qua)          : {counters['dry_run_skipped']}")

    for reason in ['low_valid_ratio', 'insufficient_raw_frames', 'download_failed']:
        if reject_urls[reason]:
            print(f"\n  URLs bi loai ({reason}):")
            for u in reject_urls[reason]:
                print(f"    - {u}")
    print('=' * 55)
    sys.exit(0)


if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        if not isinstance(e, SystemExit) or e.code != 0:
            import traceback
            print(f"\n=== CAUGHT {type(e).__name__}: {e} ===", flush=True)
            traceback.print_exc(file=sys.stdout)
        sys.exit(getattr(e, 'code', 1))

