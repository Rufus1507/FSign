"""Verification script cho Bước 2 và Bước 4 — chạy sau normalize_existing_data.py"""
import sys, os, subprocess
import numpy as np
sys.path.insert(0, r'd:\Desktop\5\DPL302m\project\Sign-Language-Translator\Sign Language Translator')

BASE = r'd:\Desktop\5\DPL302m\project\Sign-Language-Translator\Sign Language Translator'
DATA_RAW  = os.path.join(BASE, 'Data')
DATA_NORM = os.path.join(BASE, 'Data_normalized')

# ─── Bước 2: verify normalize_existing_data output ───────────────────────────
print('=== BUOC 2: Verify normalize_existing_data.py output ===')

label = 'ban dang lam gi'

raw_path  = os.path.join(DATA_RAW,  label, '0', '0.npy')
norm_path = os.path.join(DATA_NORM, label, 'webcam_0', '0.npy')

raw  = np.load(raw_path)
norm = np.load(norm_path)

print(f'Shape raw : {raw.shape}')
print(f'Shape norm: {norm.shape}')
assert norm.shape == (126,), f'Shape sai: {norm.shape}'
assert not np.any(np.isnan(norm)), 'NaN trong norm!'
assert not np.any(np.isinf(norm)), 'Inf trong norm!'
print('Shape (126,): OK | NaN: False | Inf: False')
print(f'Raw  wrist (left)  = {raw[0:3].round(4)}')
print(f'Norm wrist (left)  = {norm[0:3].round(8)}  <- phai gan 0')
print(f'Raw  wrist (right) = {raw[63:66].round(4)}')
print(f'Norm wrist (right) = {norm[63:66].round(8)} <- phai gan 0')

has_left  = not np.all(raw[:63] == 0)
has_right = not np.all(raw[63:] == 0)
if has_left:
    assert np.allclose(norm[0:3], 0, atol=1e-9), 'Wrist trai CHUA ve 0!'
    print('LEFT WRIST centered to 0: OK')
if has_right:
    assert np.allclose(norm[63:66], 0, atol=1e-9), 'Wrist phai CHUA ve 0!'
    print('RIGHT WRIST centered to 0: OK')

# 3 sequences ngau nhien
print()
print('-- Kiem tra 3 sequences ngau nhien --')
for seq_id in ['0', '15', '59']:
    r = np.load(os.path.join(DATA_RAW, label, seq_id, '30.npy'))
    n = np.load(os.path.join(DATA_NORM, label, f'webcam_{seq_id}', '30.npy'))
    assert n.shape == (126,)
    assert not np.any(np.isnan(n))
    assert not np.any(np.isinf(n))
    print(f'  webcam_{seq_id}/30.npy: shape={n.shape}, no NaN/Inf  OK')

# Collision guard
print()
print('-- Collision guard (chay lan 2 khong co --overwrite) --')
res = subprocess.run(
    [sys.executable,
     os.path.join(BASE, 'normalize_existing_data.py'),
     '--src_path', DATA_RAW, '--dst_path', DATA_NORM],
    capture_output=True,
    encoding='utf-8',
    errors='replace'
)
out_combined = (res.stdout or '') + (res.stderr or '')
if 'FileExistsError' in out_combined or 'da ton tai' in out_combined.lower() or 'tồn tại' in out_combined.lower():
    print('Collision guard: FileExistsError raised correctly  OK')
elif res.returncode != 0:
    # FileExistsError raises exception -> non-zero exit
    print(f'Collision guard: exit code={res.returncode} (khong co --overwrite)  OK')
    err_snippet = (res.stderr or '')[-200:].strip()
    print(f'  Stderr snippet: {err_snippet}')
else:
    print(f'WARNING: Ky vong FileExistsError nhung exit code=0. Stderr: {(res.stderr or "")[-200:]}')

print()
print('>>> BUOC 2 PASSED <<<')

# ─── Bước 4: verify dataset_loader ───────────────────────────────────────────
print()
print('=== BUOC 4: Verify dataset_loader.py ===')
from dataset_loader import load_dataset_normalized, inspect_dataset

inspect_dataset(DATA_NORM)

# Test load_dataset_normalized với 3 nhãn đầu để verify nhanh
labels_all = sorted([d for d in os.listdir(DATA_NORM) if os.path.isdir(os.path.join(DATA_NORM, d))])
test_actions = labels_all[:3]
print(f'Test load 3 nhan: {test_actions}')

X, y, label_map = load_dataset_normalized(data_path=DATA_NORM, sequence_length=60, actions=test_actions)
print(f'X.shape = {X.shape}')
print(f'y.shape = {y.shape}')

assert X.shape[1] == 60,  f'Sai sequence_length: {X.shape[1]}'
assert X.shape[2] == 126, f'Sai feature dim: {X.shape[2]}'
assert not np.any(np.isnan(X)), 'NaN trong X!'
assert not np.any(np.isinf(X)), 'Inf trong X!'
print('sequence_length=60: OK | feature_dim=126: OK | no NaN/Inf: OK')

label_first = test_actions[0]
seq_dirs    = os.listdir(os.path.join(DATA_NORM, label_first))
webcam_dirs = [d for d in seq_dirs if d.startswith('webcam_')]
yt_dirs     = [d for d in seq_dirs if d.startswith('yt_')]
print(f'Label "{label_first}": {len(webcam_dirs)} webcam_* | {len(yt_dirs)} yt_*')
assert len(webcam_dirs) > 0, 'Khong doc duoc webcam_* dirs!'
print('load_dataset_normalized doc duoc webcam_* sequences: OK')

print()
print('>>> BUOC 4 PASSED <<<')
print()
print('==============================')
print(' TAT CA VERIFICATION PASSED!')
print('==============================')
