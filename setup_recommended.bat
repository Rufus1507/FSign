@echo off
echo ========================================================
echo [1/3] Tao moi truong ao Python 3.9 bang uv...
uv venv --python 3.9 .venv
if %errorlevel% neq 0 (
    echo [Loi] Khong the tao moi truong voi Python 3.9. Vui long kiem tra uv.
    pause
    exit /b %errorlevel%
)

echo.
echo [2/3] Dang cai dat TensorFlow 2.10.0, OpenCV, MediaPipe 0.10.11, NumPy, Scikit-Learn, Matplotlib, Ipykernel...
uv pip install --python .venv\Scripts\python.exe tensorflow==2.10.0 opencv-python "mediapipe==0.10.11" "numpy<2" scikit-learn matplotlib ipykernel
uv pip install --python .venv\Scripts\python.exe "protobuf==3.20.3" --no-deps
if %errorlevel% neq 0 (
    echo [Loi] Qua trinh cai dat thu vien gap su co.
    pause
    exit /b %errorlevel%
)

echo.
echo [3/3] Kiem tra cai dat va import...
.venv\Scripts\python.exe -c "import tensorflow as tf; import mediapipe as mp; import cv2; import sklearn; import matplotlib; print('=> CAI DAT THANH CONG!'); print('TensorFlow:', tf.__version__); print('MediaPipe:', mp.__version__); print('OpenCV:', cv2.__version__)"

echo ========================================================
echo Hoan tat! Ban co the chon kernel .venv trong VS Code khi mo notebook.
pause
