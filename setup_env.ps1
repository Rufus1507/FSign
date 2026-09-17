Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "[1/3] Tạo môi trường ảo Python 3.9 bằng uv..." -ForegroundColor Yellow
uv venv --python 3.9 .venv

Write-Host "`n[2/3] Cài đặt các thư viện vào .venv..." -ForegroundColor Yellow
# Lựa chọn 1 (Khuyên dùng): TensorFlow 2.10.0 tương thích tốt nhất với MediaPipe hiện tại
uv pip install --python .venv\Scripts\python.exe tensorflow==2.10.0 opencv-python "mediapipe==0.10.11" "numpy<2" scikit-learn matplotlib ipykernel
uv pip install --python .venv\Scripts\python.exe "protobuf==3.20.3" --no-deps

Write-Host "`n[3/3] Kiểm tra import các thư viện..." -ForegroundColor Yellow
& .venv\Scripts\python.exe -c "import tensorflow as tf; import mediapipe as mp; import cv2; import sklearn; import matplotlib; print('=> CÀI ĐẶT THÀNH CÔNG!'); print('TensorFlow:', tf.__version__); print('MediaPipe:', mp.__version__); print('OpenCV:', cv2.__version__)"

Write-Host "`nHoàn tất! Môi trường ảo đã sẵn sàng tại thư mục .venv" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
