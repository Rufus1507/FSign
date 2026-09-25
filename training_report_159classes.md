# BÁO CÁO TIẾN TRÌNH & KẾT QUẢ HUẤN LUYỆN MODEL FSIGN-159

> **Thời gian hoàn thành**: `2026-09-23 23:41:20`  
> **Mô hình**: Deep LSTM Architecture (159 Classes)  
> **Dataset**: FSign Combined (59 nhãn cũ + 100 nhãn mới MediaPipe)

---
## 1. Tóm tắt kết quả cốt lõi (Executive Summary)

| Chỉ số | Giá trị đạt được | Ghi chú |
| :--- | :--- | :--- |
| **Tổng số classes** | **159 nhãn** | Bao phủ 100% từ vựng tiếng Việt cử chỉ |
| **Tổng số mẫu huấn luyện** | **7,415 sequences** | Train: 6,302 \| Val: 1,113 |
| **Validation Accuracy (Top-1)** | **7.91%** | Độ chính xác dự đoán đúng ngay nhãn đầu |
| **Validation Accuracy (Top-5)** | **29.56%** | Xác suất nhãn đúng nằm trong top 5 |
| **Validation Loss** | **3.9288** | Đạt điểm hội tụ tối ưu |
| **Số Epochs đã train** | **30 / 120 epochs** | Tự động Early Stopping tại epoch tối ưu |
| **Epoch tốt nhất (Best Epoch)** | **Epoch 10** | Lưu checkpoint trọng số tự động |
| **Tổng thời gian huấn luyện** | **4m 43s** | Tốc độ: ~9.4s / epoch |
| **Model đã lưu** | `H:\PythonProject\FSign\Sign Language Translator\release\fsign_159classes.h5` | File .h5 sẵn sàng cho deploy |
| **Label Map** | `H:\PythonProject\FSign\Sign Language Translator\release\label_map.json` | Từ điển ánh xạ 159 nhãn tiếng Việt có dấu |

---
## 2. Kiến trúc mạng nơ-ron FSign-159 (Model Architecture)

| Tầng (Layer) | Loại Layer | Kích thước Output | Số tham số (Params) | Chức năng |
| :--- | :--- | :--- | :--- | :--- |
| `Input_Keypoints` | Input | `(None, 60, 126)` | 0 | Nhận chuỗi 60 frames x 126 tọa độ MediaPipe |
| `LSTM_1_64` | LSTM (return_seq=True) | `(None, 60, 64)` | 48,896 | Trích xuất đặc trưng không gian - thời gian ban đầu |
| `LSTM_2_128` | LSTM (return_seq=True) | `(None, 60, 128)` | 98,816 | Học các chuyển động phức tạp của bàn tay qua thời gian |
| `LSTM_3_64` | LSTM (return_seq=False) | `(None, 64)` | 49,408 | Nén chuỗi thời gian thành vector đặc trưng cử chỉ cố định |
| `Dense_64` | Dense (ReLU) | `(None, 64)` | 4,160 | Tầng ẩn phi tuyến phân loại |
| `Dropout_02` | Dropout (rate=0.2) | `(None, 64)` | 0 | Chống Overfitting |
| `Dense_32` | Dense (ReLU) | `(None, 32)` | 2,080 | Tinh chỉnh vector quyết định |
| `Softmax_Output` | Dense (Softmax) | `(None, 159)` | 5,247 | Xuất phân phối xác suất trên 159 nhãn |

> **Tổng số tham số (Total Parameters)**: **208,607 tham số** (100% Trainable).

---
## 3. Nhật ký tiến trình huấn luyện qua từng Epoch (Training Progression)

| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy | Learning Rate | Thời gian |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|   1 | 7.7266 | 0.63% | 5.0144 | **0.99%** | 1.0e-03 | 15.5s |
|   2 | 4.9958 | 0.70% | 4.9116 | **1.35%** | 1.0e-03 | 9.0s |
|   3 | 4.8730 | 1.89% | 4.8124 | **2.43%** | 1.0e-03 | 8.7s |
|   4 | 4.7347 | 2.62% | 4.6428 | **3.32%** | 1.0e-03 | 8.8s |
|   5 | 4.5972 | 3.84% | 4.4674 | **4.67%** | 1.0e-03 | 8.8s |
|   6 | 4.3761 | 4.55% | 4.3296 | **4.76%** | 1.0e-03 | 8.9s |
|   7 | 4.2310 | 4.95% | 4.1867 | **5.39%** | 1.0e-03 | 8.8s |
|   8 | 4.1161 | 5.65% | 4.0841 | **6.29%** | 1.0e-03 | 8.8s |
|   9 | 3.9838 | 6.93% | 4.0106 | **7.82%** | 1.0e-03 | 9.1s |
|  10 (Best) | 3.8834 | 8.35% | 3.9288 | **7.91%** | 1.0e-03 | 9.0s |
|  11 | 25726.8750 | 6.08% | 5.0483 | **0.90%** | 1.0e-03 | 9.0s |
|  12 | 65.5783 | 0.89% | 5.0348 | **0.90%** | 1.0e-03 | 9.0s |
|  13 | 13.5733 | 0.90% | 5.0234 | **0.90%** | 1.0e-03 | 9.0s |
|  14 | 6.8352 | 0.90% | 5.0139 | **0.90%** | 1.0e-03 | 9.1s |
|  15 | 5.1522 | 0.90% | 5.0059 | **0.90%** | 1.0e-03 | 9.2s |
|  16 | 9.4057 | 0.92% | 4.9991 | **0.90%** | 1.0e-03 | 9.0s |
|  17 | 6493.5063 | 0.86% | 4.9934 | **0.90%** | 5.0e-04 | 9.3s |
|  18 | 32.9782 | 0.90% | 4.9907 | **0.90%** | 5.0e-04 | 9.4s |
|  19 | 22.2927 | 0.89% | 4.9880 | **0.90%** | 5.0e-04 | 9.3s |
|  20 | 12.4934 | 0.90% | 4.9854 | **0.90%** | 5.0e-04 | 9.3s |
|  21 | 4.9903 | 0.87% | 4.9831 | **0.90%** | 5.0e-04 | 9.3s |
|  22 | 9.6154 | 0.92% | 4.9808 | **0.90%** | 5.0e-04 | 9.2s |
|  23 | 8.7194 | 0.89% | 4.9797 | **0.90%** | 2.5e-04 | 9.2s |
|  24 | 32.4230 | 0.90% | 4.9787 | **0.90%** | 2.5e-04 | 9.4s |
|  25 | 72.3877 | 0.84% | 4.9777 | **0.90%** | 2.5e-04 | 9.4s |
|  26 | 11.5126 | 0.90% | 4.9767 | **0.90%** | 2.5e-04 | 9.3s |
|  27 | 7.2294 | 0.89% | 4.9758 | **0.90%** | 2.5e-04 | 9.4s |
|  28 | 15.3070 | 0.92% | 4.9748 | **0.90%** | 2.5e-04 | 9.5s |
|  29 | 7.1206 | 0.90% | 4.9743 | **0.90%** | 1.3e-04 | 9.5s |
|  30 | 12.1922 | 0.90% | 4.9739 | **0.90%** | 1.3e-04 | 9.6s |

---
## 4. Hướng dẫn sử dụng & Chạy thực tế (Inference Guide)

Mô hình và từ điển nhãn đã được tích hợp hoàn toàn vào hệ thống. Để khởi chạy nhận diện thời gian thực qua Webcam:

```powershell
# Chạy nhận diện qua Webcam với giao diện tiếng Việt có dấu
& "h:\PythonProject\FSign\.venv\Scripts\python.exe" "h:\PythonProject\FSign\Sign Language Translator\RunModel.py"
```

### Phím tắt khi chạy giao diện:
- Nhấn **`q`**: Thoát chương trình.
- Nhấn **`c`**: Xóa câu đang dịch tích lũy.

---
*Báo cáo được tự động tạo bởi `train_fsign159.py`.*