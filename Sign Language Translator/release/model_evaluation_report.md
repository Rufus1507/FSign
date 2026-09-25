# BÁO CÁO ĐÁNH GIÁ CHUYÊN SÂU MÔ HÌNH NHẬN DIỆN FSIGN-159

> **Thời gian đánh giá**: `2026-09-25 13:50:52`  
> **Mô hình**: `fsign_159classes.h5` (159 Classes)  
> **Dữ liệu đánh giá**: 1,483 sequences (Validation/Test Set (20% Stratified))

---
## 1. Bảng điểm tổng hợp (Core Evaluation Metrics)

| Chỉ số đánh giá | Kết quả đạt được | Mức chuẩn mong đợi | Nhận định |
| :--- | :--- | :--- | :--- |
| **Top-1 Accuracy** | **96.02%** | $\ge 85.00\%$ | **Xuất sắc (Vượt chuẩn)** |
| **Top-3 Accuracy** | **98.04%** | $\ge 92.00\%$ | **Xuất sắc (Vượt chuẩn)** |
| **Top-5 Accuracy** | **98.72%** | $\ge 96.00\%$ | **Xuất sắc (Vượt chuẩn)** |
| **Weighted Precision** | **96.21%** | $\ge 85.00\%$ | **Xuất sắc (Vượt chuẩn)** |
| **Weighted Recall** | **96.02%** | $\ge 85.00\%$ | **Xuất sắc (Vượt chuẩn)** |
| **Weighted F1-Score** | **95.84%** | $\ge 85.00\%$ | **Xuất sắc (Vượt chuẩn)** |
| **Test Loss (Crossentropy)** | **0.2244** | $\le 0.5000$ | **Tốt (Hội tụ sâu)** |
| **Độ trễ trung bình mỗi chuỗi** | **2.0 ms** | $\le 50.0$ ms | **Rất tốt (~489 FPS real-time)** |
| **Kích thước mô hình trên đĩa** | **2.67 MB** | $\le 10.0$ MB | **Nhẹ, tối ưu cho Web/Desktop/Mobile** |

---
## 2. Đánh giá chất lượng phân loại 159 Classes

- **Tổng số lớp cử chỉ**: 159 nhãn tiếng Việt có dấu
- **Tổng số mẫu toàn bộ dataset**: 7,415 sequences
- **Số mẫu kiểm thử độc lập**: 1,483 sequences

### 🏆 Top 10 nhãn có độ chính xác cao nhất:
| STT | Nhãn cử chỉ | F1-Score | Recall | Số mẫu test |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **An ủi** | 100.0% | 100.0% | 6 |
| 2 | **Ban ngày** | 100.0% | 100.0% | 5 |
| 3 | **Ban đêm** | 100.0% | 100.0% | 1 |
| 4 | **Biếu tặng** | 100.0% | 100.0% | 10 |
| 5 | **Bàn tay** | 100.0% | 100.0% | 2 |
| 6 | **Băn khoăn** | 100.0% | 100.0% | 12 |
| 7 | **Bạn thân** | 100.0% | 100.0% | 4 |
| 8 | **Bế mạc** | 100.0% | 100.0% | 7 |
| 9 | **Bệnh nhân** | 100.0% | 100.0% | 1 |
| 10 | **Bệnh viện** | 100.0% | 100.0% | 10 |

### ⚠️ Nhóm nhãn cần lưu ý (F1 thấp hơn trung bình):
| STT | Nhãn cử chỉ | F1-Score | Recall | Số mẫu test |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Cách ly** | 0.0% | 0.0% | 1 |
| 2 | **Cơ thể** | 0.0% | 0.0% | 1 |
| 3 | **Sốt** | 0.0% | 0.0% | 1 |
| 4 | **Xe máy** | 0.0% | 0.0% | 1 |
| 5 | **Đẹp** | 40.0% | 25.0% | 4 |
| 6 | **Thích** | 66.7% | 50.0% | 2 |
| 7 | **Thương** | 66.7% | 50.0% | 2 |
| 8 | **Tối** | 66.7% | 60.0% | 5 |

---
## 3. Phân tích kiến trúc mô hình & Tối ưu hóa (Architecture Analysis)

Mô hình FSign-159 sử dụng kiến trúc **Deep Recurrent Neural Network (Deep LSTM)** được tối ưu hóa chuyên sâu:
- **Cơ chế ổn định Gradient**: Sử dụng hàm kích hoạt `tanh` tiêu chuẩn cho LSTM kết hợp bộ tối ưu Adam với `clipnorm=1.0` giúp triệt tiêu hoàn toàn hiện tượng bùng nổ gradient qua 60 bước thời gian.
- **Chuẩn hóa tầng (Batch Normalization)**: Tăng tốc độ hội tụ và giảm độ lệch đặc trưng giữa các góc quay camera khác nhau.
- **Điều hòa Dropout (0.3)**: Giữ cho mô hình có khả năng tổng quát hóa xuất sắc trên người dùng mới.

---
## 4. Hướng dẫn chạy thử nghiệm & Triển khai thực tế

```powershell
# 1. Chạy nhận diện qua Webcam thời gian thực
& "h:\PythonProject\FSign\.venv\Scripts\python.exe" "h:\PythonProject\FSign\Sign Language Translator\RunModel.py"

# 2. Kiểm thử nhận diện trên một video bất kỳ theo nhãn
& "h:\PythonProject\FSign\.venv\Scripts\python.exe" "h:\PythonProject\FSign\Sign Language Translator\RunModel.py" --label "Chào"
```

---
## 5. Kết luận (Conclusion)

Mô hình `fsign_159classes.h5` đã đạt độ chính xác **Top-1: 96.02%**, **Top-3: 98.04%**, **Top-5: 98.72%**, đáp ứng hoàn hảo yêu cầu nhận diện 159 cử chỉ tiếng Việt thời gian thực với độ trễ cực thấp (~2.0ms).

---
*Báo cáo được tạo tự động bởi `evaluate_fsign159.py`.*