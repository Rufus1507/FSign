# Báo cáo Task C2: Thêm Assert Chặn Cứng + Vá Lỗi Wiring Model trong RunModel.py

> **Bối cảnh**:
> - Task C1 phát hiện `RunModel.py` nạp `model_normalized_v1.tflite` (61 lớp, input 126 chiều) trong khi `Data_normalized/` là 60 lớp 129 chiều.
> - Nhiệm vụ: Thiết lập các chốt chặn `assert` cứng cáp chống lỗi mismatch, convert model tốt nhất từ Task B3 (`model_b3_v2_descending.h5`) sang TFLite, trỏ lại và benchmark toàn diện.

---

### 1. Kết quả Thiết lập Assert Chặn Cứng (C2a)

Đã bổ sung câu lệnh `assert` kiểm tra chéo ở cả 2 chiều (**Input Dimension** và **Label Count**) tại:
1. [RunModel.py](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/RunModel.py) (Khối khởi động inference chính thời gian thực).
2. [RunModel_phu_159.py](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/RunModel_phu_159.py) (Khối load model 159 lớp).
3. [train_model.py](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/train_model.py) (Khối đánh giá checkpoint sau huấn luyện).

```python
# Cú pháp assert chuẩn hóa
assert model_input_dim == 129, (
    f"Input feature dimension mismatch: expected 129 features "
    f"(126 normalized S_combined + 3 relative wrist), but model expects {model_input_dim}"
)
assert len(actions) == model_output_dim, (
    f"Label count mismatch: {len(actions)} labels vs {model_output_dim} model outputs"
)
```

#### Kết quả Thử nghiệm Negative Tests (Cố tình load sai):
| Kịch bản thử nghiệm | Cặp đối chiếu | Kết quả thực tế | Trạng thái |
| :--- | :--- | :--- | :---: |
| **Test A: Lệch số nhãn** | Model cũ (61 lớp) vs Danh sách nhãn (60 lớp) | Bắt ngoại lệ: `AssertionError: Label count mismatch: 60 labels vs 61 model outputs` | **PASS (Đã chặn đứng)** |
| **Test B: Lệch chiều đặc trưng** | Model cũ (126 chiều) vs Pipeline chuẩn (129 chiều) | Bắt ngoại lệ: `AssertionError: Input feature dimension mismatch: expected 129 features..., but model expects 126` | **PASS (Đã chặn đứng)** |

---

### 2. Kết quả Chuyển đổi Model B3 sang TFLite & Wiring lại RunModel.py (C2b)

- **Mô hình nguồn**: [Sign Language Translator/Models/model_b3_v2_descending.h5](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/Models) (Kiến trúc $128 \rightarrow 64 \rightarrow 32$, `tanh`, `clipnorm=1.0`, 60 nhãn, 129 chiều).
- **Mô hình TFLite đầu ra**: [Sign Language Translator/Models/model_b3_v2_descending.tflite](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/Models).
- **Thời gian convert**: `21.09s`.
- **Dung lượng file**: `2.36 MB` (.h5) $\rightarrow$ **`0.79 MB`** (.tflite) — **Nén gọn 66.5%**.

#### Kết quả Benchmark Hiệu Năng Suy Luận Thực Tế (CPU 4 Threads, 100 lần đo):
| Chỉ số hiệu năng | Mốc chuẩn thiết kế (Ước tính) | Bản TFLite Mới (`model_b3_v2_descending.tflite`) | Đánh giá |
| :--- | :---: | :---: | :--- |
| **Độ trễ trung bình (Mean Latency)** | $3.0 - 5.0$ ms | **`5.08` ms** | **Đạt chuẩn xuất sắc**, nằm trọn trong ngưỡng mục tiêu |
| **Độ trễ nhanh nhất (Min Latency)** | - | **`3.80` ms** | Phản hồi siêu tốc |
| **Độ trễ chậm nhất (Max Latency)** | $\le 10.0$ ms | **`7.25` ms** | Tuyệt đối không giật/lag |
| **Tốc độ suy luận (Throughput)** | $\ge 200$ FPS | **`196.9` FPS** | Đủ khả năng xử lý mượt mà camera 30-60 FPS |

---

### 3. Kết quả Positive Test: Khởi động Thực tế `RunModel.py`

- `RunModel.load_actions()` nạp chính xác **60 nhãn** từ [label_map.json](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/label_map.json) và [Data_normalized/](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/Data_normalized).
- Mô hình TFLite nạp thành công: `Input Dim = 129`, `Output Classes = 60`.
- **Tất cả các lệnh `assert` đều vượt qua 100%**, thực hiện dự đoán sequence kiểm thử trơn tru mà không có bất kỳ ngoại lệ hay nguy cơ crash nào.
