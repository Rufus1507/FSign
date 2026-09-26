# BÁO CÁO ĐÁNH GIÁ MÔ HÌNH & NÂNG CẤP GIAO DIỆN UI CAMERA FSIGN-159

> **Thời gian cập nhật**: `2026-09-25 14:00:10`  
> **Mô hình**: `fsign_159classes.h5` (159 Nhãn tiếng Việt)  
> **Phiên bản UI Camera**: `RunModel.py` (Clean Floating HUD & Aspect Ratio Preserved)  
> **Tập dữ liệu kiểm thử**: 1,483 sequences (20% Stratified Test Split)

---

## 1. ĐÁNH GIÁ CHUYÊN SÂU MÔ HÌNH (MODEL RE-EVALUATION)

### 1.1 Bảng điểm chỉ số tổng hợp (Core Evaluation Metrics)

| Chỉ số đánh giá | Kết quả đạt được | Mức chuẩn mong đợi | Nhận định chi tiết |
| :--- | :--- | :--- | :--- |
| **Top-1 Accuracy** | **96.02%** | $\ge 85.00\%$ | **Xuất sắc (Vượt chuẩn 11.02%)** |
| **Top-3 Accuracy** | **98.04%** | $\ge 92.00\%$ | **Xuất sắc (Vượt chuẩn 6.04%)** |
| **Top-5 Accuracy** | **98.72%** | $\ge 96.00\%$ | **Xuất sắc (Vượt chuẩn 2.72%)** |
| **Weighted Precision** | **96.21%** | $\ge 85.00\%$ | **Độ chính xác cực cao, ít báo nhầm** |
| **Weighted Recall** | **96.02%** | $\ge 85.00\%$ | **Khả năng phát hiện cử chỉ bao phủ tốt** |
| **Weighted F1-Score** | **95.84%** | $\ge 85.00\%$ | **Cân bằng hoàn hảo giữa Precision & Recall** |
| **Test Loss (Crossentropy)** | **0.2244** | $\le 0.5000$ | **Mô hình hội tụ sâu, ít Overfitting** |
| **Độ trễ xử lý (Latency)** | **2.0 ms** | $\le 50.0$ ms | **Tốc độ cực nhanh (~489 - 500 FPS)** |
| **Dung lượng file Model** | **2.67 MB** | $\le 10.0$ MB | **Siêu nhẹ, tối ưu cho Web/Desktop/Mobile** |

---

### 1.2 Phân tích chất lượng phân loại 159 Classes

- **Tổng số lớp cử chỉ**: 159 nhãn cử chỉ tiếng Việt có dấu
- **Tổng số mẫu dataset**: 7,415 sequences (Chuỗi 60 khung hình)
- **Số mẫu test độc lập**: 1,483 sequences

#### 🏆 Top 10 Nhãn cử chỉ đạt độ chính xác cao nhất (F1-Score 100%):
| STT | Nhãn cử chỉ | Precision | Recall | F1-Score | Số mẫu test |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | **An ủi** | 100.0% | 100.0% | **100.0%** | 6 |
| 2 | **Ban ngày** | 100.0% | 100.0% | **100.0%** | 5 |
| 3 | **Ban đêm** | 100.0% | 100.0% | **100.0%** | 1 |
| 4 | **Biếu tặng** | 100.0% | 100.0% | **100.0%** | 10 |
| 5 | **Bàn tay** | 100.0% | 100.0% | **100.0%** | 2 |
| 6 | **Băn khoăn** | 100.0% | 100.0% | **100.0%** | 12 |
| 7 | **Bạn thân** | 100.0% | 100.0% | **100.0%** | 4 |
| 8 | **Bế mạc** | 100.0% | 100.0% | **100.0%** | 7 |
| 9 | **Bệnh nhân** | 100.0% | 100.0% | **100.0%** | 1 |
| 10 | **Bệnh viện** | 100.0% | 100.0% | **100.0%** | 10 |

---

## 2. NÂNG CẤP GIAO DIỆN UI CAMERA DỄ NHÌN, DỄ DÙNG & ĐÚNG TỈ LỆ

Giao diện Camera UI trong [`RunModel.py`](file:///h:/PythonProject/FSign/Sign%20Language%20Translator/RunModel.py) đã được tinh chỉnh hoàn thiện với 3 tiêu chuẩn cốt lõi:

```
+-----------------------------------------------------------------------------------------+
| [⚡ FSIGN AI  🟢 LIVE]     [DỊCH: Bạn thân của tôi rất nhiệt tình]    [FPS: 30 | 60/60] |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|         ┌──────────┐                                           ┌──────────┐             |
|         │ TAY TRÁI │ (Magenta Corner)                          │ TAY PHẢI │ (Cyan)      |
|         └──────────┘                                           └──────────┘             |
|                                                                                         |
+-----------------------------------------------------------------------------------------+
| CỬ CHỈ DỰ ĐOÁN: BẠN THÂN                                                                |
| [====================================================================] Độ tin cậy: 98.4%|
| [F] Fullscreen  |  [C] Xóa câu  |  [S] Xương: BẬT  |  [SPACE] Tạm dừng  |  [ESC] Thoát     |
+-----------------------------------------------------------------------------------------+
```

### 2.1 Chi tiết các điểm cải tiến UI/UX

1. **Giữ nguyên Tỉ lệ Khung hình (Aspect Ratio Preservation - Đúng tỉ lệ)**:
   - Tích hợp hàm `preserve_aspect_ratio_display` tự động tính toán căn chỉnh khung hình video với màn hình hiển thị.
   - Triệt tiêu hoàn toàn hiện tượng méo hay giãn hình khi kéo resize cửa sổ OpenCV hoặc bật chế độ Toàn màn hình (Fullscreen).

2. **Giao diện Floating Cards bo tròn góc (Dễ nhìn & Tinh tế)**:
   - Thay thế các thanh dải dài che khuất màn hình bằng các khối thẻ **Floating Cards góc bo tròn (`radius=12`)** thiết kế theo chuẩn Slate-900 / Cyan-500.
   - Vùng trung tâm màn hình được giải phóng 100%, giúp người dùng theo dõi cử chỉ tay và khuôn mặt vô cùng thoải mái.

3. **Thiết kế tối ưu trải nghiệm (Dễ dùng)**:
   - Hướng dẫn phím bấm được tinh gọn tối đa ở góc dưới thẻ điều khiển:  
     `[F] Fullscreen  |  [C] Xóa câu  |  [S] Xương: BẬT  |  [SPACE] Tạm dừng  |  [ESC] Thoát`
   - Thanh tiến trình độ tin cậy được thiết kế gọn gàng với hiệu ứng đổi màu 3 cấp độ (🟢 Xanh lá $\ge 75\%$, 🔵 Xanh lam $\ge 50\%$, 🟠 Cam $< 50\%$).
   - Font chữ Tiếng Việt nét chuẩn HD với tương phản cao (`#F8FAFC` trên nền `#0F172A`), dễ đọc ngay cả khi xem từ xa.

---

## 3. HƯỚNG DẪN KHỞI CHẠY CHƯƠNG TRÌNH

### 3.1 Chạy mặc định (Cửa sổ lớn HD 1280x720):
```powershell
& "h:\PythonProject\FSign\.venv\Scripts\python.exe" "h:\PythonProject\FSign\Sign Language Translator\RunModel.py"
```

### 3.2 Chạy trực tiếp ở Chế độ Toàn màn hình (Fullscreen):
```powershell
& "h:\PythonProject\FSign\.venv\Scripts\python.exe" "h:\PythonProject\FSign\Sign Language Translator\RunModel.py" --fullscreen
```

---

## 4. KẾT LUẬN

Giao diện Camera UI hiện tại đã đạt độ tối ưu hoàn hảo về mặt thị giác và trải nghiệm người dùng:
- **Chuẩn tỉ lệ**: Video giữ đúng aspect ratio 16:9, không bị biến dạng hình ảnh.
- **Dễ nhìn**: Không gian xem rộng rãi, màu sắc dịu mắt, font tiếng Việt rõ đẹp.
- **Dễ dùng**: Đầy đủ phím nóng điều khiển trực quan và thông báo phản hồi sinh động.

---
*Báo cáo được tổng hợp và xuất tự động bởi hệ thống FSign Translator.*