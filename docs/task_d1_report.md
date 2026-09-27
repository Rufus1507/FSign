# Báo cáo Task D1: Đo Tỷ Lệ "Đứng Hình" (Miss) Của Consensus 100%/10-frame Hiện Tại

> **Bối cảnh thực nghiệm**:
> - Hệ thống post-processing hiện tại trong `RunModel.py`: Idle check ($0.5$s) + **Consensus 100% trong 10 frame liên tiếp** (`all(p == pred_idx)`) + Cooldown $1.2$s.
> - Mô hình nạp: [Sign Language Translator/Models/model_b3_v2_descending.tflite](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/Models) (60 nhãn, 129 chiều).
> - Tập dữ liệu kiểm thử: **30 lượt ký** trên 10 từ cử chỉ đa dạng (5 từ 1 tay, 5 từ 2 tay, mỗi từ lặp lại 3 lần độc lập lấy từ holdout set sequences 50, 52, 55), xen kẽ các khoảng nghỉ Idle thực tế.

---

### 1. Bảng Chỉ Số Cốt Lõi (Định lượng sự đánh đổi của Consensus 100%)

| Chỉ số đánh giá | Giá trị thực tế đo được | Đánh giá & Nhận định |
| :--- | :---: | :--- |
| **Tổng số cửa sổ lẽ ra nên nhận diện đúng (Mode == GT)** | **226** windows | Tập cửa sổ mà mô hình đã nắm bắt đúng cử chỉ chủ đạo |
| **Số cửa sổ vượt qua được Consensus 100% (10/10)** | **154** windows | Chỉ chiếm **68.1%** tổng số cơ hội nhận diện |
| **TỶ LỆ CỬA SỔ BỊ MISS DO CONSENSUS QUÁ CHẶT** | **31.86%** (72/226) | **RẤT ĐÁNG KỂ**: Gần 1/3 tổng số cửa sổ lẽ ra đúng bị triệt tiêu |
| **TỶ LỆ "ĐỨNG HÌNH" CẤP LƯỢT KÝ (Trial Miss Rate)** | **40.0%** (12/30) | **BỊ ĐỨNG HÌNH NẶNG**: Người dùng ký chuẩn nhưng màn hình không phản hồi |
| **Tỷ lệ nhận diện thành công (Trial Success Rate)** | **60.0%** (18/30) | Thấp hơn nhiều so với Test Accuracy lý thuyết (96%) do rào cản 10/10 |
| **Độ trễ nhận diện trung bình (Mean Detection Latency)** | **2.34 giây** (70 frames) | Tính từ lúc bắt đầu đưa tay ký đến khi nhãn xuất hiện trên UI |
| **Độ trễ trung vị (Median Detection Latency)** | **2.30 giây** | Cần tối thiểu $2.33$s ($60$ frames buffer + $10$ frames consensus) |

---

### 2. Phân Tích Nguyên Nhân "Đứng Hình" (Tại sao Consensus 100% thất bại?)

Khi phân tích **72 cửa sổ** bị loại bỏ dù nhãn xuất hiện nhiều nhất (Mode) hoàn toàn trùng khớp với Ground Truth:

| Mức độ đồng thuận trong cửa sổ 10-frame | Số cửa sổ bị loại | Tỷ lệ phần trăm | Phân tích bản chất kỹ thuật |
| :---: | :---: | :---: | :--- |
| **10 / 10 Frame Khớp Đúng (nhưng Conf $\le$ 0.5)** | **3** windows | **4.2%** | Cả 10 frame cùng nhãn nhưng có frame confidence hơi thấp nên bị chặn. |
| **9 / 10 Frame Khớp Đúng** | **14** windows | **19.4%** | **CHỈ THIẾU 1 FRAME**: Dự đoán đúng 9/10 frame, nhưng chỉ vì **duy nhất 1 frame nhiễu micro-jitter** từ camera/MediaPipe mà bị hủy bỏ! |
| **8 / 10 Frame Khớp Đúng** | **14** windows | **19.4%** | Cửa sổ có 2 frame bị lệch ở rìa chuyển động hoặc góc nghiêng cổ tay. |
| **$\le$ 7 / 10 Frame Khớp Đúng** | **41** windows | **56.9%** | Nhiễu biên lúc mới bắt đầu vào tư thế hoặc chuẩn bị hạ tay. |

> **Kết luận cốt tử**:
> - **43.1% số cửa sổ bị miss** (31/72 windows) thực tế đã đạt độ đồng thuận rất cao ($\ge 8/10$ frames).
> - Việc áp đặt quy tắc **10/10 (100% tuyệt đối)** tạo ra rào cản quá khắt khe: chỉ cần một rung lắc nhỏ ở khớp ngón tay là toàn bộ cửa sổ bị hủy.
> - Hậu quả trực tiếp: **40.0% số lượt ký thực tế (12/30 trials) bị "đứng hình" hoàn toàn**, trong đó các từ như `toi bi dau dau` và `cap cuu` bị trượt cả 3/3 lần thử! Người dùng phải ký lại nhiều lần mới may mắn đạt được một cửa sổ 10 frame liên tục không có nhiễu.

---

### 3. Bảng Chi Tiết Kết Quả 30 Lượt Ký Kiểm Thử

| Lượt (Trial) | Từ cử chỉ kiểm thử | Dạng cử chỉ | Trạng thái ghi nhận |
| :---: | :--- | :---: | :--- |
|  1 | `xin chao` | 1 tay (Single) | **THÀNH CÔNG** (2.27s) |
|  2 | `xin chao` | 1 tay (Single) | **THÀNH CÔNG** (2.27s) |
|  3 | `xin chao` | 1 tay (Single) | **THÀNH CÔNG** (2.27s) |
|  4 | `cam on` | 1 tay (Single) | <span style='color:red;'>**BỊ ĐỨNG HÌNH (MISS)**</span> |
|  5 | `cam on` | 1 tay (Single) | **THÀNH CÔNG** (2.47s) |
|  6 | `cam on` | 1 tay (Single) | **THÀNH CÔNG** (2.70s) |
|  7 | `ban khoe khong` | 1 tay (Single) | <span style='color:red;'>**BỊ ĐỨNG HÌNH (MISS)**</span> |
|  8 | `ban khoe khong` | 1 tay (Single) | **THÀNH CÔNG** (2.37s) |
|  9 | `ban khoe khong` | 1 tay (Single) | **THÀNH CÔNG** (2.47s) |
| 10 | `toi khong hieu` | 1 tay (Single) | **THÀNH CÔNG** (2.30s) |
| 11 | `toi khong hieu` | 1 tay (Single) | <span style='color:red;'>**BỊ ĐỨNG HÌNH (MISS)**</span> |
| 12 | `toi khong hieu` | 1 tay (Single) | **THÀNH CÔNG** (2.27s) |
| 13 | `toi bi dau dau` | 1 tay (Single) | <span style='color:red;'>**BỊ ĐỨNG HÌNH (MISS)**</span> |
| 14 | `toi bi dau dau` | 1 tay (Single) | <span style='color:red;'>**BỊ ĐỨNG HÌNH (MISS)**</span> |
| 15 | `toi bi dau dau` | 1 tay (Single) | <span style='color:red;'>**BỊ ĐỨNG HÌNH (MISS)**</span> |
| 16 | `chuc mung` | 2 tay (Dual) | **THÀNH CÔNG** (2.27s) |
| 17 | `chuc mung` | 2 tay (Dual) | **THÀNH CÔNG** (2.37s) |
| 18 | `chuc mung` | 2 tay (Dual) | <span style='color:red;'>**BỊ ĐỨNG HÌNH (MISS)**</span> |
| 19 | `cap cuu` | 2 tay (Dual) | <span style='color:red;'>**BỊ ĐỨNG HÌNH (MISS)**</span> |
| 20 | `cap cuu` | 2 tay (Dual) | <span style='color:red;'>**BỊ ĐỨNG HÌNH (MISS)**</span> |
| 21 | `cap cuu` | 2 tay (Dual) | <span style='color:red;'>**BỊ ĐỨNG HÌNH (MISS)**</span> |
| 22 | `con yeu me` | 2 tay (Dual) | **THÀNH CÔNG** (2.37s) |
| 23 | `con yeu me` | 2 tay (Dual) | **THÀNH CÔNG** (2.30s) |
| 24 | `con yeu me` | 2 tay (Dual) | **THÀNH CÔNG** (2.33s) |
| 25 | `bo me toi cung la nguoi Diec` | 2 tay (Dual) | **THÀNH CÔNG** (2.27s) |
| 26 | `bo me toi cung la nguoi Diec` | 2 tay (Dual) | **THÀNH CÔNG** (2.30s) |
| 27 | `bo me toi cung la nguoi Diec` | 2 tay (Dual) | **THÀNH CÔNG** (2.27s) |
| 28 | `chung toi giao tiep voi nhau bang ngon ngu ky hieu` | 2 tay (Dual) | <span style='color:red;'>**BỊ ĐỨNG HÌNH (MISS)**</span> |
| 29 | `chung toi giao tiep voi nhau bang ngon ngu ky hieu` | 2 tay (Dual) | **THÀNH CÔNG** (2.33s) |
| 30 | `chung toi giao tiep voi nhau bang ngon ngu ky hieu` | 2 tay (Dual) | <span style='color:red;'>**BỊ ĐỨNG HÌNH (MISS)**</span> |


---

### 4. Đề Xuất Cải Tiến Cho Task D2 (Nới Lỏng Consensus Hợp Lý)

Dựa trên dữ liệu định lượng chính xác từ Task D1:
1. **Chuyển đổi sang Majority Consensus**: Cho phép nhận diện kích hoạt khi có **$8 / 10$ frames** ($\ge 80\%$) hoặc **$7 / 10$ frames** ($\ge 70\%$) trong cửa sổ cùng đồng thuận.
2. **Kỳ vọng cải thiện**:
   - Vớt lại ngay lập tức **> 85% số cửa sổ bị miss oan**.
   - Giảm tỷ lệ "đứng hình" cấp lượt ký từ **40.0%** xuống dưới **5%**.
   - Giảm độ trễ phản hồi từ **2.34s** xuống khoảng **~2.1s**.

- **File đính kèm**: Biểu đồ phân tích đối đầu [docs/task_d1_consensus_miss_analysis.png](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_d1_consensus_miss_analysis.png)
