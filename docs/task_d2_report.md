# Báo cáo Task D2: Thử Nghiệm Phương Án Nới Lỏng (Majority Vote & EMA)

> **Mục tiêu thực nghiệm**:
> - D1 chỉ ra: Consensus 100% (10/10 frame) gây ra **Trial Miss Rate 40.0%** (12/30 lượt ký bị "đứng hình").
> - Task D2 đánh giá định lượng 4 phương án trên cùng 1 video test stream D1 (30 trials) và đo lường chi phí đánh đổi qua **False Positive Test** (45s video nhiễu casual).
> - Phân tích riêng 2 từ bị miss 100% (`toi bi dau dau` và `cap cuu`).

---

### 1. Bảng Tổng Hợp Đối Đầu Toàn Diện 4 Phương Án

| Phương án đánh giá | Cấu hình kỹ thuật | Trial Success Rate | Trial Miss Rate (Đứng hình) | Độ trễ TB (Latency) | False Positive (Nhiễu 45s) | `toi bi dau dau` | `cap cuu` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (D1)** | 100% / 10 frames | **60.0%** (18/30) | **40.0%** (12/30) | **2.34s** | **18 lần** / 45s | 0/3 | 0/3 |
| **Biến thể 1 (Majority 8/10)** | Majority $\ge 8/10$ + Mean Conf > 0.5 | **66.7%** (20/30) | **33.3%** (10/30) | **2.35s** | **21 lần** / 45s | 0/3 | 0/3 |
| **Biến thể 2 (EMA 0.3 + Maj 8/10)** | EMA $\alpha=0.3$ + Maj $\ge 8/10$ | **66.7%** (20/30) | **33.3%** (10/30) | **2.35s** | **21 lần** / 45s | 0/3 | 0/3 |
| **Biến thể 3 (EMA 0.5 + Maj 8/10)** | EMA $\alpha=0.5$ + Maj $\ge 8/10$ | **63.3%** (19/30) | **36.7%** (11/30) | **2.34s** | **20 lần** / 45s | 0/3 | 0/3 |

---

### 2. Phân Tích Chi Tiết Hiệu Năng & Chi Phí Đánh Đổi

#### A. Khả năng giải cứu "Đứng hình" (Trial Success Rate):
- **Baseline (D1)**: Đạt **60.0%** (18/30). Bị "đứng hình" tới **40.0%**.
- **Biến thể 1 (Majority Vote $\ge 8/10$)**:
  - Tỷ lệ nhận diện tăng lên **66.7%** (20/30).
  - Giảm tỷ lệ đứng hình từ **40.0% xuống 33.3%** (+2 trials nhận diện thành công: `cam on` trial 4 và `toi khong hieu` trial 11).
- **Biến thể 2 (EMA $\alpha=0.3$ + Majority 8/10)**:
  - Tỷ lệ nhận diện: **66.7%** (tương đương Biến thể 1, việc làm mượt EMA giúp xác suất ổn định hơn nhưng ngưỡng majority 8/10 đã lọc tốt).
- **Biến thể 3 (EMA $\alpha=0.5$ + Majority 8/10)**:
  - Tỷ lệ nhận diện: **63.3%** (thấp hơn Biến thể 1 và 2 do $\alpha=0.5$ phản ứng quá nhạy với biến động cục bộ của frame nhiễu).

#### B. Thử nghiệm False Positive (Chi phí đánh đổi khi cử động tay casual):
- Trên đoạn video 45s mô phỏng cử chỉ ngẫu nhiên (gõ phím, gãi đầu, chỉnh kính, đưa tay nói chuyện):
  - **Baseline (D1)**: Ghi nhận **18 lần** phát nhãn rác.
  - **Biến thể 1**: Ghi nhận **21 lần** (+3 lần so với baseline, tương đương tăng 16.7%).
  - **Biến thể 2**: Ghi nhận **21 lần**.
  - **Biến thể 3**: Ghi nhận **20 lần**.
- **Nhận xét quan trọng về False Positive**:
  - Ngay cả Baseline (100% / 10 frame) cũng bị tới 18 lần phát nhãn rác trên 45s cử động casual!
  - Lý do: Mô hình phân loại 60 lớp hiện tại **chưa có lớp "Negative / Idle background"** (mọi vector 129 chiều khi tay được giơ lên đều bị ép vào 1 trong 60 lớp cử chỉ).
  - Khi người dùng đưa tay lên làm việc casual kéo dài $>2.0$s (đủ 60 frame buffer), mô hình dự đoán ra 1 lớp bất kỳ với confidence cao và lặp lại liên tục $\ge 10$ frame, khiến cả consensus 100% lẫn majority vote đều bị qua mặt.
  - Do đó, việc nới lỏng consensus từ 100% xuống 8/10 **chỉ làm tăng nhẹ 3 lần False Positive (từ 18 lên 21 lần)**, trong khi vớt lại được 2 lượt ký thật bị đứng hình.

#### C. Độ trễ nhận diện (Latency):
- Cả 4 phương án đều duy trì độ trễ quanh mức **2.34s – 2.35s** (Median 2.30s – 2.32s).
- Đây là ngưỡng trễ tối ưu vì bị ràng buộc bởi độ dài cửa sổ buffer vật lý 60 frame (2.0s ở 30 FPS) cộng thêm 10 frame tích lũy consensus (0.33s).

---

### 3. Phân Tích Chuyên Sâu 2 Từ Miss 100% Ở D1: `toi bi dau dau` & `cap cuu`

| Từ cử chỉ | Baseline | Biến thể 1 | Biến thể 2 | Biến thể 3 | Kết luận bản chất lỗi |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `toi bi dau dau` (1 tay) | **0/3** | **0/3** | **0/3** | **0/3** | **VẪN MISS 100%**: Do MediaPipe Occlusion + Premature Idle Reset |
| `cap cuu` (2 tay) | **0/3** | **0/3** | **0/3** | **0/3** | **VẪN MISS 100%**: Do MediaPipe Hand-on-Hand Occlusion + Premature Idle Reset |

#### Dữ liệu phân tích landmark từ thực nghiệm:
1. **`toi bi dau dau` (Chạm tay lên thái dương)**:
   - Tay trái: 60/60 frame là zero (chuẩn cử chỉ 1 tay).
   - Tay phải: **34 / 60 frames bị mất dấu hoàn toàn (zeros)** do bàn tay áp sát vùng đầu/tai làm MediaPipe Holistic không tách được bàn tay!
   - Tuy nhiên, khi đưa đủ 60 frame dạng batch vào model, **model phân loại chính xác 99.23%** nhãn `toi bi dau dau`!
   - **Cơ chế gây miss trong thời gian thực**: Trong `RunModel.py`, khi mất dấu tay $\ge 0.5$s (15 frames zeros liên tiếp), bộ xử lý tự động kích hoạt `Idle Reset` $\rightarrow$ xóa sạch buffer sequence (`sequence.clear()`). Do đó, buffer không bao giờ tích lũy đủ 60 frame để thực hiện suy luận!

2. **`cap cuu` (Hai tay bắt chéo trước ngực)**:
   - Khi hai cổ tay bắt chéo nhau, hiện tượng **Hand-on-Hand Occlusion** khiến MediaPipe bị mất dấu đồng thời cả 2 bàn tay trong **39 / 60 frames**!
   - Trong chế độ offline batch, model dự đoán đúng `cap cuu` với xác suất **99.98%**!
   - Nhưng trong runtime, việc mất dấu tay liên tục khiến hệ thống hiểu nhầm là người dùng đã hạ tay nghỉ (Idle) và liên tục reset buffer.

> **KẾT LUẬN CỐT TỬ CỦA TASK D2**:
> 1. **Nguyên nhân gốc của 2 từ `toi bi dau dau` và `cap cuu` KHÔNG PHẢI do Consensus quá chặt**, mà nằm ở:
>    - Tầng Feature Extraction: **MediaPipe Holistic bị che khuất (Occlusion)** khi tay chạm đầu hoặc hai tay bắt chéo.
>    - Tầng Quản lý Trạng thái: **Ngưỡng Idle check (0.5s)** quá nhạy, tự động dọn sạch buffer khi mất dấu tay tạm thời trong quá trình ký.
> 2. **Nới lỏng Consensus sang Majority 8/10**:
>    - Cứu được các từ bị nhiễu micro-jitter nhẹ (tăng Success Rate từ 60.0% lên 66.7%).
>    - Chi phí False Positive tăng nhẹ (+3 lần trên 45s nhiễu).

---

### 4. Đề Xuất Lựa Chọn & Hướng Đi Tiếp Theo

1. **Về cơ chế Consensus**:
   - **Chọn Biến thể 1 (Majority Vote $\ge 8/10$ + Mean Confidence > 0.5)** làm cấu hình chuẩn thay thế cho 100% consensus hiện tại:
     - Giảm tỷ lệ đứng hình từ 40.0% xuống 33.3%.
     - Không cần duy trì thêm state EMA phức tạp (Biến thể 2 cho kết quả tương đương Biến thể 1).
2. **Về giải quyết triệt để 2 từ Occlusion (`toi bi dau dau`, `cap cuu`)**:
   - Cần bổ sung cơ chế **"Dropout Tolerance" trong Idle Detection**: khi mất dấu tay ngắn dưới $0.7 - 0.8$s trong lúc đang thực hiện cử chỉ dở dang, thay vì xóa sạch buffer, tiếp tục giữ nguyên tư thế cuối cùng (hold last valid landmarks) hoặc chỉ reset khi thật sự hạ tay xuống vùng thắt lưng/ngoài khung hình.
3. **Về False Positive**:
   - Cần bổ sung lớp threshold động hoặc bộ lọc OOD (Out-of-Distribution) / No-gesture detector để triệt tiêu việc phát nhãn khi gõ phím hoặc di chuyển tay casual.

- **File đính kèm**: Biểu đồ đối đầu 4 phương án [docs/task_d2_relaxation_comparison.png](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_d2_relaxation_comparison.png)
