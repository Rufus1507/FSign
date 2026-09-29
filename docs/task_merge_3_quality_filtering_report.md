# TASK MERGE-3: BÁO CÁO ÁP DỤNG TIÊU CHÍ LỌC CHẤT LƯỢNG LÊN TOÀN BỘ 100 NHÃN VIDEO
*(Báo Cáo Nghiệm Thu Chốt Tập Dữ Liệu Hợp Nhất Sau Lọc Định Lượng & Chất Lượng)*

> **Thông tin kiểm định**:
> - **Ngày thực hiện**: 27/09/2026
> - **Phạm vi áp dụng**: Toàn bộ 100 nhãn nguồn video từ `dataset/train/` kết hợp với 59 nhãn nguồn webcam từ `Data/`.
> - **Quy tắc lọc đồng thời 2 lớp**:
>   1. **Lớp 1 (Số lượng mẫu — Task MERGE-1)**: Số sequence $\ge 30$ mẫu (loại bỏ hiện tượng thiếu mẫu nghiêm trọng, bảo đảm tỷ lệ test 20% có ít nhất 6 mẫu).
>   2. **Lớp 2 (Chất lượng video gốc — Task MERGE-2/3)**: Loại bỏ các nhãn có độ dài video gốc $< 20.0$ frames (loại video $< 1.0\text{s}$) **HOẶC** tốc độ khung hình trung bình $\text{FPS} < 10.0\text{ fps}$ (loại bỏ video quay giật cục, thiếu thông tin động học).

---

## 1. BẢNG DANH SÁCH ĐẦY ĐỦ 40 NHÃN NGUỒN VIDEO BỊ LOẠI

Trong tổng số 100 nhãn video mới, có đúng **40 nhãn bị loại** do vi phạm một hoặc cả hai tiêu chí:

| STT | Tên nhãn bị loại | Số Sequence | FPS Gốc | Frames Gốc | Nhóm Lý Do Bị Loại | Phân Tích Cụ Thể |
| :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| 1 | **Bế mạc** | **35** | **`9.8`** | 30.0 | **Chất lượng video kém** | Đủ mẫu (35), nhưng FPS < 10.0 (9.8 fps) $\rightarrow$ Giật cục, mất thông tin vi gia tốc |
| 2 | **Ghét** | **32** | 14.8 | **`27.2` (min 16)** | **Chất lượng video kém** | Đủ mẫu (32), nhưng video gốc ngắn nhất 16 frames (0.8s) $\rightarrow$ Bị kéo giãn $3.75\times$ |
| 3 | **Thức ăn** | **56** | **`9.7`** | 30.0 | **Chất lượng video kém** | Đủ mẫu (56), nhưng FPS < 10.0 (9.7 fps) $\rightarrow$ Cảnh báo FPS thấp x5 |
| 4 | **Bộ y tế** | **17** | **`9.6`** | 30.0 | **Cả 2 lý do** | Thiếu mẫu nghiêm trọng (17 < 30) VÀ FPS < 10.0 (9.6 fps) |
| 5 | **Chúng ta** | **14** | **`9.9`** | 30.0 | **Cả 2 lý do** | Thiếu mẫu nghiêm trọng (14 < 30) VÀ FPS < 10.0 (9.9 fps) |
| 6 | **Bệnh nhân** | **6** | 13.1 | 30.0 | **Thiếu mẫu nghiêm trọng** | Chỉ 6 mẫu $\rightarrow$ Tập Test 20% chỉ có đúng 1 mẫu |
| 7 | **Cách ly** | **6** | 12.5 | 30.0 | **Thiếu mẫu nghiêm trọng** | Chỉ 6 mẫu $\rightarrow$ Tập Test 20% chỉ có đúng 1 mẫu |
| 8 | **Cơ thể** | **6** | 12.8 | 30.0 | **Thiếu mẫu nghiêm trọng** | Chỉ 6 mẫu $\rightarrow$ Tập Test 20% chỉ có đúng 1 mẫu |
| 9 | **Ho** | **6** | 16.3 | 30.0 | **Thiếu mẫu nghiêm trọng** | Chỉ 6 mẫu $\rightarrow$ Tập Test 20% chỉ có đúng 1 mẫu |
| 10 | **Phục hồi** | **6** | 12.0 | 30.0 | **Thiếu mẫu nghiêm trọng** | Chỉ 6 mẫu $\rightarrow$ Tập Test 20% chỉ có đúng 1 mẫu |
| 11 | **Sốt** | **6** | 14.0 | 30.0 | **Thiếu mẫu nghiêm trọng** | Chỉ 6 mẫu $\rightarrow$ Tập Test 20% chỉ có đúng 1 mẫu |
| 12 | **Xe máy** | **6** | 13.4 | 30.0 | **Thiếu mẫu nghiêm trọng** | Chỉ 6 mẫu $\rightarrow$ Tập Test 20% chỉ có đúng 1 mẫu |
| 13 | **Ủng hộ** | **6** | 12.2 | 30.0 | **Thiếu mẫu nghiêm trọng** | Chỉ 6 mẫu $\rightarrow$ Tập Test 20% chỉ có đúng 1 mẫu |
| 14 | **Ban đêm** | **7** | 12.7 | 30.0 | **Thiếu mẫu nghiêm trọng** | Chỉ 7 mẫu $\rightarrow$ Tập Test 20% chỉ có đúng 1 mẫu |
| 15 | **Chấp nhận** | **7** | 15.0 | 30.0 | **Thiếu mẫu nghiêm trọng** | Chỉ 7 mẫu $\rightarrow$ Tập Test 20% chỉ có đúng 1 mẫu |
| 16 | **Cần** | **7** | 11.8 | 30.0 | **Thiếu mẫu nghiêm trọng** | Chỉ 7 mẫu $\rightarrow$ Tập Test 20% chỉ có đúng 1 mẫu |
| 17 | **Học sinh** | **7** | 11.3 | 30.0 | **Thiếu mẫu nghiêm trọng** | Chỉ 7 mẫu $\rightarrow$ Tập Test 20% chỉ có đúng 1 mẫu |
| 18 | **Lo lắng** | **7** | 12.5 | 30.0 | **Thiếu mẫu nghiêm trọng** | Chỉ 7 mẫu $\rightarrow$ Tập Test 20% chỉ có đúng 1 mẫu |
| 19 | **Ngón tay** | **7** | 10.0 | 30.0 | **Thiếu mẫu nghiêm trọng** | Chỉ 7 mẫu $\rightarrow$ Tập Test 20% chỉ có đúng 1 mẫu |
| 20 | **Nhầm** | **7** | 10.9 | 30.0 | **Thiếu mẫu nghiêm trọng** | Chỉ 7 mẫu $\rightarrow$ Tập Test 20% chỉ có đúng 1 mẫu |
| 21 | **Virus** | **7** | 11.0 | 30.0 | **Thiếu mẫu nghiêm trọng** | Chỉ 7 mẫu $\rightarrow$ Tập Test 20% chỉ có đúng 1 mẫu |
| 22 | **Bàn tay** | **8** | 12.1 | 30.0 | **Thiếu mẫu nghiêm trọng** | Chỉ 8 mẫu $\rightarrow$ Tập Test 20% chỉ có đúng 1 mẫu |
| 23 | **Thương** | **8** | 14.1 | 30.0 | **Thiếu mẫu nghiêm trọng** | Chỉ 8 mẫu $\rightarrow$ Tập Test 20% chỉ có đúng 1 mẫu |
| 24 | **Vâng lời** | **8** | 14.7 | 30.0 | **Thiếu mẫu nghiêm trọng** | Chỉ 8 mẫu $\rightarrow$ Tập Test 20% chỉ có đúng 1 mẫu |
| 25 | **Khu cách ly** | **11** | 16.7 | 30.0 | **Thiếu mẫu** | 11 mẫu (< 30) |
| 26 | **Khẩu trang** | **11** | 10.9 | 30.0 | **Thiếu mẫu** | 11 mẫu (< 30) |
| 27 | **Thích** | **11** | 17.9 | 30.0 | **Thiếu mẫu** | 11 mẫu (< 30) |
| 28 | **Tập luyện** | **11** | 13.0 | 30.0 | **Thiếu mẫu** | 11 mẫu (< 30) |
| 29 | **Rẽ phải** | **15** | 10.5 | 30.0 | **Thiếu mẫu** | 15 mẫu (< 30) |
| 30 | **Xe đạp** | **16** | 11.5 | 30.0 | **Thiếu mẫu** | 16 mẫu (< 30) |
| 31 | **Bạn thân** | **18** | 11.3 | 30.0 | **Thiếu mẫu** | 18 mẫu (< 30) |
| 32 | **Đẹp** | **18** | 11.0 | 30.0 | **Thiếu mẫu** | 18 mẫu (< 30) |
| 33 | **Thăm** | **19** | 18.9 | 30.0 | **Thiếu mẫu** | 19 mẫu (< 30) |
| 34 | **Xin lỗi** | **19** | 11.3 | 30.0 | **Thiếu mẫu** | 19 mẫu (< 30) |
| 35 | **Có thể** | **22** | 12.8 | 30.0 | **Thiếu mẫu** | 22 mẫu (< 30) |
| 36 | **Ban ngày** | **23** | 10.1 | 30.0 | **Thiếu mẫu** | 23 mẫu (< 30) |
| 37 | **Thất lạc** | **24** | 16.2 | 30.0 | **Thiếu mẫu** | 24 mẫu (< 30) |
| 38 | **Tối** | **24** | 14.5 | 30.0 | **Thiếu mẫu** | 24 mẫu (< 30) |
| 39 | **Uống** | **25** | 12.2 | 30.0 | **Thiếu mẫu** | 25 mẫu (< 30) |
| 40 | **Hôm nay** | **26** | 13.7 | 30.0 | **Thiếu mẫu** | 26 mẫu (< 30) |

### Thống kê phân loại lý do bị loại:
- **Bị loại do chất lượng video kém thuần túy**: **3 nhãn** (`Bế mạc`, `Ghét`, `Thức ăn`) — Tổng mẫu: **123 sequences**.
- **Bị loại do cả 2 lý do (thiếu mẫu VÀ FPS < 10)**: **2 nhãn** (`Bộ y tế`, `Chúng ta`) — Tổng mẫu: **31 sequences**.
- **Bị loại do thiếu mẫu thuần túy (< 30 samples)**: **35 nhãn** — Tổng mẫu: **421 sequences**.
- **Tổng số mẫu bị loại**: $123 + 31 + 421 = \mathbf{575 \text{ sequences}}$ (chỉ chiếm $7.7\%$ toàn bộ dataset 7,415 mẫu).

---

## 2. TỔNG HỢP SỐ LIỆU SAU KHI LỌC ĐẦY ĐỦ 2 LỚP TIÊU CHÍ

| Nhóm dữ liệu | Số lượng nhãn | Số lượng sequence | Tỷ lệ số mẫu | Mẫu TB / Nhãn | Đặc tính kỹ thuật |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Nhóm Webcam Chuẩn (Giữ 100%)** | **59 nhãn** | **3,540 sequences** | **51.8%** | **60.0** | 100% video thực tế 30 FPS, cân bằng tuyệt đối |
| **Nhóm Video Đạt Chuẩn (Đã lọc)** | **60 nhãn** | **3,300 sequences** | **48.2%** | **55.0** | $FPS \ge 10.0$, độ dài $\ge 20$ frames, mẫu $\ge 30$ |
| **TẬP CHỐT HỢP NHẤT (FINAL)** | **`119 nhãn`** | **`6,840 sequences`** | **`100.0%`** | **`57.5`** | **Cân bằng tỷ lệ nguồn 52% : 48%, siêu sạch** |

---

## 3. DANH SÁCH 60 NHÃN NGUỒN VIDEO ĐẠT CHUẨN ĐƯỢC GIỮ LẠI

Toàn bộ 60 nhãn video dưới đây đều thỏa mãn đồng thời $\text{Số mẫu} \ge 30$, $\text{FPS} \ge 10.0$ và $\text{Độ dài} \ge 20$ frames:

| STT | Tên nhãn | Số mẫu | FPS avg | STT | Tên nhãn | Số mẫu | FPS avg | STT | Tên nhãn | Số mẫu | FPS avg |
| :---: | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :--- | :---: | :---: |
| 1 | **An ủi** | 32 | 11.9 | 21 | **Giúp** | 45 | 20.1 | 41 | **Phạt** | 37 | 17.5 |
| 2 | **Biết** | 67 | 15.9 | 22 | **Hy sinh** | 47 | 10.9 | 42 | **Phỏng vấn** | 45 | 17.9 |
| 3 | **Biếu tặng** | 51 | 12.8 | 23 | **Hâm mộ** | 50 | 13.5 | 43 | **Rau** | 72 | 14.0 |
| 4 | **Băn khoăn** | 62 | 18.7 | 24 | **Họ** | 48 | 10.1 | 44 | **Rẽ trái** | 65 | 10.6 |
| 5 | **Bệnh viện** | 48 | 10.3 | 25 | **Khai báo** | 56 | 17.5 | 45 | **San sẻ** | 46 | 10.6 |
| 6 | **Chiều** | 60 | 12.7 | 26 | **Khóc** | 67 | 12.1 | 46 | **Sử dụng** | 59 | 13.5 |
| 7 | **Cho** | 49 | 12.4 | 27 | **Kết hôn** | 31 | 14.3 | 47 | **Thức dậy** | 67 | 11.7 |
| 8 | **Chào** | 34 | 11.8 | 28 | **Lây bệnh** | 50 | 12.2 | 48 | **Trưa** | 57 | 14.4 |
| 9 | **Chân** | 62 | 11.0 | 29 | **Mua** | 35 | 13.0 | 49 | **Trường học** | 74 | 10.9 |
| 10 | **Chạy** | 67 | 13.9 | 30 | **Mời vào** | 56 | 11.3 | 50 | **Tôi** | 52 | 12.5 |
| 11 | **Chậm lại** | 69 | 10.3 | 31 | **Nghe** | 68 | 12.4 | 51 | **Xa** | 62 | 14.0 |
| 12 | **Con gấu** | 57 | 11.7 | 32 | **Nghỉ ngơi** | 67 | 14.2 | 52 | **Xin phép** | 38 | 13.3 |
| 13 | **Cá** | 58 | 11.1 | 33 | **Nhà** | 71 | 12.8 | 53 | **Xuất viện** | 48 | 10.3 |
| 14 | **Cám dỗ** | 68 | 15.8 | 34 | **Nhìn** | 60 | 11.7 | 54 | **Xúc động** | 59 | 10.9 |
| 15 | **Cảm ơn** | 62 | 13.4 | 35 | **Nhớ** | 57 | 11.6 | 55 | **Áp dụng** | 71 | 12.0 |
| 16 | **Cứu** | 53 | 12.0 | 36 | **Nói** | 30 | 11.3 | 56 | **Ô tô** | 47 | 12.4 |
| 17 | **Dạy dỗ** | 42 | 14.8 | 37 | **Nói xấu** | 68 | 16.9 | 57 | **Ăn** | 60 | 14.4 |
| 18 | **Dễ** | 51 | 12.9 | 38 | **Nôn ói** | 44 | 13.7 | 58 | **Ăn mừng** | 62 | 14.6 |
| 19 | **Đi** | 62 | 12.2 | 39 | **Nặng** | 48 | 12.9 | 59 | **Đâu** | 53 | 15.2 |
| 20 | **Đầu** | 54 | 11.8 | 40 | **Phía sau** | 67 | 15.9 | 60 | **Đồng ý** | 53 | 11.2 |

---

## 4. KẾT LUẬN & ĐÁNH GIÁ Ý NGHĨA KHOA HỌC

1. **Tính cân bằng tối ưu**: Tỷ lệ mẫu giữa Webcam và Video sau lọc là **51.8% : 48.2%** (gần như 1:1 tuyệt đối). Mức cân bằng này triệt tiêu hoàn toàn nguy cơ thiên lệch (Bias) sang bất kỳ nguồn thu thập nào.
2. **Loại bỏ triệt để rủi ro may rủi**: Không còn bất kỳ nhãn nào dưới 30 mẫu. Mọi nhãn đều bảo đảm có từ 6 đến 15 mẫu kiểm thử độc lập, bảo vệ vững chắc tính khách quan của F1-Score khi trình bày trước Hội đồng chấm Luận văn.
3. **Loại bỏ méo mó động học**: 3 nhãn có hiện tượng kéo giãn quá $3\times$ hoặc tốc độ dưới 10 fps (`Ghét`, `Bế mạc`, `Thức ăn`) đã được loại bỏ, ngăn chặn nguy cơ sai lệch miền (Domain Shift) khi mô hình chạy suy luận trực tiếp trên Camera thời gian thực.

---
*Báo cáo được hoàn thành trong khuôn khổ Task MERGE-3 của dự án FSign.*
