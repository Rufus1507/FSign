[CẬP NHẬT SAU BÁO CÁO LẦN 1] Task A4 — Đo lại CV đúng kịch bản (bắt buộc)
Lý do: Báo cáo Task A1 dùng sequence "chuc mung" (không có động tác nắm/xòe tay rõ rệt), nên chưa kiểm chứng được đúng giả thuyết ban đầu (co ngón làm scale cũ mất ổn định). CV đo được của scale mới còn cao hơn scale cũ trên data này — không dùng được để kết luận.

Nguyên tắc: làm từng task nhỏ một, báo cáo xong mới nhận task tiếp theo — không gộp nhiều bước vào 1 lần chạy/1 báo cáo.

Task A4.1 — Chuẩn bị video test chuyên biệt
Quay/lấy 1 đoạn video: người ký nắm tay lại → xòe ra → lặp lại 5-6 lần, giữ cố định vị trí đứng và hạn chế xoay cổ tay (yaw/roll) trong lúc quay.
Chạy MediaPipe Hands trên video, lưu lại raw landmark sequence (chưa tính scale gì cả) ra file (.npy/.json) để dùng chung cho A4.2–A4.4.
Báo cáo: số frame thu được, xác nhận video có thể hiện rõ động tác nắm/xòe (mô tả ngắn hoặc 1-2 frame ảnh minh họa).
Task A4.2 — Tính CV cho S_old
Trên raw landmark sequence từ A4.1, tính S_old = ‖wrist(0) − middle_tip(12)‖ cho từng frame.
Báo cáo: mean, std, CV.
Task A4.3 — Tính CV cho S_alt
Trên cùng sequence, tính S_alt = ‖MCP5 − MCP17‖ cho từng frame.
Báo cáo: mean, std, CV.
Task A4.4 — Tính CV cho S_combined
Trên cùng sequence, tính S_combined = sqrt(‖wrist−MCP9‖² + ‖MCP5−MCP17‖²) cho từng frame.
Báo cáo: mean, std, CV.
Task A4.5 — Tổng hợp & kết luận
Gộp kết quả A4.2–A4.4 thành 1 bảng so sánh CV của cả 3 công thức trên cùng 1 video.
Kết luận công thức nào ổn định nhất trong kịch bản nắm/xòe tay, đề xuất công thức chốt để dùng ở Task A5.
Task A5 — Triển khai chính thức (chỉ bắt đầu sau khi A4.5 có kết luận)
Nguyên tắc: làm từng task nhỏ, báo cáo xong mới sang task kế.

Task A5.1 — Cập nhật hàm chuẩn hóa
Sửa _normalize_one_hand() trong preprocessing.py, dùng công thức scale đã chốt ở A4.5.
Chạy thử trên 1 sequence mẫu, báo cáo output shape và vài giá trị mẫu để xác nhận không lỗi.
Task A5.2 — Thêm vector tương đối 2 cổ tay
Thêm (Wrist_RH − Wrist_LH), 3 chiều, nối vào cuối feature vector hiện tại → tổng 129 chiều (thay vì 126).
Chạy thử trên 1 sequence có 2 tay, báo cáo output shape (phải ra (60, 129)) và xác nhận 3 chiều cuối đúng là vector tương đối (khác 0 khi 2 tay không trùng vị trí).
Task A5.3 — Regenerate toàn bộ dataset
Chạy lại pipeline chuẩn hóa (đã có A5.1+A5.2) trên toàn bộ Data/ → ghi đè/tạo mới Data_normalized/.
Báo cáo: số sequence xử lý thành công, số lỗi (nếu có), thời gian chạy.
Task A5.4 — Verify & báo cáo tổng kết
Chạy lại verify_pipeline.py và check_data_balance.py trên Data_normalized/ mới.
Báo cáo: có lỗi shape/NaN nào không, phân bố nhãn có còn cân bằng như trước không.
Không bắt đầu Hạng mục B trước khi Task A5.4 xác nhận sạch — vì thay đổi input dimension (126→129) sẽ làm mọi thí nghiệm kiến trúc ở B phải chạy lại nếu làm trước.
