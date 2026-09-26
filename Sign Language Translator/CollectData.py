import cv2
import numpy as np
import os
import mediapipe as mp

# Chuyển working directory về thư mục chứa file script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# Khởi tạo MediaPipe Holistic
mp_hands = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    # Vẽ bàn tay trái
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    )
    # Vẽ bàn tay phải
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

def collect_data_for_action(action_name, start_seq=0, num_sequences=60, sequence_length=60, data_path='Data'):
    """
    Thu thập dữ liệu cử chỉ từ webcam và lưu thành các file .npy
    """
    # Tạo cấu trúc thư mục lưu dữ liệu
    for seq in range(start_seq, start_seq + num_sequences):
        seq_dir = os.path.join(data_path, action_name, str(seq))
        os.makedirs(seq_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Lỗi] Không thể mở webcam!")
        return

    print(f"\n=> Bắt đầu thu thập dữ liệu cho hành động: '{action_name}'")
    print(f"=> Số video (sequence): {num_sequences}, Mỗi video: {sequence_length} frames")
    print("=> Nhấn 'q' bất cứ lúc nào để dừng.\n")

    with mp_hands.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for sequence in range(start_seq, start_seq + num_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    break

                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                # Thông báo khi bắt đầu quay 1 sequence mới
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Action: {action_name} | Video: {sequence}', (15, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed - FSign Collection', image)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, f'Action: {action_name} | Video: {sequence} | Frame: {frame_num}/{sequence_length}', (15, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed - FSign Collection', image)

                # Trích xuất và lưu keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(data_path, action_name, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    print("\n[Dừng] Người dùng đã hủy thu thập.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n=> Hoàn tất thu thập dữ liệu cho: '{action_name}'!")

def test_camera_landmarks():
    """
    Test nhanh camera và hiển thị landmark
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Lỗi] Không thể mở webcam!")
        return

    print("=> Đang test camera. Nhấn 'q' để thoát.")
    with mp_hands.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            cv2.putText(image, 'TESTING MODE (Press q to exit)', (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed - Test', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("=== THU THẬP DỮ LIỆU CỬ CHỈ (FSign) ===")
    print("1. Test camera và nhận diện bàn tay")
    print("2. Thu thập dữ liệu mới cho 1 câu/cử chỉ")
    choice = input("Lựa chọn (1 hoặc 2): ").strip()

    if choice == '1':
        test_camera_landmarks()
    elif choice == '2':
        new_action = input("Nhập tên câu/cử chỉ mới (không dấu hoặc có dấu): ").strip()
        if new_action:
            collect_data_for_action(new_action, start_seq=0, num_sequences=60, sequence_length=60)
        else:
            print("Tên không hợp lệ.")
    else:
        print("Lựa chọn không hợp lệ.")
