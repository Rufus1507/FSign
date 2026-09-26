import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

# Chuyển working directory về thư mục chứa file script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# Khởi tạo mô hình MediaPipe Holistic
mp_hands = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Chuyển đổi màu BGR sang RGB
    image.flags.writeable = False                  # Tối ưu xử lý ảnh
    results = model.process(image)                 # Dự đoán landmark
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Chuyển lại màu BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)

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

# Danh sách 60 câu cử chỉ nhận diện
actions = np.array([
    'ban dang lam gi', 'ban di dau the', 'ban hieu ngon ngu ky hieu khong', 'ban hoc lop may',
    'ban khoe khong', 'ban muon gio roi', 'ban phai canh giac', 'ban ten la gi', 'ban tien bo day',
    'ban trong cau co the', 'bo me toi cung la nguoi Diec', 'cai nay bao nhieu tien', 
    'cai nay la cai gi', 'cam on', 'cap cuu', 'chuc mung', 'chung toi giao tiep voi nhau bang ngon ngu ky hieu', 
    'con yeu me', 'cong viec cua ban la gi', 'hen gap lai cac ban', 'mon nay khong ngon', 'toi bi chong mat', 
    'toi bi cuop', 'toi bi dau dau', 'toi bi dau hong', 'toi bi ket xe', 'toi bi lac', 'toi bi phan biet doi xu', 
    'toi cam thay rat hoi hop', 'toi cam thay rat vui', 'toi can an sang', 'toi can di ve sinh', 
    'toi can gap bac si', 'toi can phien dich', 'toi can thuoc', 'toi dang an sang', 'toi dang buon', 
    'toi dang o ben xe', 'toi dang o cong vien', 'toi dang phai cach ly', 'toi dang phan van', 'toi di sieu thi', 
    'toi di toi Ha Noi', 'toi doc kem', 'toi khoi benh roi', 'toi khong dem theo tien', 'toi khong hieu', 
    'toi khong quan tam', 'toi la hoc sinh', 'toi la nguoi Diec', 'toi la tho theu', 'toi lam viec o cua hang', 
    'toi nham dia chi', 'toi song o Ha Noi', 'toi thay doi bung', 'toi thay nho ban', 'toi thich an mi', 
    'toi thich phim truyen', 'toi viet kem', 'xin chao'
])

def build_model(num_actions):
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, activation='relu', input_shape=(60, 126)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_actions, activation='softmax'))
    return model

def run_realtime_detection(model):
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Lỗi] Không thể mở webcam!")
        return

    print("=> Bắt đầu nhận diện qua webcam. Nhấn 'q' để thoát.")
    with mp_hands.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Dự đoán landmark
            image, results = mediapipe_detection(frame, holistic)

            # Vẽ landmark
            draw_styled_landmarks(image, results)

            # Dự đoán hành động
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-60:]

            if len(sequence) == 60:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                pred_idx = np.argmax(res)
                predictions.append(pred_idx)

                if np.unique(predictions[-10:])[0] == pred_idx:
                    if res[pred_idx] > threshold:
                        predicted_action = actions[pred_idx]
                        if len(sentence) > 0:
                            if predicted_action != sentence[-1]:
                                sentence.append(predicted_action)
                        else:
                            sentence.append(predicted_action)

                if len(sentence) > 5:
                    sentence = sentence[-5:]

            # Hiển thị thanh kết quả
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(
                image, ' '.join(sentence), (3, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
            )

            cv2.imshow('OpenCV Feed - FSign', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    print("Khởi tạo mô hình...")
    model = build_model(len(actions))

    # Ưu tiên load weights từ Structure 5 hoặc release
    weights_path = os.path.join('Structure', 'Structure 5 (final)', '94,58.h5')
    if not os.path.exists(weights_path):
        weights_path = os.path.join('release', '94,58.h5')

    if os.path.exists(weights_path):
        print(f"Đang tải trọng số từ: {weights_path}")
        model.load_weights(weights_path)
        print("=> Tải trọng số thành công!")
    else:
        print(f"[Cảnh báo] Không tìm thấy file weights tại {weights_path}, chạy với model chưa train.")

    run_realtime_detection(model)
