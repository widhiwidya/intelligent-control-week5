from ultralytics import YOLO
import mediapipe as mp
import cv2

# Load model YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Nama bagian tubuh berdasarkan indeks keypoints
BODY_PARTS = {
    0: "Hidung", 1: "Mata kiri", 2: "Mata kanan", 3: "Telinga kiri", 4: "Telinga kanan",
    5: "Pundak kiri", 6: "Pundak kanan", 7: "Siku kiri", 8: "Siku kanan",
    9: "Tangan kiri", 10: "Tangan kanan", 11: "Lutut kiri", 12: "Lutut kanan",
    13: "Kaki kiri", 14: "Kaki kanan"
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Konversi ke RGB untuk MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)
    
    # Deteksi pose dengan YOLOv8
    results = model(frame)
    
    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy()  # Ambil koordinat keypoints
        annotated_frame = frame.copy()
        
        for i, (x, y) in enumerate(keypoints[0]):
            if i in BODY_PARTS:
                cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Gambar titik
                cv2.putText(annotated_frame, BODY_PARTS[i], (int(x) + 5, int(y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)  # Label
    
    # Deteksi tangan dengan MediaPipe
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow("YOLOv8 Pose + MediaPipe Hands", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan kamera dan tutup semua jendela OpenCV
cap.release()
cv2.destroyAllWindows()