from ultralytics import YOLO
import cv2

# Load model YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Indeks keypoints untuk titik yang ingin ditampilkan
KEYPOINTS_TO_SHOW = [5, 6, 7, 8, 9, 10, 11, 12]  # Pundak, siku, tangan, lutut, kaki

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Hentikan loop jika kamera tidak bisa membaca frame

    # Deteksi pose
    results = model(frame)

    for result in results:
        # Ambil keypoints
        keypoints = result.keypoints.xy.cpu().numpy()  # Konversi ke numpy array

        # Gambar titik yang dipilih
        for kp_idx in KEYPOINTS_TO_SHOW:
            if kp_idx < len(keypoints[0]):  # Pastikan indeks valid
                x, y = keypoints[0][kp_idx]
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Gambar titik hijau

    # Tampilkan hasil
    cv2.imshow("YOLOv8 Pose Estimation (Filtered)", frame)

    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bebaskan kamera dan tutup jendela
cap.release()
cv2.destroyAllWindows()