import cv2
import os
from deepface import DeepFace
from datetime import datetime
import numpy as np

# Direktori untuk menyimpan dataset wajah dan log
DATA_DIR = "dataset"
LOG_DIR = "attendance_logs"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "attendance_logs.txt")

# Load dataset wajah (simpan embedding wajah)
known_embeddings = []
known_names = []

def load_dataset():
    for file in os.listdir(DATA_DIR):
        img_path = os.path.join(DATA_DIR, file)
        name = os.path.splitext(file)[0]
        try:
            embedding = DeepFace.represent(img_path, model_name="VGG-Face", detector_backend="opencv")[0]["embedding"]
            known_embeddings.append(np.array(embedding))
            known_names.append(name)
            print(f"Loaded: {name}")
        except Exception as e:
            print(f"Error loading {name}: {e}")
    print("Dataset loaded successfully.")

# Muat dataset
print("Loading dataset...")
load_dataset()

correct_predictions = 0
total_predictions = 0

# Inisialisasi Kamera
cap = cv2.VideoCapture(0)
cap.set(3, 800)  # Lebar
cap.set(4, 480)  # Tinggi

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    try:
        # Deteksi wajah menggunakan DeepFace dengan OpenCV
        faces = DeepFace.detectFace(frame, detector_backend="opencv")
        for face in faces:
            x, y, w, h = face['facial_area'].values()
            face_img = frame[y:y+h, x:x+w]

            # Ekstrak embedding wajah dari frame saat ini
            current_embedding = DeepFace.represent(face_img, model_name="VGG-Face", detector_backend="opencv")[0]["embedding"]
            current_embedding = np.array(current_embedding)

            # Bandingkan dengan known embeddings
            distances = [np.linalg.norm(current_embedding - known_emb) for known_emb in known_embeddings]
            name = "Unknown"
            min_distance = min(distances) if distances else float('inf')

            # Asumsikan label sebenarnya (untuk pengujian, Anda perlu tahu siapa yang seharusnya terdeteksi)
            actual_name = "John"  # Ganti dengan nama sebenarnya atau logika untuk mendapatkan label benar

            if min_distance < 0.6:  # Threshold untuk kecocokan
                match_index = distances.index(min_distance)
                name = known_names[match_index]

                # Simpan log presensi
                with open(LOG_FILE, "a") as log:
                    log.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {name} hadir\n")
                    print(f"Logged: {name} hadir")

                # Hitung akurasi
                total_predictions += 1
                if name == actual_name:  # Bandingkan dengan label sebenarnya
                    correct_predictions += 1
                    print(f"Correct prediction: {name}")
                else:
                    print(f"Wrong prediction: Predicted {name}, Actual {actual_name}")

            # Tampilkan akurasi saat ini (opsional)
            if total_predictions > 0:
                accuracy = (correct_predictions / total_predictions) * 100
                print(f"Current Accuracy: {accuracy:.2f}%")

            # Gambar kotak di sekitar wajah
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    except Exception as e:
        print(f"Error in face detection: {e}")

    # Tampilkan hasil dalam layar penuh
    cv2.namedWindow("Face Recognition", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Face Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Tampilkan akurasi akhir
if total_predictions > 0:
    final_accuracy = (correct_predictions / total_predictions) * 100
    print(f"Final Accuracy: {final_accuracy:.2f}%")
else:
    print("No predictions made.")