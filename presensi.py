import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
from dataset_trainer import load_dataset

# Direktori untuk menyimpan log
LOG_DIR = "attendance_logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "attendance_logs.txt")

# Load dataset dari file eksternal
print("Loading dataset...")
known_encodings, known_names = load_dataset()

# Inisialisasi Kamera
cap = cv2.VideoCapture(0)
cap.set(3, 800)  # Lebar
cap.set(4, 480)  # Tinggi

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"
        
        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]
            
            # Simpan log presensi
            with open(LOG_FILE, "a") as log:
                log.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {name} hadir\n")
                print(f"Logged: {name} hadir")
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0) if name != "Unknown" else (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if name != "Unknown" else (0, 0, 255), 2)
    
    # Tampilkan hasil dalam layar penuh
    cv2.namedWindow("Face Recognition", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Face Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Face Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
