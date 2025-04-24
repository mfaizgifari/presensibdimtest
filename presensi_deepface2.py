import os
import cv2
from deepface import DeepFace
from datetime import datetime
import time
import numpy as np

# Konfigurasi
RESOLUTION = (800, 480)
DATASET_PATH = "cleaned_dataset"
LOG_PATH = "log_presensi"
MIN_ACCURACY = 0.8  # 80% confidence for face detection
FACE_MATCH_THRESHOLD = 0.65  # Optimal threshold for Facenet
COOLDOWN = 5  # detik antara deteksi

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

def load_dataset_and_attendance():
    dataset = {}
    name_mapping = {}
    attendance_today = set()  # Menyimpan nama yang sudah absen hari ini
    
    # Cek presensi hari ini
    today = datetime.now().strftime("%Y-%m-%d")
    today_log_dir = os.path.join(LOG_PATH, today)
    
    if os.path.exists(today_log_dir):
        for filename in os.listdir(today_log_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                name = filename.split('_')[0].lower()
                attendance_today.add(name)
    
    # Load dataset
    for filename in os.listdir(DATASET_PATH):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            base_name = "_".join(filename.split("_")[:-1])
            name_key = base_name.lower()
            
            if name_key not in dataset:
                dataset[name_key] = []
                name_mapping[name_key] = base_name
            
            dataset[name_key].append(os.path.join(DATASET_PATH, filename))
    
    return dataset, name_mapping, attendance_today

# Inisialisasi kamera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

# Font settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_color = (0, 255, 0)  # Green
attended_color = (0, 0, 255)  # Red untuk yang sudah absen
accuracy_color = (0, 165, 255)  # Orange
thickness = 2

print("Sistem Presensi Face Recognition siap digunakan...")
last_detection_time = 0
try:
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Gagal mengambil frame dari kamera")
            break
        
        frame = cv2.resize(frame, RESOLUTION)
        today = datetime.now().strftime("%Y-%m-%d")
        
        try:
            dataset, name_mapping, attendance_today = load_dataset_and_attendance()
            detections = DeepFace.extract_faces(
                img_path=frame,
                detector_backend='opencv',
                enforce_detection=False,
                align=False
            )
            
            for detection in detections:
                if detection['confidence'] > MIN_ACCURACY:
                    x, y, w, h = detection['facial_area']['x'], detection['facial_area']['y'], \
                                 detection['facial_area']['w'], detection['facial_area']['h']
                    
                    # Gambar kotak di sekitar wajah
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    current_time = time.time()
                    if current_time - last_detection_time > COOLDOWN:
                        face_img = frame[y:y+h, x:x+w]
                        
                        for name_key, images in dataset.items():
                            # Skip jika sudah absen hari ini
                            if name_key in attendance_today:
                                cv2.putText(frame, f"{name_mapping[name_key]} (Sudah Absen)", 
                                           (x, y-10), font, font_scale, attended_color, thickness)
                                continue
                                
                            best_match = {'accuracy': 0, 'img_path': None}
                            
                            for img_path in images:
                                try:
                                    result = DeepFace.verify(
                                        img1_path=face_img,
                                        img2_path=img_path,
                                        model_name='Facenet',
                                        detector_backend='opencv',
                                        distance_metric='cosine',
                                        threshold=FACE_MATCH_THRESHOLD,
                                        enforce_detection=False,
                                        align=False
                                    )
                                    
                                    current_accuracy = (1 - result['distance']) * 100
                                    
                                    if current_accuracy > best_match['accuracy']:
                                        best_match = {
                                            'accuracy': current_accuracy,
                                            'result': result
                                        }
                                        
                                except Exception as e:
                                    print(f"Error verifikasi {img_path}: {str(e)}")
                            
                            # Jika akurasi cukup dan belum absen
                            if best_match['accuracy'] >= 75:  # Only accept if accuracy is 75% or higher
                                display_name = name_mapping[name_key]
                                accuracy_text = f"Accuracy: {best_match['accuracy']:.1f}%"
                                
                                cv2.putText(frame, display_name, (x, y-10), 
                                            font, font_scale, font_color, thickness)
                                cv2.putText(frame, accuracy_text, (x, y+h+25), 
                                            font, font_scale*0.8, accuracy_color, thickness)
                                
                                # Simpan log presensi
                                log_dir = os.path.join(LOG_PATH, today)
                                os.makedirs(log_dir, exist_ok=True)
                                
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                log_filename = f"{display_name}_{timestamp}.jpg"
                                cv2.imwrite(os.path.join(log_dir, log_filename), frame)
                                
                                print(f"Presensi: {display_name} | Accuracy: {best_match['accuracy']:.1f}%")
                                attendance_today.add(name_key)  # Tandai sudah absen
                                last_detection_time = current_time
                            else:
                                # Display "Unknown" if accuracy is below 75%
                                cv2.putText(frame, "", (x, y-10), 
                                            font, font_scale, attended_color, thickness)
                                print(f"Accuracy too low ({best_match['accuracy']:.1f}%), not accepted.")
                        
        except Exception as e:
            print(f"Error deteksi wajah: {str(e)}")
        
        cv2.imshow('Presensi Wajah', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()