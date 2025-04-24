import os
import cv2
import time
from datetime import datetime
from threading import Thread
from deepface import DeepFace
import numpy as np

# Konfigurasi
RESOLUTION = (640, 480)  
DATASET_PATH = "dataset"
LOG_PATH = "log_presensi"
MODEL_NAME = "Facenet"  # Fixed capitalization
DETECTOR_BACKEND = "ssd"
FACE_MATCH_THRESHOLD = 0.4
MIN_CONFIDENCE = 0.8
FRAME_SKIP = 2

def load_embeddings():
    embeddings = {}
    for name in os.listdir(DATASET_PATH):
        if not os.path.isdir(os.path.join(DATASET_PATH, name)):
            continue
        embeddings[name] = []
        for img_file in os.listdir(os.path.join(DATASET_PATH, name)):
            img_path = os.path.join(DATASET_PATH, name, img_file)
            embedding = DeepFace.represent(
                img_path=img_path,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False
            )
            embeddings[name].append(embedding)
    return embeddings

class FaceVerifier(Thread):
    def __init__(self, face_img, embeddings):
        Thread.__init__(self)
        self.face_img = face_img
        self.embeddings = embeddings
        self.result = None
    
    def run(self):
        try:
            # Dapatkan embedding dengan bentuk yang konsisten
            target_embedding = np.array(DeepFace.represent(
                img_path=self.face_img,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False
            )).flatten()  # Pastikan bentuk (n,)
            
            best_match = {"name": None, "distance": float('inf')}
            
            for name, emb_list in self.embeddings.items():
                for emb in emb_list:
                    # Pastikan embedding referensi juga flat
                    emb_array = np.array(emb).flatten()
                    distance = np.linalg.norm(target_embedding - emb_array)
                    
                    if distance < best_match["distance"]:
                        best_match = {"name": name, "distance": distance}
            
            # Debug: Print distance terbaik
            print(f"Best match distance: {best_match['distance']:.4f}")
            
            self.result = best_match if best_match["distance"] < FACE_MATCH_THRESHOLD else None
            
        except Exception as e:
            print(f"[Error] Verifikasi: {str(e)}")
            import traceback
            traceback.print_exc()
            
def main():
    embeddings = load_embeddings()
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    cv2.ocl.setUseOpenCL(True)
    
    frame_count = 0
    last_attendance = {}
    
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue
        
        try:
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
                align=False
            )
            
            for face in faces:
                if face["confidence"] < MIN_CONFIDENCE:
                    continue
                
                # FIXED: Access facial_area as dictionary
                facial_area = face["facial_area"]
                x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                verifier = FaceVerifier(frame[y:y+h, x:x+w], embeddings)
                verifier.start()
                verifier.join(timeout=0.5)
                
                if verifier.result:
                    name = verifier.result["name"]
                    accuracy = (1 - verifier.result["distance"]) * 100
                    
                    today = datetime.now().strftime("%Y-%m-%d")
                    if last_attendance.get(name) == today:
                        cv2.putText(frame, f"{name} (Sudah Absen)", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        continue
                    
                    log_dir = os.path.join(LOG_PATH, today)
                    os.makedirs(log_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(os.path.join(log_dir, f"{name}_{timestamp}.jpg"), frame)
                    
                    last_attendance[name] = today
                    print(f"[Presensi] {name} - {accuracy:.1f}% - {timestamp}")
                    
        except Exception as e:
            print(f"[Error] Main loop: {str(e)}")
        
        cv2.imshow("Presensi", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()