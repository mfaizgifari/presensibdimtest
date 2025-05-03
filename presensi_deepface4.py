import os
import cv2
import threading
import queue
import time
import numpy as np
from deepface import DeepFace
from datetime import datetime
import subprocess

RESOLUTION = (800, 480)
DATASET_PATH = "cleaned_dataset"
LOG_PATH = "log_presensi"
FACE_MATCH_THRESHOLD = 0.6
COOLDOWN = 1
SKIP_FRAMES = 1
DETECTION_PERSISTENCE = 2
PROCESSING_SIZE = (480, 480)

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

class FaceRecognitionSystem:
    def __init__(self):
        self.dataset = {}
        self.name_mapping = {}
        self.attendance_today = set()
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.recognition_queue = queue.Queue(maxsize=1)
        self.last_detection_time = 0
        self.running = True
        self.frame_count = 0
        self.fps_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.last_detections = []
        self.detection_counter = 0
        self.load_dataset_and_attendance()

    def load_dataset_and_attendance(self):
        today = datetime.now().strftime("%Y-%m-%d")
        today_log_dir = os.path.join(LOG_PATH, today)
        if os.path.exists(today_log_dir):
            for filename in os.listdir(today_log_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.attendance_today.add(filename.split('_')[0].lower())
        for filename in os.listdir(DATASET_PATH):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                base_name = "_".join(filename.split("_")[:-1])
                name_key = base_name.lower()
                self.dataset.setdefault(name_key, []).append(os.path.join(DATASET_PATH, filename))
                self.name_mapping[name_key] = base_name

    def update_attendance(self, name_key):
        self.attendance_today.add(name_key)

    def capture_thread(self):
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            while self.running:
                ret, frame = camera.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                current_time = time.time()
                self.fps_count += 1
                if current_time - self.last_fps_time >= 1.0:
                    self.fps = self.fps_count
                    self.fps_count = 0
                    self.last_fps_time = current_time
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put(frame.copy())
                time.sleep(0.001)
        finally:
            camera.release()

    def face_detection_thread(self):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                self.frame_count += 1
                clean_frame = frame.copy()
                detections = []
                small_frame = cv2.resize(frame, PROCESSING_SIZE)
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                scale_x = frame.shape[1] / PROCESSING_SIZE[0]
                scale_y = frame.shape[0] / PROCESSING_SIZE[1]
                for (x_small, y_small, w_small, h_small) in faces:
                    x, y, w, h = int(x_small * scale_x), int(y_small * scale_y), int(w_small * scale_x), int(h_small * scale_y)
                    face_info = {'box': (x, y, w, h), 'name': "Unknown", 'accuracy': 0, 'already_attended': False}
                    current_time = time.time()
                    if self.frame_count % SKIP_FRAMES == 0 and current_time - self.last_detection_time > COOLDOWN:
                        face_img = frame[y:y+h, x:x+w]
                        if not self.recognition_queue.full():
                            self.recognition_queue.put((face_img, face_info, clean_frame))
                            self.last_detection_time = current_time
                    detections.append(face_info)
                if detections:
                    self.detection_counter = 0
                    if self.last_detections:
                        last_faces = {(d['box'][0], d['box'][1]): d for d in self.last_detections}
                        for i, new_face in enumerate(detections):
                            new_x, new_y = new_face['box'][0], new_face['box'][1]
                            closest_match = None
                            min_distance = 50
                            for (last_x, last_y), last_face in last_faces.items():
                                distance = np.sqrt((new_x - last_x)**2 + (new_y - last_y)**2)
                                if distance < min_distance:
                                    min_distance = distance
                                    closest_match = last_face
                            if closest_match and closest_match['name'] != "Unknown":
                                detections[i].update({
                                    'name': closest_match['name'],
                                    'accuracy': closest_match['accuracy'],
                                    'already_attended': closest_match['already_attended']
                                })
                    self.last_detections = detections
                else:
                    self.detection_counter += 1
                    if self.detection_counter < DETECTION_PERSISTENCE:
                        detections = self.last_detections
                if self.result_queue.full():
                    self.result_queue.get_nowait()
                self.result_queue.put((frame, detections))
            except queue.Empty:
                pass
            time.sleep(0.001)

    def recognition_thread(self):
        while self.running:
            try:
                face_img, face_info, clean_frame = self.recognition_queue.get(timeout=1)
                highest_accuracy = 0
                best_match_name = None
                is_already_attended = False
                for name_key, images in self.dataset.items():
                    img_path = images[0]
                    try:
                        result = DeepFace.verify(
                            img1_path=face_img,
                            img2_path=img_path,
                            model_name='Facenet',
                            detector_backend='skip',
                            distance_metric='cosine',
                            threshold=FACE_MATCH_THRESHOLD,
                            enforce_detection=False,
                            align=False
                        )
                        current_accuracy = (1 - result['distance']) * 100
                        if current_accuracy > highest_accuracy:
                            highest_accuracy = current_accuracy
                            best_match_name = name_key
                            is_already_attended = (name_key in self.attendance_today)
                    except:
                        continue
                if highest_accuracy >= 75 and best_match_name:
                    x, y, w, h = face_info['box']
                    for face in self.last_detections:
                        if abs(face['box'][0] - x) < 30 and abs(face['box'][1] - y) < 30:
                            face.update({
                                'name': self.name_mapping[best_match_name],
                                'accuracy': highest_accuracy,
                                'already_attended': is_already_attended
                            })
                    if not is_already_attended:
                        today = datetime.now().strftime("%Y-%m-%d")
                        log_dir = os.path.join(LOG_PATH, today)
                        os.makedirs(log_dir, exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        log_filename = f"{self.name_mapping[best_match_name]}_{timestamp}.jpg"
                        cv2.imwrite(os.path.join(log_dir, log_filename), clean_frame)
                        print(f"Attendance: {self.name_mapping[best_match_name]} | Accuracy: {highest_accuracy:.1f}%")
                        self.update_attendance(best_match_name)
                        subprocess.run(["python", "presensi_firebase.py"])
                    else:
                        print(f"Already attended: {self.name_mapping[best_match_name]} | Accuracy: {highest_accuracy:.1f}%")
            except queue.Empty:
                pass
            time.sleep(0.001)

    def display_thread(self):
        while self.running:
            try:
                frame, detections = self.result_queue.get(timeout=1)
                display_frame = frame.copy()
                cv2.putText(display_frame, f"FPS: {self.fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                for face in detections:
                    x, y, w, h = face['box']
                    color = (0, 0, 255) if face['already_attended'] else (0, 255, 0)
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                    label = f"{face['name']} ({face['accuracy']:.1f}%)" if face['accuracy'] > 0 else face['name']
                    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    y_label = max(y - 10, label_size[1])
                    cv2.rectangle(display_frame, (x, y_label - label_size[1]), (x + label_size[0], y_label + base_line), (0, 0, 0), cv2.FILLED)
                    cv2.putText(display_frame, label, (x, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow('Face Recognition System', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
            except queue.Empty:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
            time.sleep(0.001)

    def run(self):
        capture_thread = threading.Thread(target=self.capture_thread, daemon=True)
        detection_thread = threading.Thread(target=self.face_detection_thread, daemon=True)
        recognition_thread = threading.Thread(target=self.recognition_thread, daemon=True)
        display_thread = threading.Thread(target=self.display_thread, daemon=True)
        capture_thread.start()
        detection_thread.start()
        recognition_thread.start()
        display_thread.start()
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.running = False
        capture_thread.join()
        detection_thread.join()
        recognition_thread.join()
        display_thread.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.run()