import os
import cv2
import threading
import queue
import time
import numpy as np
from deepface import DeepFace
from datetime import datetime

# Configuration
RESOLUTION = (800, 480)  # Keeping original resolution as requested
DATASET_PATH = "cleaned_dataset"
LOG_PATH = "log_presensi"
MIN_ACCURACY = 0.8  # 80% confidence for face detection
FACE_MATCH_THRESHOLD = 0.65  # Optimal threshold for Facenet
COOLDOWN = 5  # seconds between detections
SKIP_FRAMES = 2  # Process every nth frame for recognition
DETECTION_PERSISTENCE = 5  # Number of frames to keep showing detection when not processing

# Create necessary directories
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

class FaceRecognitionSystem:
    def __init__(self):
        self.dataset = {}
        self.name_mapping = {}
        self.attendance_today = set()
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.last_detection_time = 0
        self.running = True
        self.frame_count = 0
        self.fps_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.last_detections = []  # Store previous detections
        self.detection_counter = 0  # Counter for detection persistence
        
        # Load dataset once at initialization
        self.load_dataset_and_attendance()
        
    def load_dataset_and_attendance(self):
        # Check attendance for today
        today = datetime.now().strftime("%Y-%m-%d")
        today_log_dir = os.path.join(LOG_PATH, today)
        
        if os.path.exists(today_log_dir):
            for filename in os.listdir(today_log_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    name = filename.split('_')[0].lower()
                    self.attendance_today.add(name)
        
        # Load dataset
        for filename in os.listdir(DATASET_PATH):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                base_name = "_".join(filename.split("_")[:-1])
                name_key = base_name.lower()
                
                if name_key not in self.dataset:
                    self.dataset[name_key] = []
                    self.name_mapping[name_key] = base_name
                
                self.dataset[name_key].append(os.path.join(DATASET_PATH, filename))
                
        print(f"Loaded {len(self.dataset)} identities from dataset")
        print(f"Already attended today: {len(self.attendance_today)} people")
        
    def update_attendance(self, name_key):
        self.attendance_today.add(name_key)
    
    def capture_thread(self):
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
        
        try:
            while self.running:
                ret, frame = camera.read()
                if not ret:
                    print("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Calculate FPS
                current_time = time.time()
                self.fps_count += 1
                if current_time - self.last_fps_time >= 1.0:
                    self.fps = self.fps_count
                    self.fps_count = 0
                    self.last_fps_time = current_time
                
                # Replace any existing frame in the queue
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.frame_queue.put(frame.copy())  # Use copy to ensure clean frame
        finally:
            camera.release()
    
    def process_thread(self):
        while self.running:
            try:
                # Get the latest frame
                frame = self.frame_queue.get(timeout=1)
                self.frame_count += 1
                
                # Keep a clean copy for logging purposes
                clean_frame = frame.copy()
                
                # Process the frame for face detection on every frame for smooth tracking
                detections = []
                try:
                    # Detect faces on every frame for smoother tracking
                    face_detections = DeepFace.extract_faces(
                        img_path=frame,
                        detector_backend='opencv',
                        enforce_detection=False,
                        align=False
                    )
                    
                    for detection in face_detections:
                        if detection['confidence'] > MIN_ACCURACY:
                            x, y, w, h = detection['facial_area']['x'], detection['facial_area']['y'], \
                                        detection['facial_area']['w'], detection['facial_area']['h']
                            
                            face_info = {
                                'box': (x, y, w, h),
                                'name': "Unknown",
                                'accuracy': 0,
                                'already_attended': False
                            }
                            
                            # Only do face recognition every SKIP_FRAMES frames to reduce processing
                            current_time = time.time()
                            if self.frame_count % SKIP_FRAMES == 0 and current_time - self.last_detection_time > COOLDOWN:
                                face_img = frame[y:y+h, x:x+w]
                                
                                highest_accuracy = 0
                                best_match_name = None
                                is_already_attended = False
                                
                                for name_key, images in self.dataset.items():
                                    # Only check first image for speed
                                    img_path = images[0]
                                    
                                    try:
                                        result = DeepFace.verify(
                                            img1_path=face_img,
                                            img2_path=img_path,
                                            model_name='Facenet',
                                            detector_backend='skip',  # Skip detection since we already have the face
                                            distance_metric='cosine',
                                            threshold=FACE_MATCH_THRESHOLD,
                                            enforce_detection=False,
                                            align=False
                                        )
                                        
                                        current_accuracy = (1 - result['distance']) * 100
                                        
                                        if current_accuracy > highest_accuracy:
                                            highest_accuracy = current_accuracy
                                            best_match_name = name_key
                                            # Check if this matched person already attended
                                            is_already_attended = (name_key in self.attendance_today)
                                            
                                    except Exception as e:
                                        pass
                                
                                # If we found a match with good confidence
                                if highest_accuracy >= 75 and best_match_name:
                                    face_info['name'] = self.name_mapping[best_match_name]
                                    face_info['accuracy'] = highest_accuracy
                                    face_info['already_attended'] = is_already_attended
                                    
                                    # Only log attendance if person hasn't attended yet
                                    if not is_already_attended:
                                        # Log attendance with clean frame (no overlays)
                                        today = datetime.now().strftime("%Y-%m-%d")
                                        log_dir = os.path.join(LOG_PATH, today)
                                        os.makedirs(log_dir, exist_ok=True)
                                        
                                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        log_filename = f"{face_info['name']}_{timestamp}.jpg"
                                        
                                        # Save the clean attendance log asynchronously
                                        threading.Thread(
                                            target=lambda: cv2.imwrite(
                                                os.path.join(log_dir, log_filename), 
                                                clean_frame  # Use the clean frame without any overlays
                                            )
                                        ).start()
                                        
                                        print(f"Attendance: {face_info['name']} | Accuracy: {highest_accuracy:.1f}%")
                                        self.update_attendance(best_match_name)
                                    else:
                                        print(f"Already attended: {face_info['name']} | Accuracy: {highest_accuracy:.1f}%")
                                    
                                    self.last_detection_time = current_time
                            
                            detections.append(face_info)
                
                    # Reset counter if we have new detections
                    if detections:
                        self.detection_counter = 0
                        self.last_detections = detections
                    else:
                        # Increment counter when no faces detected
                        self.detection_counter += 1
                        # Keep showing the last detection for a few frames to avoid flickering
                        if self.detection_counter < DETECTION_PERSISTENCE:
                            detections = self.last_detections
                
                except Exception as e:
                    print(f"Error in face detection: {str(e)}")
                
                # Put results in the queue, replacing any existing results
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.result_queue.put((frame, detections))
                
            except queue.Empty:
                pass
    
    def display_thread(self):
        while self.running:
            try:
                frame, detections = self.result_queue.get(timeout=1)
                display_frame = frame.copy()  # Create a copy for display purposes
                
                # Draw FPS counter
                cv2.putText(display_frame, f"FPS: {self.fps}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Draw face detections on display frame only - rectangle only, no text
                for face in detections:
                    x, y, w, h = face['box']
                    color = (0, 0, 255) if face['already_attended'] else (0, 255, 0)  # Red if attended, green if not
                    
                    # Draw rectangle around face (only on display frame)
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                
                cv2.imshow('Face Recognition System', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
            
            except queue.Empty:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
    
    def run(self):
        print("Starting Face Recognition System...")
        
        # Start threads
        capture_thread = threading.Thread(target=self.capture_thread)
        process_thread = threading.Thread(target=self.process_thread)
        display_thread = threading.Thread(target=self.display_thread)
        
        capture_thread.start()
        process_thread.start()
        display_thread.start()
        
        # Wait for all threads to finish
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Shutting down...")
            self.running = False
        
        capture_thread.join()
        process_thread.join()
        display_thread.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.run()