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
COOLDOWN = 1  # seconds between detections
SKIP_FRAMES = 2  # Process every nth frame for recognition
DETECTION_PERSISTENCE = 2  # Number of frames to keep showing detection when not processing
PROCESSING_SIZE = (320, 240)  # Smaller size for processing to improve performance

# Create necessary directories
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

class FaceRecognitionSystem:
    def __init__(self):
        self.dataset = {}
        self.name_mapping = {}
        self.attendance_today = set()
        self.frame_queue = queue.Queue(maxsize=2)  # Allow 2 frames in queue
        self.result_queue = queue.Queue(maxsize=2)  # Allow 2 results in queue
        self.recognition_queue = queue.Queue(maxsize=1)  # Queue for face recognition tasks
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
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering for real-time
        
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
                time.sleep(0.001)  # Small sleep to free up CPU
        finally:
            camera.release()
    
    def face_detection_thread(self):
        """Dedicated thread just for face detection to maintain smooth tracking"""
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
                    # Resize for faster detection (but keep original for display)
                    small_frame = cv2.resize(frame, PROCESSING_SIZE)
                    
                    # Use DeepFace with SSD detector for face detection
                    faces = []
                    try:
                        # Only run detector on certain frames to maintain performance
                        if self.frame_count % SKIP_FRAMES == 0:
                            faces = DeepFace.extract_faces(
                                img_path=small_frame,
                                detector_backend='mtcnn',
                                enforce_detection=False,
                                align=False
                            )
                    except Exception as e:
                        print(f"Error in SSD detection: {str(e)}")
                        faces = []
                    
                    # Scale back to original frame size
                    scale_x = frame.shape[1] / PROCESSING_SIZE[0]
                    scale_y = frame.shape[0] / PROCESSING_SIZE[1]
                    
                    for face_data in faces:
                        if 'facial_area' in face_data:
                            # Get facial area coordinates
                            x_small = face_data['facial_area']['x']
                            y_small = face_data['facial_area']['y']
                            w_small = face_data['facial_area']['w']
                            h_small = face_data['facial_area']['h']
                            
                            # Scale back to original frame coordinates
                            x = int(x_small * scale_x)
                            y = int(y_small * scale_y)
                            w = int(w_small * scale_x)
                            h = int(h_small * scale_y)
                            
                            # Ensure dimensions are within frame bounds
                            x = max(0, x)
                            y = max(0, y)
                            w = min(w, frame.shape[1] - x)
                            h = min(h, frame.shape[0] - y)
                            
                            face_info = {
                                'box': (x, y, w, h),
                                'name': "Unknown",
                                'accuracy': 0,
                                'already_attended': False
                            }
                            
                            # Only queue for recognition if it's time
                            current_time = time.time()
                            if current_time - self.last_detection_time > COOLDOWN:
                                face_img = frame[y:y+h, x:x+w]
                                # Put in recognition queue if not full
                                if not self.recognition_queue.full():
                                    self.recognition_queue.put((face_img, face_info, clean_frame))
                                    self.last_detection_time = current_time
                            
                            detections.append(face_info)
                
                    # Reset counter if we have new detections
                    if detections:
                        self.detection_counter = 0
                        # Update last_detections only for faces that don't have pending recognition
                        # This preserves names and accuracy from previous recognitions
                        if self.last_detections:
                            # Create a mapping of face positions
                            last_faces = {(d['box'][0], d['box'][1]): d for d in self.last_detections}
                            
                            # For each new detection, check if we have a similar face in last_detections
                            for i, new_face in enumerate(detections):
                                new_x, new_y = new_face['box'][0], new_face['box'][1]
                                
                                # Find the closest face from last detections
                                closest_match = None
                                min_distance = 50  # Maximum allowed distance to consider same face
                                
                                for (last_x, last_y), last_face in last_faces.items():
                                    distance = np.sqrt((new_x - last_x)**2 + (new_y - last_y)**2)
                                    if distance < min_distance:
                                        min_distance = distance
                                        closest_match = last_face
                                
                                # If we found a close match, preserve the name and accuracy
                                if closest_match and closest_match['name'] != "Unknown":
                                    detections[i]['name'] = closest_match['name']
                                    detections[i]['accuracy'] = closest_match['accuracy']
                                    detections[i]['already_attended'] = closest_match['already_attended']
                        
                        self.last_detections = detections
                    else:
                        # Use previous detections when no new faces detected
                        if self.last_detections:
                            detections = self.last_detections
                            # Increment counter when using previous detections
                            self.detection_counter += 1
                            # Only keep showing detections for a limited number of frames
                            if self.detection_counter >= DETECTION_PERSISTENCE:
                                self.last_detections = []
                                detections = []
                
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
            
            time.sleep(0.001)  # Small sleep to prevent CPU hogging
    
    def recognition_thread(self):
        """Dedicated thread for face recognition to avoid impacting tracking FPS"""
        while self.running:
            try:
                face_img, face_info, clean_frame = self.recognition_queue.get(timeout=1)
                
                highest_accuracy = 0
                best_match_name = None
                is_already_attended = False
                
                for name_key, images in self.dataset.items():
                    # Only check first image for speed
                    img_path = images[0]
                    
                    try:
                        # Use DeepFace for actual recognition
                        result = DeepFace.verify(
                            img1_path=face_img,
                            img2_path=img_path,
                            model_name='Facenet',
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
                        continue
                
                # If we found a match with good confidence
                if highest_accuracy >= 75 and best_match_name:
                    x, y, w, h = face_info['box']
                    
                    # Update the face info in last_detections
                    for face in self.last_detections:
                        face_x, face_y, face_w, face_h = face['box']
                        # If this is the same face (close enough position)
                        if abs(face_x - x) < 30 and abs(face_y - y) < 30:
                            face['name'] = self.name_mapping[best_match_name]
                            face['accuracy'] = highest_accuracy
                            face['already_attended'] = is_already_attended
                    
                    # Only log attendance if person hasn't attended yet
                    if not is_already_attended:
                        # Log attendance with clean frame (no overlays)
                        today = datetime.now().strftime("%Y-%m-%d")
                        log_dir = os.path.join(LOG_PATH, today)
                        os.makedirs(log_dir, exist_ok=True)
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        log_filename = f"{self.name_mapping[best_match_name]}_{timestamp}.jpg"
                        
                        # Save the clean attendance log asynchronously
                        cv2.imwrite(os.path.join(log_dir, log_filename), clean_frame)
                        
                        print(f"Attendance: {self.name_mapping[best_match_name]} | Accuracy: {highest_accuracy:.1f}%")
                        self.update_attendance(best_match_name)
                    else:
                        print(f"Already attended: {self.name_mapping[best_match_name]} | Accuracy: {highest_accuracy:.1f}%")
                
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error in recognition: {str(e)}")
            
            time.sleep(0.001)  # Small sleep to prevent CPU hogging
    
    def display_thread(self):
        while self.running:
            try:
                frame, detections = self.result_queue.get(timeout=1)
                display_frame = frame.copy()  # Create a copy for display purposes
                
                # Draw FPS counter
                cv2.putText(display_frame, f"FPS: {self.fps}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Draw face detections with name and accuracy
                for face in detections:
                    x, y, w, h = face['box']
                    color = (0, 0, 255) if face['already_attended'] else (0, 255, 0)  # Red if attended, green if not
                    
                    # Draw rectangle around face
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Create label with name and accuracy
                    label = f"{face['name']}"
                    if face['accuracy'] > 0:
                        label += f" ({face['accuracy']:.1f}%)"
                    
                    # Calculate label position and draw background
                    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    y_label = max(y - 10, label_size[1])
                    
                    # Draw label background
                    cv2.rectangle(
                        display_frame, 
                        (x, y_label - label_size[1]), 
                        (x + label_size[0], y_label + base_line), 
                        (0, 0, 0), 
                        cv2.FILLED
                    )
                    
                    # Draw label text
                    cv2.putText(
                        display_frame, 
                        label, 
                        (x, y_label), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (255, 255, 255), 
                        1
                    )
                
                cv2.imshow('Face Recognition System', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
            
            except queue.Empty:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
            
            time.sleep(0.001)  # Small sleep to prevent CPU hogging
    
    def run(self):
        print("Starting Face Recognition System...")
        
        # Start threads
        capture_thread = threading.Thread(target=self.capture_thread)
        detection_thread = threading.Thread(target=self.face_detection_thread)
        recognition_thread = threading.Thread(target=self.recognition_thread)
        display_thread = threading.Thread(target=self.display_thread)
        
        capture_thread.daemon = True
        detection_thread.daemon = True
        recognition_thread.daemon = True
        display_thread.daemon = True
        
        capture_thread.start()
        detection_thread.start()
        recognition_thread.start()
        display_thread.start()
        
        # Wait for all threads to finish
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Shutting down...")
            self.running = False
        
        capture_thread.join()
        detection_thread.join()
        recognition_thread.join()
        display_thread.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.run()