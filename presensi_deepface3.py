import os
import cv2
import threading
import queue
import time
import numpy as np
from deepface import DeepFace
from datetime import datetime
import pickle
import firebase_admin
from firebase_admin import credentials, storage, db
import uuid
import subprocess

# Configuration
RESOLUTION = (800, 480)
DATASET_PATH = "cleaned_dataset"
LOG_PATH = "log_presensi"
EMBEDDINGS_PATH = "embeddings.pkl"
FACE_MATCH_THRESHOLD = 0.80
COOLDOWN = 1
SKIP_FRAMES = 1
DETECTION_PERSISTENCE = 2
PROCESSING_SIZE = (480, 480)
VERIFICATION_SIZE = (160, 160)  

# Firebase Configuration
FIREBASE_CREDENTIALS = "serviceAccountKey.json"  # Path to your Firebase credentials file
FIREBASE_STORAGE_BUCKET = "your-project-id.appspot.com"  # Replace with your Firebase Storage bucket
FIREBASE_DATABASE_URL = "https://your-project-id.firebaseio.com"  # Replace with your Firebase DB URL

# Create necessary directories
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

class FaceRecognitionSystem:
    def __init__(self):
        self.dataset = {}
        self.name_mapping = {}
        self.embeddings = {}  # Store precomputed face embeddings
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
        
        # Initialize Firebase
        self.init_firebase()
        
        # Load dataset and compute/load embeddings
        self.load_dataset_and_attendance()
        
    def init_firebase(self):
        """Initialize Firebase Admin SDK"""
        try:
            # Check if already initialized
            firebase_admin.get_app()
        except ValueError:
            # Initialize Firebase app
            try:
                cred = credentials.Certificate(FIREBASE_CREDENTIALS)
                firebase_admin.initialize_app(cred, {
                    'storageBucket': FIREBASE_STORAGE_BUCKET,
                    'databaseURL': FIREBASE_DATABASE_URL
                })
                print("Firebase initialized successfully")
            except Exception as e:
                print(f"Error initializing Firebase: {e}")
                print("Continuing with local storage only...")
                
        # Get references to Firebase services
        try:
            self.bucket = storage.bucket()
            self.db_ref = db.reference('attendance')
            print("Firebase storage and database references created")
        except Exception as e:
            print(f"Error setting up Firebase references: {e}")
            self.bucket = None
            self.db_ref = None
        
    def load_dataset_and_attendance(self):
        # Check attendance for today
        today = datetime.now().strftime("%Y-%m-%d")
        today_log_dir = os.path.join(LOG_PATH, today)
        
        # Check local attendance records
        if os.path.exists(today_log_dir):
            for filename in os.listdir(today_log_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    name = filename.split('_')[0].lower()
                    self.attendance_today.add(name)
        
        # Check Firebase attendance records
        if self.db_ref:
            try:
                firebase_attendance = self.db_ref.child(today).get()
                if firebase_attendance:
                    for person_id, data in firebase_attendance.items():
                        if isinstance(data, dict) and 'name' in data:
                            self.attendance_today.add(data['name'].lower())
            except Exception as e:
                print(f"Error getting Firebase attendance: {e}")
        
        # Load dataset
        for filename in os.listdir(DATASET_PATH):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                base_name = "_".join(filename.split("_")[:-1])
                name_key = base_name.lower()
                
                if name_key not in self.dataset:
                    self.dataset[name_key] = []
                    self.name_mapping[name_key] = base_name
                
                self.dataset[name_key].append(os.path.join(DATASET_PATH, filename))
        
        # Load or compute embeddings
        self.load_or_compute_embeddings()
                
        print(f"Loaded {len(self.dataset)} identities from dataset")
        print(f"Already attended today: {len(self.attendance_today)} people")
    
    def load_or_compute_embeddings(self):
        """Load precomputed embeddings or compute them if not available"""
        if os.path.exists(EMBEDDINGS_PATH):
            try:
                print("Loading precomputed embeddings...")
                with open(EMBEDDINGS_PATH, 'rb') as f:
                    self.embeddings = pickle.load(f)
                print(f"Loaded {len(self.embeddings)} precomputed embeddings")
                return
            except Exception as e:
                print(f"Error loading embeddings: {e}")
        
        print("Computing embeddings for dataset (this may take a while)...")
        # Create embeddings for all dataset images
        for name_key, images in self.dataset.items():
            self.embeddings[name_key] = []
            for img_path in images:
                try:
                    print(f"Computing embedding for {img_path}")
                    # Get embedding using DeepFace
                    embedding_objs = DeepFace.represent(
                        img_path=img_path,
                        model_name='Facenet',
                        enforce_detection=False,
                        detector_backend='opencv',
                        align=True,
                        normalization='Facenet'
                    )
                    if embedding_objs and len(embedding_objs) > 0:
                        self.embeddings[name_key].append(embedding_objs[0]['embedding'])
                except Exception as e:
                    print(f"Error computing embedding for {img_path}: {e}")
        
        # Save embeddings to file
        try:
            with open(EMBEDDINGS_PATH, 'wb') as f:
                pickle.dump(self.embeddings, f)
            print("Embeddings saved successfully")
        except Exception as e:
            print(f"Error saving embeddings: {e}")
        
    def update_attendance(self, name_key, image_path=None, clean_frame=None):
        """Update attendance records locally and in Firebase"""
        self.attendance_today.add(name_key)
        
        # Get current timestamp and formatted date
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        
        # Prepare attendance data
        attendance_data = {
            'name': self.name_mapping[name_key],
            'timestamp': timestamp,
            'date': today
        }
        
        # Upload image to Firebase Storage if available
        image_url = None
        if self.bucket and clean_frame is not None:
            try:
                # Create a unique ID for the image
                image_id = str(uuid.uuid4())
                image_path = f"attendance/{today}/{self.name_mapping[name_key]}_{image_id}.jpg"
                
                # Save image to temporary file
                temp_image_path = f"temp_{image_id}.jpg"
                cv2.imwrite(temp_image_path, clean_frame)
                
                # Upload to Firebase Storage
                blob = self.bucket.blob(image_path)
                blob.upload_from_filename(temp_image_path)
                blob.make_public()
                image_url = blob.public_url
                
                # Remove temporary file
                os.remove(temp_image_path)
                
                # Add image URL to attendance data
                attendance_data['image_url'] = image_url
                
                print(f"Image uploaded to Firebase: {image_url}")
            except Exception as e:
                print(f"Error uploading image to Firebase: {e}")
        
        # Update Firebase Realtime Database
        if self.db_ref:
            try:
                # Generate a unique key for this attendance record
                attendance_id = str(uuid.uuid4())
                
                # Structure: attendance/YYYY-MM-DD/[attendance_id]/data
                self.db_ref.child(today).child(attendance_id).set(attendance_data)
                print(f"Attendance data uploaded to Firebase for {self.name_mapping[name_key]}")
            except Exception as e:
                print(f"Error updating Firebase database: {e}")
                # Fallback to local logging
                print(f"Logging attendance locally instead")
    
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
                
                self.frame_queue.put(frame.copy())
                time.sleep(0.001)
        finally:
            camera.release()
    
    def face_detection_thread(self):
        """Dedicated thread just for face detection to maintain smooth tracking"""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        while self.running:
            try:
                # Get the latest frame
                frame = self.frame_queue.get(timeout=1)
                self.frame_count += 1
                
                # Keep a clean copy for logging purposes
                clean_frame = frame.copy()
                
                # Process the frame for face detection
                detections = []
                try:
                    # Resize for faster detection (but keep original for display)
                    small_frame = cv2.resize(frame, PROCESSING_SIZE)
                    
                    # Detect faces using OpenCV's Haar Cascade
                    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                    
                    # Scale back to original frame size
                    scale_x = frame.shape[1] / PROCESSING_SIZE[0]
                    scale_y = frame.shape[0] / PROCESSING_SIZE[1]
                    
                    for (x_small, y_small, w_small, h_small) in faces:
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
                        if self.frame_count % SKIP_FRAMES == 0 and current_time - self.last_detection_time > COOLDOWN:
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
            
            time.sleep(0.001)  # Small sleep to prevent CPU hogging
    
    def get_face_embedding(self, face_img):
        """Extract facial embedding from image using FaceNet"""
        try:
            # Resize to optimal size for FaceNet
            face_img = cv2.resize(face_img, VERIFICATION_SIZE)
            
            # Get embedding directly
            embedding_objs = DeepFace.represent(
                img_path=face_img,
                model_name='Facenet',
                enforce_detection=False,
                detector_backend='skip',  # Skip detection since we already have the face
                align=True,
                normalization='Facenet'
            )
            
            if embedding_objs and len(embedding_objs) > 0:
                return embedding_objs[0]['embedding']
            return None
        except Exception as e:
            print(f"Error getting face embedding: {e}")
            return None
    
    def compare_embeddings(self, embedding1, embedding2):
        """Compare two facial embeddings and return similarity score"""
        if embedding1 is None or embedding2 is None:
            return 0
        
        # Convert to numpy arrays
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        cosine = dot / (norm1 * norm2)
        
        # Convert to distance (0 is identical, 2 is completely different)
        distance = 1 - cosine
        
        # Convert to accuracy percentage (100% is identical)
        accuracy = (1 - distance) * 100
        return accuracy
    
    def recognition_thread(self):
        """Dedicated thread for face recognition to avoid impacting tracking FPS"""
        while self.running:
            try:
                face_img, face_info, clean_frame = self.recognition_queue.get(timeout=1)
                
                # If we have precomputed embeddings, use them
                if self.embeddings:
                    # Extract embedding for the detected face
                    face_embedding = self.get_face_embedding(face_img)
                    
                    if face_embedding is None:
                        continue
                    
                    highest_accuracy = 0
                    best_match_name = None
                    is_already_attended = False
                    
                    # Compare with precomputed embeddings
                    for name_key, stored_embeddings in self.embeddings.items():
                        # Skip if no valid embeddings for this person
                        if not stored_embeddings:
                            continue
                        
                        # Compare with each stored embedding for this person
                        for stored_embedding in stored_embeddings:
                            current_accuracy = self.compare_embeddings(face_embedding, stored_embedding)
                            
                            if current_accuracy > highest_accuracy:
                                highest_accuracy = current_accuracy
                                best_match_name = name_key
                                # Check if this matched person already attended
                                is_already_attended = (name_key in self.attendance_today)
                else:
                    # Fallback to direct image comparison if no embeddings
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
                                # Check if this matched person already attended
                                is_already_attended = (name_key in self.attendance_today)
                                
                        except Exception as e:
                            continue
                
                # If we found a match with good confidence
                if highest_accuracy >= 80 and best_match_name:
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
                        # Log attendance locally first
                        today = datetime.now().strftime("%Y-%m-%d")
                        log_dir = os.path.join(LOG_PATH, today)
                        os.makedirs(log_dir, exist_ok=True)
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        log_filename = f"{self.name_mapping[best_match_name]}_{timestamp}.jpg"
                        log_filepath = os.path.join(log_dir, log_filename)
                        
                        # Save the clean attendance log locally
                        cv2.imwrite(log_filepath, clean_frame)
                        
                        # Update attendance records (both local and Firebase)
                        self.update_attendance(best_match_name, log_filepath, clean_frame)
                        
                        print(f"Attendance: {self.name_mapping[best_match_name]} | Accuracy: {highest_accuracy:.1f}%")
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