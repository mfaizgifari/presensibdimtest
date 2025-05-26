import os
import cv2
import threading
import queue
import time
import numpy as np
from deepface import DeepFace
from datetime import datetime
import pickle
import subprocess
import tkinter as tk
from tkinter import Toplevel
from monitor_utils import start_monitoring
from animasi_window import show_checkin_notification, show_checkout_notification

start_monitoring()

# First run the prerequisite scripts
print("Step 1: Updating dataset from Firebase...")
try:
    # Run the presensi_update.py script to download the latest data from Firebase
    subprocess.run(['python', 'presensi_update.py'], check=True)
    print("Dataset update completed.")
except Exception as e:
    print(f"Error updating dataset: {e}")

print("Step 2: Cleaning the dataset...")
try:
    # Run the dataset_cleaner.py script to process the downloaded images
    subprocess.run(['python', 'dataset_cleaner.py'], check=True)
    print("Dataset cleaning completed.")
except Exception as e:
    print(f"Error cleaning dataset: {e}")

print("Step 3: Starting face recognition system...")

# Configuration
RESOLUTION = (800, 480) 
DATASET_PATH = "cleaned_dataset"
LOG_PATH = "log_presensi"
EMBEDDINGS_PATH = "embeddings.pkl"  # Store precomputed embeddings
EMBEDDINGS_META_PATH = "embeddings_meta.pkl"  # New: Store metadata about embeddings
MIN_ACCURACY = 0.8  # 80% confidence for face detection
FACE_MATCH_THRESHOLD = 0.55 
COOLDOWN = 1  # Increased cooldown to reduce recognition attempts
SKIP_FRAMES = 3  # Process every nth frame for recognition (increased)
DETECTION_PERSISTENCE = 1  
PROCESSING_SIZE = (320, 240)  # Smaller size for processing to improve performance
VERIFICATION_SIZE = (160, 160)  # Standard size for FaceNet inputs
FONT = cv2.FONT_HERSHEY_COMPLEX  
RECOGNITION_COOLDOWN =  3600

# Create necessary directories
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

class FaceRecognitionSystem:
    def __init__(self):
        # Initialize Tkinter root but keep it hidden
        self.tk_root = tk.Tk()
        self.tk_root.withdraw()  # Hide the main root window
        
        self.dataset = {}
        self.name_mapping = {}
        self.embeddings = {}  # Store precomputed face embeddings
        self.embeddings_meta = {
            'dataset_files': [],  # List of files in dataset when embeddings were last computed
            'timestamp': 0        # When embeddings were last computed
        }
        self.attendance_today = {}  # Changed to dict to track check-in/check-out status
        self.last_recognition_time = {}  # Track last recognition time for each person
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
        self.attendance_updated = False  # Flag to track if attendance was updated
        
        # Load dataset and compute/load embeddings
        self.load_dataset_and_attendance()
        
    def load_dataset_and_attendance(self):
        # Check attendance for today
        today = datetime.now().strftime("%Y-%m-%d")
        today_log_dir = os.path.join(LOG_PATH, today)
        
        if os.path.exists(today_log_dir):
            for filename in os.listdir(today_log_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Extract name from filename (everything before the first underscore)
                    name = filename.split('_')[0].lower()
                    
                    # Determine if it's check-in or check-out based on filename
                    if "_masuk_" in filename.lower():
                        if name not in self.attendance_today:
                            self.attendance_today[name] = {'checkin': True, 'checkout': False}
                        else:
                            self.attendance_today[name]['checkin'] = True
                    elif "_keluar_" in filename.lower():
                        if name not in self.attendance_today:
                            self.attendance_today[name] = {'checkin': False, 'checkout': True}
                        else:
                            self.attendance_today[name]['checkout'] = True
        
        # Load dataset
        dataset_files = []
        for filename in os.listdir(DATASET_PATH):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                dataset_files.append(filename)
                
                # Use the FULL name before the last underscore (not just partial name)
                parts = filename.split("_")
                if len(parts) > 1:
                    # Get everything before the number part (last part)
                    name_parts = parts[:-1]
                    base_name = "_".join(name_parts)
                    name_key = base_name.lower()
                    
                    if name_key not in self.dataset:
                        self.dataset[name_key] = []
                        self.name_mapping[name_key] = base_name
                    
                    self.dataset[name_key].append(os.path.join(DATASET_PATH, filename))
        
        # Load or compute embeddings with consistency check
        self.check_and_update_embeddings(dataset_files)
                
        print(f"Loaded {len(self.dataset)} identities from dataset")
        print(f"Attendance status today: {len(self.attendance_today)} people")
    
    def check_and_update_embeddings(self, current_files):
        """Check if embeddings are up-to-date and update if needed"""
        embeddings_need_update = True
        
        # Try to load existing embeddings metadata
        if os.path.exists(EMBEDDINGS_META_PATH):
            try:
                with open(EMBEDDINGS_META_PATH, 'rb') as f:
                    self.embeddings_meta = pickle.load(f)
                
                # Compare dataset files when embeddings were computed with current files
                old_files_set = set(self.embeddings_meta['dataset_files'])
                current_files_set = set(current_files)
                
                # If dataset hasn't changed, we can use existing embeddings
                if old_files_set == current_files_set and os.path.exists(EMBEDDINGS_PATH):
                    try:
                        print("Loading existing embeddings (dataset unchanged)...")
                        with open(EMBEDDINGS_PATH, 'rb') as f:
                            self.embeddings = pickle.load(f)
                        print(f"Loaded {len(self.embeddings)} existing embeddings")
                        embeddings_need_update = False
                    except Exception as e:
                        print(f"Error loading embeddings: {e}")
                        embeddings_need_update = True
                else:
                    print("Dataset has changed since last run. Rebuilding embeddings...")
            except Exception as e:
                print(f"Error loading embeddings metadata: {e}")
        
        # Compute embeddings if needed
        if embeddings_need_update:
            self.compute_embeddings(current_files)
    
    def compute_embeddings(self, current_files):
        """Compute embeddings for all dataset images"""
        print("Computing embeddings for dataset...")
        self.embeddings = {}
        
        # Create embeddings for all dataset images
        for name_key, images in self.dataset.items():
            print(f"Computing embeddings for person: {name_key}")
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
            # Save embeddings
            with open(EMBEDDINGS_PATH, 'wb') as f:
                pickle.dump(self.embeddings, f)
            
            # Save metadata about when embeddings were computed
            self.embeddings_meta = {
                'dataset_files': current_files,
                'timestamp': time.time()
            }
            with open(EMBEDDINGS_META_PATH, 'wb') as f:
                pickle.dump(self.embeddings_meta, f)
                
            print("Embeddings and metadata saved successfully")
        except Exception as e:
            print(f"Error saving embeddings: {e}")
        
    def update_attendance(self, name_key, is_checkout=False):
        if name_key not in self.attendance_today:
            self.attendance_today[name_key] = {'checkin': False, 'checkout': False}
        
        if is_checkout:
            self.attendance_today[name_key]['checkout'] = True
        else:
            self.attendance_today[name_key]['checkin'] = True
            
        self.attendance_updated = True
    
    def get_attendance_status(self, name_key):
        """Get attendance status for a person"""
        if name_key not in self.attendance_today:
            return "none"  # No attendance yet
        
        status = self.attendance_today[name_key]
        if status['checkin'] and status['checkout']:
            return "both"  # Both check-in and check-out done
        elif status['checkin']:
            return "checkin"  # Only check-in done
        elif status['checkout']:
            return "checkout"  # Only check-out done (unusual case)
        else:
            return "none"  # No attendance yet
    
    def can_perform_action(self, name_key):
        """Determine what action can be performed for a person"""
        current_time = time.time()
        
        # Check cooldown period
        if name_key in self.last_recognition_time:
            if current_time - self.last_recognition_time[name_key] < RECOGNITION_COOLDOWN:
                return None  # Still in cooldown
        
        status = self.get_attendance_status(name_key)
        
        if status == "none":
            return "checkin"  # Can do check-in
        elif status == "checkin":
            return "checkout"  # Can do check-out
        else:
            return None  # Already completed both or in unusual state
    
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
        # Load face detector once
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
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
                    
                    # Detect faces on every frame for smoother tracking using OpenCV's faster detector
                    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                    
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
                            'status': "none",
                            'next_action': None
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
                                    detections[i]['status'] = closest_match['status']
                                    detections[i]['next_action'] = closest_match['next_action']
                        
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
                
                # Extract embedding for the detected face
                face_embedding = self.get_face_embedding(face_img)
                
                if face_embedding is None:
                    continue
                
                highest_accuracy = 0
                best_match_name = None
                
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
                
                # If we found a match with good confidence
                if highest_accuracy >= 75 and best_match_name:
                    x, y, w, h = face_info['box']
                    display_name = self.name_mapping[best_match_name]
                    
                    # Get current status and determine next action
                    current_status = self.get_attendance_status(best_match_name)
                    next_action = self.can_perform_action(best_match_name)
                    
                    # Update the face info in last_detections
                    for face in self.last_detections:
                        face_x, face_y, face_w, face_h = face['box']
                        # If this is the same face (close enough position)
                        if abs(face_x - x) < 30 and abs(face_y - y) < 30:
                            face['name'] = display_name
                            face['accuracy'] = highest_accuracy
                            face['status'] = current_status
                            face['next_action'] = next_action
                    
                    # Only log attendance if an action can be performed
                    if next_action:
                        # Update last recognition time
                        self.last_recognition_time[best_match_name] = time.time()
                        
                        # Log attendance with clean frame (no overlays)
                        today = datetime.now().strftime("%Y-%m-%d")
                        log_dir = os.path.join(LOG_PATH, today)
                        os.makedirs(log_dir, exist_ok=True)
                        
                        timestamp = datetime.now().strftime("%d-%m-%Y_%H%M%S")
                        action_type = "masuk" if next_action == "checkin" else "keluar"
                        log_filename = f"{display_name}_{action_type}_{timestamp}.jpg"
                        
                        # Save the clean attendance log
                        cv2.imwrite(os.path.join(log_dir, log_filename), clean_frame)
                        
                        # Update attendance status
                        is_checkout = (next_action == "checkout")
                        self.update_attendance(best_match_name, is_checkout)
                        
                        print(f"Attendance {action_type}: {display_name} | Accuracy: {highest_accuracy:.1f}%")
                        
                        # Show appropriate notification using the Tkinter main thread
                        if next_action == "checkin":
                            self.tk_root.after(0, lambda n=display_name: show_checkin_notification(n, self.tk_root))
                        else:
                            self.tk_root.after(0, lambda n=display_name: show_checkout_notification(n, self.tk_root))
                        
                    else:
                        if current_status == "both":
                            print(f"{display_name} has already completed both check-in and check-out today")
                        elif best_match_name in self.last_recognition_time:
                            remaining_cooldown = RECOGNITION_COOLDOWN - (time.time() - self.last_recognition_time[best_match_name])
                            if remaining_cooldown > 0:
                                print(f"{display_name} in cooldown, wait {remaining_cooldown:.0f} seconds")
                
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
                
                # Draw FPS counter with Arial-like font
                cv2.putText(display_frame, f"FPS: {self.fps}", (10, 30), 
                            FONT, 0.7, (255, 255, 255), 2)
                
                # Draw face detections with name and accuracy
                for face in detections:
                    x, y, w, h = face['box']
                    
                    # Determine color based on status and next action
                    if face['status'] == "both":
                        color = (128, 128, 128)  # Gray for completed
                    elif face['next_action'] == "checkout":
                        color = (0, 165, 255)  # Orange for checkout
                    elif face['next_action'] == "checkin":
                        color = (0, 255, 0)  # Green for checkin
                    else:
                        color = (0, 0, 255)  # Red for no action available
                    
                    # Draw rectangle around face
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Create label with name, accuracy, and status
                    label = f"{face['name']}"
                    if face['accuracy'] > 0:
                        label += f" ({face['accuracy']:.1f}%)"
                    
                    # Add status information
                    if face['status'] == "both":
                        label += " [SELESAI]"
                    elif face['status'] == "checkin":
                        label += " [SUDAH MASUK]"
                    elif face['status'] == "checkout":
                        label += " [SUDAH KELUAR]"
                    
                    # Calculate label position and draw background
                    label_size, base_line = cv2.getTextSize(label, FONT, 0.5, 1)
                    y_label = max(y - 10, label_size[1])
                    
                    # Draw label background
                    cv2.rectangle(
                        display_frame, 
                        (x, y_label - label_size[1]), 
                        (x + label_size[0], y_label + base_line), 
                        (0, 0, 0), 
                        cv2.FILLED
                    )
                    
                    # Draw label text with Arial-like font
                    cv2.putText(
                        display_frame, 
                        label, 
                        (x, y_label), 
                        FONT, 
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
    
    def firebase_upload_thread(self):
        last_upload_time = 0
        upload_delay = 7        
        def wait_for_file_stability(file_paths, max_wait=10):
            """Wait for files to be stable (not being written to)"""
            stable_count = 0
            wait_time = 0
            
            while wait_time < max_wait:
                all_stable = True
                for file_path in file_paths:
                    if os.path.exists(file_path):
                        try:
                            # Try to open file exclusively to check if it's being written
                            with open(file_path, 'r+'):
                                pass
                        except:
                            all_stable = False
                            break
                
                if all_stable:
                    stable_count += 1
                    if stable_count >= 3:  # File stable for 3 seconds
                        return True
                else:
                    stable_count = 0
                
                time.sleep(1)
                wait_time += 1
            return True  # Proceed anyway after max wait
        
        while self.running:
            current_time = time.time()
            if self.attendance_updated and current_time - last_upload_time > upload_delay:
                print("Attendance updated, preparing for Firebase upload...")
                
                # Wait for file operations to complete
                log_files = [
                    'attendance_log.csv',  # Adjust to your actual log file names
                    'attendance_data.json'  # Add other relevant files
                ]
                print("Waiting for file operations to stabilize...")
                wait_for_file_stability([f for f in log_files if os.path.exists(f)])
                
                time.sleep(2)
                print("Starting Firebase upload...")
                max_retries = 3
                retry_count = 0
                
                while retry_count < max_retries:
                    try:
                        result = subprocess.run(
                            ['python', 'presensi_firebase.py'], 
                            capture_output=True, 
                            text=True, 
                            timeout=45,
                            cwd=os.getcwd()  # Ensure correct working directory
                        )
                        
                        if result.returncode == 0:
                            print("Firebase upload completed successfully!")
                            if result.stdout:
                                print(f"Upload details: {result.stdout.strip()}")
                            break
                        else:
                            retry_count += 1
                            print(f"Upload attempt {retry_count} failed: {result.stderr}")
                            if retry_count < max_retries:
                                print(f"Retrying in 3 seconds...")
                                time.sleep(3)
                    
                    except subprocess.TimeoutExpired:
                        retry_count += 1
                        print(f"Upload attempt {retry_count} timed out")
                        if retry_count < max_retries:
                            print("Retrying...")
                            time.sleep(3)
                    except Exception as e:
                        retry_count += 1
                        print(f"Upload attempt {retry_count} error: {e}")
                        if retry_count < max_retries:
                            time.sleep(3)
                
                if retry_count >= max_retries:
                    print("Firebase upload failed after all retry attempts")
                
                self.attendance_updated = False
                last_upload_time = current_time
            
            time.sleep(1)
    
    def run(self):
        
        # Start threads
        capture_thread = threading.Thread(target=self.capture_thread)
        detection_thread = threading.Thread(target=self.face_detection_thread)
        recognition_thread = threading.Thread(target=self.recognition_thread)
        display_thread = threading.Thread(target=self.display_thread)
        firebase_thread = threading.Thread(target=self.firebase_upload_thread)
        
        capture_thread.daemon = True
        detection_thread.daemon = True
        recognition_thread.daemon = True
        display_thread.daemon = True
        firebase_thread.daemon = True
        
        capture_thread.start()
        detection_thread.start()
        recognition_thread.start()
        display_thread.start()
        firebase_thread.start()
        
        def update_tk():
            if self.running:
                self.tk_root.after(100, update_tk)
        
        update_tk()  # Start the update cycle
        
        # Wait for all threads to finish or until Ctrl+C
        try:
            # Use mainloop to handle Tkinter events properly
            self.tk_root.mainloop()
        except KeyboardInterrupt:
            print("Shutting down...")
            self.running = False
        
        # Clean up
        self.running = False
        self.tk_root.quit()
        capture_thread.join()
        detection_thread.join()
        recognition_thread.join()
        display_thread.join()
        firebase_thread.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.run()