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
EMBEDDINGS_PATH = "embeddings.pkl"  # New: Store precomputed embeddings
MIN_ACCURACY = 0.8  # 80% confidence for face detection
FACE_MATCH_THRESHOLD = 0.55 
COOLDOWN = 1  # Increased cooldown to reduce recognition attempts
SKIP_FRAMES = 3  # Process every nth frame for recognition (increased)
DETECTION_PERSISTENCE = 1  # Number of frames to keep showing detection when not processing
PROCESSING_SIZE = (320, 240)  # Smaller size for processing to improve performance
VERIFICATION_SIZE = (160, 160)  # Standard size for FaceNet inputs
FONT = cv2.FONT_HERSHEY_COMPLEX  # Changed from SIMPLEX to COMPLEX for Arial-like font

# Create necessary directories
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

# Function to create and show a success notification window
def show_success_notification(name):
    # Create a new top-level window
    notification_window = Toplevel()
    notification_window.title("Presensi Berhasil")
    
    # Set window properties
    notification_window.overrideredirect(True)  # Remove window decorations
    notification_window.attributes('-topmost', True)  # Keep on top
    notification_window.configure(bg="white")
    
    # Calculate position (center of screen)
    screen_width = notification_window.winfo_screenwidth()
    screen_height = notification_window.winfo_screenheight()
    window_width = 400
    window_height = 300
    x_position = (screen_width - window_width) // 2
    y_position = (screen_height - window_height) // 2
    
    notification_window.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
    
    # Add a frame with white background and rounded corners effect
    main_frame = tk.Frame(notification_window, bg="white", padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Add checkmark symbol (using Unicode character)
    checkmark_label = tk.Label(
        main_frame, 
        text="✓", 
        font=("Arial", 60), 
        fg="#4CAF50",  # Green color
        bg="white"
    )
    checkmark_label.pack(pady=(20, 10))
    
    # Add "Presensi Berhasil" header
    header_label = tk.Label(
        main_frame, 
        text="Presensi Berhasil", 
        font=("Arial", 18, "bold"), 
        fg="#333333",
        bg="white"
    )
    header_label.pack(pady=(0, 10))
    
    # Add name subheader
    subheader_label = tk.Label(
        main_frame, 
        text=f"{name} berhasil melakukan presensi!", 
        font=("Arial", 14), 
        fg="#666666",
        bg="white"
    )
    subheader_label.pack(pady=(0, 20))
    
    # Auto-close after 5 seconds
    notification_window.after(5000, notification_window.destroy)
    
    return notification_window

class FaceRecognitionSystem:
    def __init__(self):
        # Initialize Tkinter root but keep it hidden
        self.tk_root = tk.Tk()
        self.tk_root.withdraw()  # Hide the main root window
        
        self.dataset = {}
        self.name_mapping = {}
        self.embeddings = {}  # New: Store precomputed face embeddings
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
        self.attendance_updated = False  # Flag to track if attendance was updated
        
        # Load dataset and compute/load embeddings
        self.load_dataset_and_attendance()
        
    def load_dataset_and_attendance(self):
        # Check attendance for today
        today = datetime.now().strftime("%d-%m-%Y")  # Changed date format to day-month-year
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
        
    def update_attendance(self, name_key):
        self.attendance_today.add(name_key)
        self.attendance_updated = True
    
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
                
                # If we found a match with good confidence
                if highest_accuracy >= 80 and best_match_name:
                    x, y, w, h = face_info['box']
                    display_name = self.name_mapping[best_match_name]
                    
                    # Update the face info in last_detections
                    for face in self.last_detections:
                        face_x, face_y, face_w, face_h = face['box']
                        # If this is the same face (close enough position)
                        if abs(face_x - x) < 30 and abs(face_y - y) < 30:
                            face['name'] = display_name
                            face['accuracy'] = highest_accuracy
                            face['already_attended'] = is_already_attended
                    
                    # Only log attendance if person hasn't attended yet
                    if not is_already_attended:
                        # Log attendance with clean frame (no overlays)
                        today = datetime.now().strftime("%d-%m-%Y")  
                        log_dir = os.path.join(LOG_PATH, today)
                        os.makedirs(log_dir, exist_ok=True)
                        
                        timestamp = datetime.now().strftime("%d-%m-%Y_%H%M%S")  # Changed timestamp format to day-month-year
                        log_filename = f"{display_name}_{timestamp}.jpg"
                        
                        # Save the clean attendance log
                        cv2.imwrite(os.path.join(log_dir, log_filename), clean_frame)
                        
                        print(f"Attendance: {display_name} | Accuracy: {highest_accuracy:.1f}%")
                        self.update_attendance(best_match_name)
                        
                        # Show success notification using the Tkinter main thread
                        self.tk_root.after(0, lambda n=display_name: show_success_notification(n))
                        
                    else:
                        print(f"Already attended: {display_name} | Accuracy: {highest_accuracy:.1f}%")
                
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
                    color = (0, 0, 255) if face['already_attended'] else (0, 255, 0)  # Red if attended, green if not
                    
                    # Draw rectangle around face
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Create label with name and accuracy
                    label = f"{face['name']}"
                    if face['accuracy'] > 0:
                        label += f" ({face['accuracy']:.1f}%)"
                    
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
        """Thread to check for attendance updates and trigger Firebase upload"""
        last_upload_time = 0
        while self.running:
            current_time = time.time()
            
            # Check if attendance was updated and if enough time has passed since last upload (5 seconds)
            if self.attendance_updated and current_time - last_upload_time > 5:
                print("Attendance updated, uploading to Firebase...")
                try:
                    # Run the Firebase upload script
                    subprocess.run(['python', 'presensi_firebase.py'])
                    print("Firebase upload completed!")
                    self.attendance_updated = False
                    last_upload_time = current_time
                except Exception as e:
                    print(f"Error running Firebase upload: {e}")
            
            time.sleep(1)  # Check every second
    
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
        
        # Run Tkinter main loop in the background with minimal update frequency
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