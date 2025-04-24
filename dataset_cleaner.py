import cv2
import os
import numpy as np
import re

# Configuration
INPUT_DIR = "dataset"          # Folder containing original images
OUTPUT_DIR = "cleaned_dataset" # Folder for processed faces
TARGET_SIZE = (250, 250)       # Output image size (250x250 pixels)
MARGIN_PERCENT = 0.15          # 15% margin around detected face
JPEG_QUALITY = 90              # Output image quality (0-100)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_face_detector():
    """Load face detection model with error handling"""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Haar cascade file not found at {cascade_path}")
    
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("Failed to load face detector")
    return detector

def detect_face(image, detector):
    """Detect faces with improved parameters"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Improve contrast for detection
    
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,       # Increased from 5 for better accuracy
        minSize=(50, 50),     # Increased minimum face size
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces

def format_filename(old_name):
    """Convert filename from 'faiz1.jpg' to 'faiz_1.jpg' format"""
    # Remove extension
    name_part = os.path.splitext(old_name)[0]
    
    # Split into letters and numbers using regex
    match = re.match(r"([a-zA-Z]+)(\d+)", name_part)
    if match:
        name = match.group(1).lower()  # Convert to lowercase
        number = match.group(2)
        return f"{name}_{number}.jpg"
    return old_name  # Return original if pattern doesn't match

def process_image(input_path, output_path, detector):
    """Process a single image with enhanced quality control"""
    try:
        # Read image with error checking
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error: Cannot read image {input_path}")
            return False

        # Detect faces with improved method
        faces = detect_face(image, detector)
        if len(faces) == 0:
            print(f"No faces detected in {input_path}")
            return False

        # Get primary face with largest area
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        x, y, w, h = faces[0]

        # Calculate margin (15% of face dimension)
        margin = int(max(w, h) * MARGIN_PERCENT)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image.shape[1], x + w + margin)
        y2 = min(image.shape[0], y + h + margin)

        # Crop and resize with quality interpolation
        face = image[y1:y2, x1:x2]
        face = cv2.resize(face, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)

        # Save with quality control
        cv2.imwrite(output_path, face, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return True

    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def clean_dataset():
    """Process all images in the dataset"""
    detector = load_face_detector()
    
    # Get supported image files
    image_files = [f for f in os.listdir(INPUT_DIR) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"No images found in {INPUT_DIR}")
        return

    # Process each image
    success_count = 0
    for filename in image_files:
        # Format the new filename
        new_filename = format_filename(filename)
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, new_filename)
        
        if process_image(input_path, output_path, detector):
            success_count += 1
            print(f"Processed: {filename} -> {new_filename}")

    print(f"Completed: Processed {success_count}/{len(image_files)} images successfully")

if __name__ == "__main__":
    clean_dataset()