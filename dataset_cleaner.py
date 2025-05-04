import os
import re
import cv2
import numpy as np
from mtcnn import MTCNN
from PIL import Image

# Config
INPUT_DIR = "dataset"
OUTPUT_DIR = "cleaned_dataset"
TARGET_SIZE = (250, 250)
MARGIN_PERCENT = 0.15
JPEG_QUALITY = 100

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load MTCNN detector
detector = MTCNN()

def format_filename(old_name):
    name_part = os.path.splitext(old_name)[0]
    match = re.match(r"([a-zA-Z]+)(\d+)", name_part)
    if match:
        return f"{match.group(1).lower()}_{match.group(2)}.jpg"
    return old_name

def crop_with_margin(image, box, margin_percent):
    x, y, w, h = box
    margin = int(max(w, h) * margin_percent)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(image.shape[1], x + w + margin)
    y2 = min(image.shape[0], y + h + margin)
    return image[y1:y2, x1:x2]

def process_image(file_path, output_path):
    image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(image)

    if not result:
        print(f"No face detected in {file_path}")
        return False

    # Ambil wajah dengan confidence tertinggi
    best_face = max(result, key=lambda x: x['confidence'])
    x, y, w, h = best_face['box']

    # Validasi ukuran wajah
    face_area = w * h
    image_area = image.shape[0] * image.shape[1]
    if face_area / image_area < 0.05:
        print(f"Face too small in {file_path}")
        return False

    face_img = crop_with_margin(image, (x, y, w, h), MARGIN_PERCENT)
    face_pil = Image.fromarray(face_img).resize(TARGET_SIZE, Image.LANCZOS)
    face_pil.save(output_path, format="JPEG", quality=JPEG_QUALITY)
    return True

def clean_dataset():
    image_files = [f for f in os.listdir(INPUT_DIR)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print("No images found.")
        return

    success = 0
    for file in image_files:
        input_path = os.path.join(INPUT_DIR, file)
        output_path = os.path.join(OUTPUT_DIR, format_filename(file))
        if process_image(input_path, output_path):
            success += 1
            print(f"[✓] Processed: {file}")
        else:
            print(f"[x] Skipped: {file}")
    print(f"\nDone: {success}/{len(image_files)} images processed.")

if __name__ == "__main__":
    clean_dataset()
