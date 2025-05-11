import os
import re
import cv2
import numpy as np
from mtcnn import MTCNN
from PIL import Image
import time

# Config
INPUT_DIR = "dataset"
OUTPUT_DIR = "cleaned_dataset"
TARGET_SIZE = (250, 250)  # Ukuran wadah akhir
MARGIN_PERCENT = 0.15     # Margin tambahan di sekitar wajah
JPEG_QUALITY = 100        # Kualitas output JPEG

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load MTCNN detector
detector = MTCNN()

def format_filename(old_name):
    name_part = os.path.splitext(old_name)[0]
    match = re.match(r"([a-zA-Z]+)(\d+)", name_part)
    if match:
        return f"{match.group(1).lower()}_{match.group(2)}.jpg"
    return f"{name_part.lower()}.jpg"

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
        print(f"Tidak ditemukan wajah di {file_path}")
        return False

    # Get face with highest confidence
    best_face = max(result, key=lambda x: x['confidence'])
    x, y, w, h = best_face['box']

    # Validate face size
    face_area = w * h
    image_area = image.shape[0] * image.shape[1]
    if face_area / image_area < 0.05:
        print(f"Wajah terlalu kecil di{file_path}")
        return False

    face_img = crop_with_margin(image, (x, y, w, h), MARGIN_PERCENT)
    face_pil = Image.fromarray(face_img).resize(TARGET_SIZE, Image.LANCZOS)
    face_pil.save(output_path, format="JPEG", quality=JPEG_QUALITY)
    return True

def clean_dataset():
    start_time = time.time()
    
    # Get list of images in input directory
    image_files = [f for f in os.listdir(INPUT_DIR)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print("Tidak ditemukan gambar.")
        return
    
    # Get list of already processed images
    existing_files = set(os.listdir(OUTPUT_DIR))
    
    # Counter variables
    success = 0
    skipped_existing = 0
    skipped_failed = 0
    
    for file in image_files:
        output_filename = format_filename(file)
        input_path = os.path.join(INPUT_DIR, file)
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Skip if file already exists in output directory
        if output_filename in existing_files:
            print(f"Skipped (foto sudah ada): {file}")
            skipped_existing += 1
            continue
        
        if process_image(input_path, output_path):
            success += 1
            print(f"Berhasil Diproses: {file}")
        else:
            skipped_failed += 1
            print(f"Skipped (gagal memproses): {file}")
    
    elapsed_time = time.time() - start_time
    print(f"\nSummary:")
    print(f"- Berhasil diproses : {success} gambar")
    print(f"- Skipped (sudah ada): {skipped_existing} gambar")
    print(f"- Skipped (gagal meproses): {skipped_failed} gambar")
    print(f"- Total gambar: {len(image_files)}")
    print(f"- Jumlah waktu: {elapsed_time:.2f} detik")

if __name__ == "__main__":
    clean_dataset()