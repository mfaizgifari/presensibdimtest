import os
import re
import cv2
import numpy as np
from mtcnn import MTCNN
from PIL import Image
import time
import concurrent.futures
import gc

# Config
INPUT_DIR = "dataset"
OUTPUT_DIR = "cleaned_dataset"
TARGET_SIZE = (250, 250)  # Ukuran wadah akhir
MARGIN_PERCENT = 0.15  # Margin tambahan di sekitar wajah
JPEG_QUALITY = 95 # Kualitas output JPEG
MAX_WORKERS = 2  # Jumlah worker untuk multiprocessing (sesuaikan dengan Pi 5)

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Inisialisasi MTCNN detector sekali saja untuk thread utama
detector = None

def get_detector():
    """Lazy loading detector untuk penggunaan di multiprocessing"""
    global detector
    if detector is None:
        detector = MTCNN()  # Removed min_face_size parameter
    return detector

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

def resize_before_detection(image, max_dimension=640):
    """Resize gambar yang sangat besar sebelum deteksi untuk menghemat memori dan waktu"""
    h, w = image.shape[:2]
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h)), scale
    return image, 1.0

def adjust_box_for_scale(box, scale):
    """Menyesuaikan koordinat box jika gambar di-resize sebelum deteksi"""
    if scale != 1.0:
        x, y, w, h = box
        return (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
    return box

def process_image(file_info):
    file_path, output_path = file_info
    
    try:
        # Menggunakan detector di thread/process ini
        local_detector = get_detector()
        
        # Baca gambar dengan optimasi
        image = cv2.imread(file_path)
        if image is None:
            print(f"Tidak dapat membaca gambar: {file_path}")
            return False
            
        # Resize gambar besar untuk mempercepat deteksi
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image, scale = resize_before_detection(image_rgb)
        
        # Deteksi wajah pada gambar yang telah dioptimasi
        result = local_detector.detect_faces(resized_image)
        
        # Pembersihan memori
        del resized_image
        
        if not result:
            print(f"Tidak ditemukan wajah di {file_path}")
            return False
        
        # Get face with highest confidence
        best_face = max(result, key=lambda x: x['confidence'])
        box = adjust_box_for_scale(best_face['box'], scale)
        x, y, w, h = box
        
        # Validate face size
        face_area = w * h
        image_area = image.shape[0] * image.shape[1]
        if face_area / image_area < 0.05:
            print(f"Wajah terlalu kecil di {file_path}")
            return False
        
        # Crop dan simpan dengan PIL untuk kualitas lebih baik
        face_img = crop_with_margin(image_rgb, (x, y, w, h), MARGIN_PERCENT)
        face_pil = Image.fromarray(face_img).resize(TARGET_SIZE, Image.LANCZOS)
        face_pil.save(output_path, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        
        # Pembersihan memori
        del face_img
        del face_pil
        del image
        del image_rgb
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"Error memproses {file_path}: {str(e)}")
        return False

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
    
    # Prepare tasks for processing
    tasks = []
    skipped_existing = 0
    
    for file in image_files:
        output_filename = format_filename(file)
        input_path = os.path.join(INPUT_DIR, file)
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Skip if file already exists in output directory
        if output_filename in existing_files:
            print(f"Skipped (foto sudah ada): {file}")
            skipped_existing += 1
            continue
            
        tasks.append((input_path, output_path))
    
    # Process images
    success = 0
    skipped_failed = 0
    
    # Gunakan ThreadPoolExecutor karena deteksi wajah MTCNN IO-bound
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(process_image, tasks))
        
        for idx, result in enumerate(results):
            file = os.path.basename(tasks[idx][0])
            if result:
                success += 1
                print(f"Berhasil Diproses: {file}")
            else:
                skipped_failed += 1
                print(f"Skipped (gagal memproses): {file}")
    
    elapsed_time = time.time() - start_time
    print(f"\nSummary:")
    print(f"- Berhasil diproses : {success} gambar")
    print(f"- Skipped (sudah ada): {skipped_existing} gambar")
    print(f"- Skipped (gagal memproses): {skipped_failed} gambar")
    print(f"- Total gambar: {len(image_files)}")

if __name__ == "__main__":
    clean_dataset()
