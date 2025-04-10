import os
import face_recognition

DATA_DIR = "dataset"

def load_dataset():
    known_encodings = []
    known_names = []

    for file in os.listdir(DATA_DIR):
        img_path = os.path.join(DATA_DIR, file)
        name = os.path.splitext(file)[0]
        img = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(img)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(name)
    
    print("Dataset loaded successfully.")
    return known_encodings, known_names
