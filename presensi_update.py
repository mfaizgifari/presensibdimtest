import os
import firebase_admin
from firebase_admin import credentials, firestore, storage

cred = credentials.Certificate('simkab-v00-firebase-adminsdk-msk6w-785957539f.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'simkab-v00.firebasestorage.app'
})

db = firestore.client()
bucket = storage.bucket()

dataset_dir = 'dataset'
os.makedirs(dataset_dir, exist_ok=True)

existing_files = os.listdir(dataset_dir)
existing_names = set("_".join(f.split("_")[:-1]).lower() for f in existing_files if f.lower().endswith(('.png', '.jpg', '.jpeg')))

karyawan_collection = db.collection('dataKaryawan')
karyawan_docs = karyawan_collection.get()

for doc in karyawan_docs:
    data = doc.to_dict()
    nama_karyawan = data.get('namaKaryawan', None)
    if not nama_karyawan:
        continue

    nama_key = nama_karyawan.replace(" ", "").lower()
    if any(nama_key in existing for existing in existing_names):
        continue

    folder_path = f"Karyawan/{nama_karyawan}/"
    blobs = list(bucket.list_blobs(prefix=folder_path))
    if not blobs:
        continue

    for i, blob in enumerate(blobs, start=1):
        filename = f"{nama_karyawan.replace(' ', '')}_{str(i).zfill(2)}.jpg"
        file_path = os.path.join(dataset_dir, filename)
        blob.download_to_filename(file_path)

print("Unduhan dataset selesai.")