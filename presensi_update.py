import os
import firebase_admin
from firebase_admin import credentials, firestore, storage

cred = credentials.Certificate('simkab-v00-firebase-adminsdk-msk6w-785957539f.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'simkab-v00.firebasestorage.app'  # <- pastikan domain ini benar
})

db = firestore.client()
bucket = storage.bucket()

dataset_dir = 'dataset'
os.makedirs(dataset_dir, exist_ok=True)

existing_files = os.listdir(dataset_dir)
existing_usernames = set(f.split("_")[0].lower() for f in existing_files if f.lower().endswith(('.png', '.jpg', '.jpeg')))

karyawan_collection = db.collection('dataKaryawan')
karyawan_docs = karyawan_collection.get()

total_downloaded = 0
for doc in karyawan_docs:
    data = doc.to_dict()
    username = data.get('username', None)
    if not username:
        continue

    username_key = username.lower()
    if username_key in existing_usernames:
        print(f"Data untuk {username} sudah ada, dilewati.")
        continue

    folder_path = f"Karyawan/{username}/"
    blobs = list(bucket.list_blobs(prefix=folder_path))
    blobs = [b for b in blobs if not b.name.endswith("/")]  # Hindari folder kosong

    if not blobs:
        print(f"Tidak ada file untuk {username}.")
        continue

    print(f"\nMengunduh {len(blobs)} file untuk {username}...")
    for i, blob in enumerate(blobs, start=1):
        filename = f"{username}_{str(i).zfill(2)}.jpg"
        file_path = os.path.join(dataset_dir, filename)
        if os.path.exists(file_path):
            print(f"  - {filename} sudah ada, dilewati.")
            continue
        blob.download_to_filename(file_path)
        print(f"  - {filename} berhasil diunduh ({i}/{len(blobs)})")
        total_downloaded += 1

if total_downloaded == 0:
    print("\nTidak ada file yang diunduh.")
else:
    print(f"\nSelesai. Total file yang diunduh: {total_downloaded}")
