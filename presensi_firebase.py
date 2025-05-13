import os
import base64
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from PIL import Image
from io import BytesIO

cred = credentials.Certificate('simkab-v00-firebase-adminsdk-msk6w-785957539f.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

today = datetime.now().strftime("%Y-%m-%d")
log_dir = os.path.join("log_presensi", today)

if not os.path.exists(log_dir):
    print(f"Tidak ada folder log untuk hari ini: {log_dir}")
    exit()

for filename in os.listdir(log_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        file_path = os.path.join(log_dir, filename)

        try:
            with Image.open(file_path) as img:
                buffer = BytesIO()
                img.convert("RGB").save(buffer, format="JPEG", quality=70)
                encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Gagal mengoptimasi gambar {filename}: {e}")
            continue

        split_name = filename.split('_')
        if len(split_name) < 3:
            print(f"Format nama file tidak sesuai: {filename}")
            continue

        username = split_name[0].lower()
        tanggal_presensi = split_name[1]
        waktu_presensi = split_name[2].split('.')[0]

        try:
            tanggal_obj = datetime.strptime(tanggal_presensi, "%d-%m-%Y")
            tanggal_formatted = tanggal_obj.strftime("%Y-%m-%d")
        except ValueError:
            print(f"Format tanggal tidak valid di file: {filename}")
            continue

        try:
            waktu_formatted = f"{waktu_presensi[:2]}:{waktu_presensi[2:4]}:{waktu_presensi[4:]}"
        except:
            print(f"Format waktu tidak valid di file: {filename}")
            continue

        doc_ref = db.collection("dataKaryawan").document(username).collection("Kehadiran").document(tanggal_formatted)

        try:
            doc = doc_ref.get()
            if doc.exists:
                data = doc.to_dict()
                waktu_kerja = data.get('waktuKerja')

                if not waktu_kerja:
                    print(f"Tidak ada data waktuKerja untuk {username} pada {tanggal_formatted}")
                    continue

                waktu_hadir_dt = datetime.strptime(f"{tanggal_formatted} {waktu_formatted}", "%Y-%m-%d %H:%M:%S")
                waktu_kerja_dt = datetime.strptime(f"{tanggal_formatted} {waktu_kerja}", "%Y-%m-%d %H:%M")

                waktu_telat = 0
                if waktu_hadir_dt > waktu_kerja_dt:
                    selisih = waktu_hadir_dt - waktu_kerja_dt
                    waktu_telat = int(selisih.total_seconds() // 60)

                doc_ref.update({
                    'photo': encoded_string,
                    'statusHadir': True,
                    'waktuHadir': waktu_formatted,
                    'waktuTelat': waktu_telat
                })
                print(f"Presensi berhasil: {username} pada {tanggal_formatted} (Telat {waktu_telat} menit)")
            else:
                print(f"Tidak ditemukan kehadiran untuk {username} pada {tanggal_formatted}")
        except Exception as e:
            print(f"Kesalahan saat memperbarui kehadiran untuk {username} pada {tanggal_formatted}: {e}")
