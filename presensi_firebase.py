import os
import base64
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

def upload_to_firebase():
    # Check if Firebase is already initialized
    if not firebase_admin._apps:
        cred = credentials.Certificate('simkab-v00-firebase-adminsdk-msk6w-785957539f.json')
        firebase_admin.initialize_app(cred)
    
    db = firestore.client()

    today = datetime.now().strftime("%Y-%m-%d")
    log_dir = os.path.join("log_presensi", today)

    if not os.path.exists(log_dir):
        print(f"Tidak ada folder log untuk hari ini: {log_dir}")
        return

    kehadiran_collection = db.collection('KehadiranKaryawan')
    kehadiran_docs = kehadiran_collection.get()
    kehadiran_data = []

    for doc in kehadiran_docs:
        data = doc.to_dict()
        kehadiran_data.append({
            'id': doc.id,
            'namaKaryawan': data.get('namaKaryawan', ''),
            'tanggalKerja': data.get('tanggalKerja', ''),
            'waktuKerja': data.get('waktuKerja', '')
        })

    upload_count = 0
    for filename in os.listdir(log_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(log_dir, filename)
            
            # Check if file has already been uploaded by creating a marker file
            marker_file = file_path + ".uploaded"
            if os.path.exists(marker_file):
                continue
                
            with open(file_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

            split_name = filename.split('_')
            nama_karyawan = split_name[0]
            tanggal_waktu = "_".join(split_name[1:]).replace(".jpg", "").replace(".jpeg", "").replace(".png", "")
            tanggal_presensi = tanggal_waktu[:8]
            waktu_presensi = tanggal_waktu[9:]

            tanggal_presensi_formatted = f"{tanggal_presensi[:4]}-{tanggal_presensi[4:6]}-{tanggal_presensi[6:]}"
            waktu_presensi_formatted = f"{waktu_presensi[:2]}:{waktu_presensi[2:4]}:{waktu_presensi[4:]}"

            matched_doc = None
            for data in kehadiran_data:
                if data['namaKaryawan'].lower().replace(' ', '') == nama_karyawan.lower() and data['tanggalKerja'] == tanggal_presensi_formatted:
                    matched_doc = data
                    break

            if matched_doc:
                doc_ref = kehadiran_collection.document(matched_doc['id'])
                doc_ref.update({
                    'photo': encoded_string,
                    'statusHadir': True,
                    'waktuHadir': waktu_presensi_formatted,
                    'waktuTelat': 0
                })
                print(f"Upload sukses untuk {nama_karyawan} pada {tanggal_presensi_formatted}")
                
                # Create marker file to indicate upload is complete
                with open(marker_file, 'w') as f:
                    f.write("1")
                    
                upload_count += 1
            else:
                print(f"Tidak ditemukan kecocokan untuk {nama_karyawan} pada {tanggal_presensi_formatted}")
    
    print(f"Uploaded {upload_count} new attendance records to Firebase")

if __name__ == "__main__":
    upload_to_firebase()