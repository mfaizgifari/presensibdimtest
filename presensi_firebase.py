import os
import base64
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timezone, timedelta
from PIL import Image
from io import BytesIO

cred = credentials.Certificate('simkab-v00-firebase-adminsdk-msk6w-32b707e55a.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

jakarta = timezone(timedelta(hours=7))
today = datetime.now(jakarta).strftime("%Y-%m-%d")
log_dir = os.path.join("log_presensi", today)
if not os.path.exists(log_dir):
    exit()

def to_local_naive(dt):
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local = dt.astimezone(jakarta)
    return local.replace(tzinfo=None)

for filename in sorted(os.listdir(log_dir)):
    if not filename.lower().endswith(('.jpg','.jpeg','.png')):
        continue
    filepath = os.path.join(log_dir, filename)
    try:
        with Image.open(filepath) as img:
            buf = BytesIO()
            img.convert("RGB").save(buf, "JPEG", quality=60)
            b64 = base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"[ERROR] {filename} compress/base64 failed: {e}")
        continue

    parts = filename.split('_')
    if len(parts) < 4:
        print(f"[SKIP] invalid filename: {filename}")
        continue

    user, action, tanggal, waktufile = parts[0].lower(), parts[1].lower(), parts[2], parts[3].split('.')[0]
    try:
        dt_doc = datetime.strptime(tanggal, "%d-%m-%Y").strftime("%Y-%m-%d")
        jam = f"{waktufile[:2]}:{waktufile[2:4]}:{waktufile[4:]}"
        presensi_naive = datetime.strptime(f"{dt_doc} {jam}", "%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print(f"[SKIP] parse datetime failed for {filename}: {e}")
        continue

    presensi_aware = presensi_naive.replace(tzinfo=jakarta)
    presensi_utc = presensi_aware.astimezone(timezone.utc)

    ref = db.collection("dataKaryawan").document(user).collection("Kehadiran").document(dt_doc)
    snap = ref.get()
    if not snap.exists:
        print(f"[SKIP] no record for {user} on {dt_doc}")
        continue

    data = snap.to_dict()

    if action == "masuk":
        if not data.get("hadir"):
            mulai_local = to_local_naive(data.get("mulaiKerja")) or presensi_naive
            telat = max(0, int((presensi_naive - mulai_local).total_seconds()//60))
            try:
                ref.update({
                    "hadir": presensi_utc,
                    "statusHadir": True,
                    "waktuTelat": telat,
                    "fotoHadir": f"data:image/jpeg;base64,{b64}"
                })
                print(f"[OK] masuk {user} {dt_doc} {jam}")
            except Exception as e:
                print(f"[FAIL] update hadir {filename}: {e}")
    elif action == "keluar":
        if data.get("hadir") and not data.get("pulang"):
            hadir_local = to_local_naive(data["hadir"])
            selesai_local = to_local_naive(data.get("selesaiKerja")) or presensi_naive
            kerja_min = max(0, int((presensi_naive - hadir_local).total_seconds()//60))
            sched_min = max(0, int((selesai_local - (to_local_naive(data.get("mulaiKerja")) or hadir_local)).total_seconds()//60))
            lembur = max(0, kerja_min - sched_min)
            try:
                ref.update({
                    "pulang": presensi_utc,
                    "totalJamKerja": kerja_min,
                    "lembur": lembur,
                    "fotoPulang": f"data:image/jpeg;base64,{b64}"
                })
                print(f"[OK] keluar {user} {dt_doc} {jam}")
            except Exception as e:
                print(f"[FAIL] update pulang {filename}: {e}")
