import firebase_admin
from firebase_admin import credentials, firestore
import datetime

# Inisialisasi Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

def mark_attendance(employee_id):
    """Tandai presensi di Firestore."""
    attendance_ref = db.collection("attendance").document(employee_id)
    attendance_data = {
        "timestamp": datetime.datetime.now(),
        "status": "present"
    }
    attendance_ref.set(attendance_data, merge=True)

def is_employee_present(employee_id):
    """Cek apakah karyawan sudah presensi."""
    attendance_ref = db.collection("attendance").document(employee_id)
    doc = attendance_ref.get()
    return doc.exists and doc.to_dict().get("status") == "present"