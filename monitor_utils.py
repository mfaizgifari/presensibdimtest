# monitor_utils.py
import time
import psutil
import csv
import os
from datetime import datetime

def get_cpu_temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as file:
            temp_raw = file.readline()
            return int(temp_raw) / 1000.0
    except FileNotFoundError:
        return None

def start_monitoring(txt_path="log_sistem.txt", csv_path="log_sistem.csv", interval=60):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Timestamp", "Suhu_CPU_C", "Penggunaan_CPU_%", "Penggunaan_RAM_%"])

    def monitor_loop():
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            temp = get_cpu_temperature()
            cpu = psutil.cpu_percent(interval=1)
            ram = psutil.virtual_memory().percent

            text = (f"{timestamp} - Suhu CPU: {temp:.2f}°C | CPU: {cpu:.1f}% | RAM: {ram:.1f}%\n"
                    if temp is not None else
                    f"{timestamp} - Gagal membaca suhu | CPU: {cpu:.1f}% | RAM: {ram:.1f}%\n")

            csv_row = [timestamp, f"{temp:.2f}" if temp else "N/A", f"{cpu:.1f}", f"{ram:.1f}"]

            with open(txt_path, "a") as tfile:
                tfile.write(text)
            with open(csv_path, "a", newline="") as cfile:
                csv.writer(cfile).writerow(csv_row)

            time.sleep(interval - 1)

    import threading
    thread = threading.Thread(target=monitor_loop, daemon=True)
    thread.start()
