import tkinter as tk
from tkinter import Toplevel

def show_checkin_notification(name, root, on_close=None):
    notification_window = Toplevel(root)
    notification_window.title("Presensi Masuk Berhasil")
    notification_window.overrideredirect(True)
    notification_window.attributes('-topmost', True)
    notification_window.configure(bg="white")

    screen_width = notification_window.winfo_screenwidth()
    screen_height = notification_window.winfo_screenheight()
    window_width = 400
    window_height = 300
    x_position = (screen_width - window_width) // 2
    y_position = (screen_height - window_height) // 2
    notification_window.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

    main_frame = tk.Frame(notification_window, bg="white", padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(main_frame, text="✓", font=("Arial", 60), fg="#4CAF50", bg="white").pack(pady=(20, 10))
    tk.Label(main_frame, text="Presensi Masuk Berhasil", font=("Arial", 18, "bold"), fg="#333333", bg="white").pack(pady=(0, 10))
    tk.Label(main_frame, text=f"{name} \n berhasil melakukan presensi masuk!", font=("Arial", 14), fg="#666666", bg="white").pack(pady=(0, 20))

    def close_and_callback():
        notification_window.destroy()
        if on_close:
            on_close()

    notification_window.after(5000, close_and_callback)

def show_checkout_notification(name, root, on_close=None):
    notification_window = Toplevel(root)
    notification_window.title("Presensi Keluar Berhasil")
    notification_window.overrideredirect(True)
    notification_window.attributes('-topmost', True)
    notification_window.configure(bg="white")

    screen_width = notification_window.winfo_screenwidth()
    screen_height = notification_window.winfo_screenheight()
    window_width = 400
    window_height = 300
    x_position = (screen_width - window_width) // 2
    y_position = (screen_height - window_height) // 2
    notification_window.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

    main_frame = tk.Frame(notification_window, bg="white", padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(main_frame, text="✓", font=("Arial", 60), fg="#FF5722", bg="white").pack(pady=(20, 10))
    tk.Label(main_frame, text="Presensi Keluar Berhasil", font=("Arial", 18, "bold"), fg="#333333", bg="white").pack(pady=(0, 10))
    tk.Label(main_frame, text=f"{name} \n berhasil melakukan presensi keluar!", font=("Arial", 14), fg="#666666", bg="white").pack(pady=(0, 20))

    def close_and_callback():
        notification_window.destroy()
        if on_close:
            on_close()

    notification_window.after(5000, close_and_callback)
