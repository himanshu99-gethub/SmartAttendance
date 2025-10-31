import cv2
import threading
from datetime import datetime
import numpy as np
from tkinter import *
from tkinter import filedialog, messagebox, ttk
from face_utils import findEncodings, load_student_images
from database import markAttendanceMySQL, fetch_attendance_by_date, delete_attendance_record

class SmartAttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Attendance System ")
        self.root.geometry("950x650")
        self.root.configure(bg="#f0f0f0")

        self.images = []
        self.studentNames = []
        self.encodeListKnown = []
        self.path = ""
        self.running = False
        self.camera_thread = None  # ✅ store camera thread
        self.stop_event = threading.Event()  # ✅ safer thread stop signal

        self.create_widgets()

    # --------------------- GUI Setup ---------------------
    def create_widgets(self):
        title = Label(self.root, text="Smart Attendance System ",
                      font=("Arial", 22, "bold"), bg="#283593", fg="white", pady=10)
        title.pack(fill=X)

        frame = Frame(self.root, bg="#f0f0f0", pady=20)
        frame.pack()

        Button(frame, text="📂 Select Image Folder", command=self.load_images,
               font=("Arial", 12), bg="#5c6bc0", fg="white", width=25).grid(row=0, column=0, padx=10, pady=10)

        Button(frame, text="🧠 Encode Faces", command=self.encode_faces,
               font=("Arial", 12), bg="#3949ab", fg="white", width=25).grid(row=0, column=1, padx=10, pady=10)

        Button(frame, text="🎥 Start Attendance", command=self.start_attendance,
               font=("Arial", 12), bg="#1e88e5", fg="white", width=25).grid(row=1, column=0, padx=10, pady=10)

        Button(frame, text="🛑 Stop Attendance", command=self.stop_attendance,
               font=("Arial", 12), bg="#c62828", fg="white", width=25).grid(row=1, column=1, padx=10, pady=10)

        Button(frame, text="📅 View Today’s Attendance", command=self.show_today_attendance,
               font=("Arial", 12), bg="#43a047", fg="white", width=25).grid(row=2, column=0, padx=10, pady=10)

        Button(frame, text="📆 View By Date", command=self.show_by_date,
               font=("Arial", 12), bg="#00897b", fg="white", width=25).grid(row=2, column=1, padx=10, pady=10)

        Button(frame, text="🗑️ Delete Selected Attendance", command=self.delete_selected_attendance,
               font=("Arial", 12), bg="#f57c00", fg="white", width=25).grid(row=3, column=0, padx=10, pady=10)

        Button(frame, text="❌ Exit", command=self.root.quit,
               font=("Arial", 12), bg="#e53935", fg="white", width=25).grid(row=3, column=1, padx=10, pady=10)

        self.status_label = Label(self.root, text="Status: Ready", font=("Arial", 12), bg="#f0f0f0", fg="black")
        self.status_label.pack(pady=5)

        self.tree = ttk.Treeview(self.root, columns=("Name", "Date", "Time"), show='headings', height=12)
        self.tree.heading("Name", text="Name")
        self.tree.heading("Date", text="Date")
        self.tree.heading("Time", text="Time")
        self.tree.pack(fill=BOTH, padx=20, pady=10)

    # --------------------- Load Images ---------------------
    def load_images(self):
        self.path = filedialog.askdirectory(title="Select Folder with Student Images")
        if not self.path:
            return
        self.images, self.studentNames = load_student_images(self.path)
        self.status_label.config(text=f"Loaded {len(self.images)} images.")
        messagebox.showinfo("Images Loaded", f"{len(self.images)} student images loaded successfully!")

    # --------------------- Encode Faces ---------------------
    def encode_faces(self):
        if not self.images:
            messagebox.showwarning("Warning", "Please load images first!")
            return
        self.status_label.config(text="Encoding faces, please wait...")
        self.root.update()
        self.encodeListKnown = findEncodings(self.images)
        self.status_label.config(text="Encoding completed!")
        messagebox.showinfo("Encoding", "Face encoding completed successfully!")

    # --------------------- Start Attendance ---------------------
    def start_attendance(self):
        if not self.encodeListKnown:
            messagebox.showwarning("Warning", "Please encode faces first!")
            return
        if self.running:
            messagebox.showinfo("Info", "Camera is already running.")
            return

        self.running = True
        self.stop_event.clear()
        self.status_label.config(text="Camera started. Press 'Stop Attendance' to exit.")
        self.camera_thread = threading.Thread(target=self.run_camera, daemon=True)
        self.camera_thread.start()

    # --------------------- Stop Attendance ---------------------
    def stop_attendance(self):
        if not self.running:
            messagebox.showinfo("Info", "Camera is not running.")
            return
        self.running = False
        self.stop_event.set()
        self.status_label.config(text="Camera stopping...")
        messagebox.showinfo("Stopped", "Camera stopped successfully!")

    # --------------------- Run Camera Thread ---------------------
    def run_camera(self):
        import face_recognition
        cap = cv2.VideoCapture(0)
        THRESHOLD = 0.5
        marked_today = set()

        while not self.stop_event.is_set():
            success, img = cap.read()
            if not success:
                break

            imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)
                matches = face_recognition.compare_faces(self.encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex] and faceDis[matchIndex] < THRESHOLD:
                    name = self.studentNames[matchIndex].upper()
                    if name not in marked_today:
                        # mark attendance in DB
                        markAttendanceMySQL(name)
                        marked_today.add(name)

                        # Update status label from main thread
                        self.root.after(0, lambda n=name: self.status_label.config(text=f"Attendance marked for {n}"))

                        # Show a message box informing attendance done (must be called from main thread)
                        self.root.after(0, lambda n=name: messagebox.showinfo("Attendance Done", f"Attendance marked for {n}"))

                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Attendance System - Press 'q' to close", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()
                break

        cap.release()
        cv2.destroyAllWindows()
        self.running = False
        # Ensure UI updates happen on main thread
        self.root.after(0, lambda: self.status_label.config(text="Camera stopped."))

    # --------------------- Show Today’s Attendance ---------------------
    def show_today_attendance(self):
        today = datetime.now().date()
        rows = fetch_attendance_by_date(today)
        self.tree.delete(*self.tree.get_children())
        for r in rows:
            self.tree.insert("", END, values=r)
        self.status_label.config(text=f"Showing attendance for {today}")

    # --------------------- Show Attendance by Date ---------------------
    def show_by_date(self):
        top = Toplevel(self.root)
        top.title("Search Attendance by Date")
        top.geometry("300x150")
        Label(top, text="Enter Date (YYYY-MM-DD):", font=("Arial", 12)).pack(pady=10)
        entry = Entry(top, font=("Arial", 12))
        entry.pack(pady=5)

        def search():
            date_val = entry.get()
            rows = fetch_attendance_by_date(date_val)
            self.tree.delete(*self.tree.get_children())
            for r in rows:
                self.tree.insert("", END, values=r)
            self.status_label.config(text=f"Showing attendance for {date_val}")
            top.destroy()

        Button(top, text="Search", command=search, font=("Arial", 12), bg="#3949ab", fg="white").pack(pady=10)

    # --------------------- Delete Selected Attendance ---------------------
    def delete_selected_attendance(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a record to delete!")
            return

        for item in selected:
            values = self.tree.item(item, "values")
            name, date, time = values
            delete_attendance_record(name, date, time)
            self.tree.delete(item)

        messagebox.showinfo("Deleted", "Selected attendance record(s) deleted successfully!")
        self.status_label.config(text="Selected record(s) deleted.")


if __name__ == "__main__":
    root = Tk()
    app = SmartAttendanceApp(root)
    root.mainloop()