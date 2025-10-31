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
        self.root.title("Smart Attendance System")
        self.root.geometry("950x650")
        self.root.configure(bg="#f0f0f0")

        # Data
        self.images = []
        self.studentNames = []
        self.encodeListKnown = []
        self.path = ""

        # Camera / threading
        self.running = False
        self.camera_thread = None
        self.stop_event = threading.Event()

        # Toast window handle
        self._toast_win = None

        self.create_widgets()

    # --------------------- GUI Setup ---------------------
    def create_widgets(self):
        title = Label(self.root, text="Smart Attendance System",
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

        Button(frame, text="❌ Exit", command=self.on_exit,
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
        # don't block main thread here; camera will stop itself
        # show a toast that stop was requested
        self._show_toast("Stopping camera...", duration=1200)

    # --------------------- Safe app exit ---------------------
    def on_exit(self):
        # stop camera if running
        if self.running:
            self.stop_event.set()
        # give camera thread time to stop gracefully
        if self.camera_thread and self.camera_thread.is_alive():
            # Join with timeout to avoid blocking indefinitely
            self.camera_thread.join(timeout=1.0)
        self.root.quit()

    # --------------------- Toast helper ---------------------
    def _show_toast(self, message, duration=1500):
        """
        Show a small non-blocking toast message for `duration` milliseconds.
        Only one toast is shown at a time; new toasts replace the current one.
        This must be called from the main thread; when calling from camera thread, use self.root.after(0, ...).
        """
        # Destroy existing toast if present
        try:
            if hasattr(self, "_toast_win") and self._toast_win is not None:
                try:
                    self._toast_win.destroy()
                except Exception:
                    pass
                self._toast_win = None
        except Exception:
            self._toast_win = None

        # Create new toast window
        self._toast_win = Toplevel(self.root)
        self._toast_win.overrideredirect(True)  # no decorations
        try:
            # On some platforms this raises; protect it
            self._toast_win.attributes("-topmost", True)
        except Exception:
            pass

        label = Label(self._toast_win, text=message, bg="#333", fg="white",
                      font=("Arial", 11), padx=10, pady=6)
        label.pack()

        # Position toast at bottom-right of main window
        self.root.update_idletasks()
        rx = self.root.winfo_rootx()
        ry = self.root.winfo_rooty()
        rw = self.root.winfo_width()
        rh = self.root.winfo_height()

        tw = self._toast_win.winfo_reqwidth()
        th = self._toast_win.winfo_reqheight()

        x = rx + rw - tw - 20
        y = ry + rh - th - 20
        self._toast_win.geometry(f"+{x}+{y}")

        # Auto destroy after duration ms
        def _destroy_toast():
            try:
                if self._toast_win is not None:
                    self._toast_win.destroy()
            finally:
                self._toast_win = None

        self._toast_win.after(duration, _destroy_toast)

    # --------------------- Run Camera Thread ---------------------
    def run_camera(self):
        """
        Camera loop runs in a background thread. All tkinter UI updates must be scheduled
        through self.root.after(...) to run on the main thread.
        """
        import face_recognition  # keep import local to avoid startup costs if not used
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            # schedule an error message on the main thread
            self.root.after(0, lambda: messagebox.showerror("Camera Error", "Unable to open camera."))
            self.root.after(0, lambda: self.status_label.config(text="Camera error."))
            self.running = False
            return

        THRESHOLD = 0.5
        marked_today = set()

        try:
            while not self.stop_event.is_set():
                success, img = cap.read()
                if not success:
                    break

                # Process smaller image for speed
                imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
                imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

                facesCurFrame = face_recognition.face_locations(imgS)
                encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

                for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                    if not self.encodeListKnown:
                        continue
                    faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)
                    matches = face_recognition.compare_faces(self.encodeListKnown, encodeFace)
                    matchIndex = np.argmin(faceDis)

                    if matches[matchIndex] and faceDis[matchIndex] < THRESHOLD:
                        name = self.studentNames[matchIndex].upper()
                        if name not in marked_today:
                            try:
                                markAttendanceMySQL(name)
                            except Exception as e:
                                # Log DB error and inform user once
                                self.root.after(0, lambda err=str(e): messagebox.showerror("DB Error", f"Failed to mark attendance: {err}"))
                            else:
                                marked_today.add(name)
                                # Update status label (on main thread)
                                self.root.after(0, lambda n=name: self.status_label.config(text=f"Attendance marked for {n}"))
                                # Show a non-blocking toast (on main thread). This avoids blocking multiple message boxes.
                                self.root.after(0, lambda n=name: self._show_toast(f"Attendance marked for {n}", duration=1500))

                        # Draw rectangles and name on the image (for preview window)
                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, name, (x1 + 6, y2 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Show preview window. This is fine from a background thread for OpenCV windows.
                cv2.imshow("Attendance System - Press 'q' to close", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    # User requested to stop via preview window
                    self.stop_event.set()
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.running = False
            # Ensure UI update happens on main thread
            self.root.after(0, lambda: self.status_label.config(text="Camera stopped."))

    # --------------------- Show Today’s Attendance ---------------------
    def show_today_attendance(self):
        today = datetime.now().date()
        try:
            rows = fetch_attendance_by_date(today)
        except Exception as e:
            messagebox.showerror("DB Error", f"Failed to fetch attendance: {e}")
            return
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
            try:
                rows = fetch_attendance_by_date(date_val)
            except Exception as e:
                messagebox.showerror("DB Error", f"Failed to fetch attendance: {e}")
                return
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
            # Expecting (Name, Date, Time)
            try:
                name, date, time = values
            except ValueError:
                continue
            try:
                delete_attendance_record(name, date, time)
            except Exception as e:
                messagebox.showerror("DB Error", f"Failed to delete record: {e}")
                continue
            self.tree.delete(item)

        messagebox.showinfo("Deleted", "Selected attendance record(s) deleted successfully!")
        self.status_label.config(text="Selected record(s) deleted.")

# --------------------- Run App ---------------------
if __name__ == "__main__":
    root = Tk()
    app = SmartAttendanceApp(root)
    root.mainloop()