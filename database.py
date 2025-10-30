import mysql.connector
from datetime import datetime
from tkinter import messagebox

# ------------------ MySQL Connection Helper ------------------
def get_connection():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="999888",   # 🔹 Change to your MySQL password
            database="smart_attendance"
        )
        return conn
    except mysql.connector.Error as err:
        messagebox.showerror("Database Error", f"Error: {err}")
        return None


# ------------------ Mark Attendance ------------------
def markAttendanceMySQL(name):
    conn = get_connection()
    if not conn:
        return

    cursor = conn.cursor()

    today = datetime.now().date()
    now_time = datetime.now().time()

    # Check if already marked today
    cursor.execute("SELECT * FROM attendance WHERE name=%s AND date=%s", (name, today))
    result = cursor.fetchone()

    if result:
        messagebox.showinfo("Attendance Info", f"Attendance already marked for {name} today!")
    else:
        cursor.execute("INSERT INTO attendance (name, date, time) VALUES (%s, %s, %s)", (name, today, now_time))
        conn.commit()
        print(f"✅ Attendance marked for {name} at {now_time}")
        messagebox.showinfo("Attendance Marked", f"Attendance marked for {name} at {now_time}")

    cursor.close()
    conn.close()


# ------------------ Fetch Attendance by Date ------------------
def fetch_attendance_by_date(date_val):
    conn = get_connection()
    if not conn:
        return []

    cursor = conn.cursor()
    cursor.execute("SELECT name, date, time FROM attendance WHERE date=%s ORDER BY time DESC", (date_val,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


# ------------------ Delete All Attendance ------------------
def delete_all_attendance():
    conn = get_connection()
    if not conn:
        return

    cursor = conn.cursor()
    cursor.execute("DELETE FROM attendance")
    conn.commit()
    cursor.close()
    conn.close()


# ------------------ Delete Single Record ------------------
def delete_attendance_record(name, date, time):
    conn = get_connection()
    if not conn:
        return

    cursor = conn.cursor()
    cursor.execute("DELETE FROM attendance WHERE name=%s AND date=%s AND time=%s", (name, date, time))
    conn.commit()
    cursor.close()
    conn.close()
