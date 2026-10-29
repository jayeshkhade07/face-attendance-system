"""
AMS_Run.py
----------
Main file for the Face Recognition Attendance Management System.
This file builds the GUI (graphical window) and connects all features.

How to run:
    python AMS_Run.py

Requirements:
    pip install opencv-contrib-python Pillow pandas

Features:
    1. Take Images    - Captures 70 face photos for a student
    2. Train Images   - Trains the AI model using captured photos
    3. Automatic Attendance - Recognizes faces and fills attendance
    4. Manually Fill Attendance - Fill attendance without face recognition
    5. Check Registered Students - Admin panel to view registered students
"""

import tkinter as tk
from tkinter import Label, Button, messagebox
import cv2
import csv
import os
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time


REQUIRED_FOLDERS = [
    'TrainingImage',
    'TrainingImageLabel',
    'Attendance',
    'Attendance/Manually Attendance',
    'StudentDetails'
]
for folder in REQUIRED_FOLDERS:
    os.makedirs(folder, exist_ok=True)


CASCADE_PATH        = "haarcascade_frontalface_alt.xml"
STUDENT_DETAILS_CSV = "StudentDetails/StudentDetails.csv"
TRAINER_MODEL_PATH  = "TrainingImageLabel/Trainner.yml"
TRAINING_IMAGE_FOLDER = "TrainingImage"



window = tk.Tk()
window.title("FAMS - Face Recognition Based Attendance Management System")
window.geometry('1280x720')
window.configure(background='grey80')



def show_warning(message):
    
    popup = tk.Toplevel(window)
    popup.title("Warning!")
    popup.geometry("360x110")
    popup.configure(background='white')
    popup.grab_set()  

    Label(popup, text=message, fg='red', bg='white',
          font=('times', 14, 'bold'), wraplength=320).pack(pady=15)
    Button(popup, text="OK", command=popup.destroy,
           fg="black", bg="lawn green", width=9,
           font=('times', 13, 'bold')).pack()



def digits_only(new_value, action_type):
   
    if action_type == '1': 
        if not new_value.isdigit():
            return False
    return True



def take_img():
    
    enrollment = txt.get().strip()
    name = txt2.get().strip()

    if not enrollment or not name:
        show_warning("Please enter both Enrollment Number and Name!")
        return

    face_detector = cv2.CascadeClassifier(CASCADE_PATH)
    if face_detector.empty():
        show_warning(f"Cascade file not found:\n{CASCADE_PATH}")
        return

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        show_warning("Cannot open camera. Please check your webcam.")
        return

    sample_count = 0
    MAX_SAMPLES  = 2  

    print(f"Capturing images for: {name} (Enrollment: {enrollment})")

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            sample_count += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            cv2.imwrite(f"TrainingImage/{name}.{enrollment}.{sample_count}.jpg", gray)
            cv2.putText(frame, f"Captured: {sample_count}/{MAX_SAMPLES}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Capturing Face Images (Press Q to stop early)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if sample_count >= MAX_SAMPLES:
            break

    cam.release()
    cv2.destroyAllWindows()

  
    ts         = time.time()
    date_today = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    time_now   = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

    file_is_new = not os.path.exists(STUDENT_DETAILS_CSV) or \
                  os.path.getsize(STUDENT_DETAILS_CSV) == 0
    with open(STUDENT_DETAILS_CSV, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if file_is_new:
            writer.writerow(['Enrollment', 'Name', 'Date', 'Time'])
        writer.writerow([enrollment, name, date_today, time_now])

    Notification.configure(
        text=f"Images saved for {name} (Enrollment: {enrollment})",
        bg="SpringGreen3", fg="black", width=55, font=('times', 14, 'bold'))
    Notification.place(x=200, y=400)
    print(f"Done! {sample_count} images saved.")



def getImagesAndLabels(folder_path):
   
    image_paths  = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    face_samples = []
    ids          = []

    for image_path in image_paths:
        pil_image = Image.open(image_path).convert('L')  # Grayscale
        image_np  = np.array(pil_image, 'uint8')

        
        try:
            student_id = int(os.path.split(image_path)[-1].split(".")[1])
        except (IndexError, ValueError):
            print(f"  Skipping (unexpected filename): {image_path}")
            continue

        faces = detector.detectMultiScale(image_np)
        for (x, y, w, h) in faces:
            face_samples.append(image_np[y: y + h, x: x + w])
            ids.append(student_id)

    return face_samples, ids


def trainimg():
    """Trains the LBPH face recognizer and saves the model."""
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        try:
            recognizer = cv2.face.createLBPHFaceRecognizer()
        except AttributeError:
            Notification.configure(
                text="Please install opencv-contrib-python",
                bg="red", fg="white", width=50, font=('times', 14, 'bold'))
            Notification.place(x=200, y=400)
            return

    global detector
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        Notification.configure(
            text=f"Cascade file missing: {CASCADE_PATH}",
            bg="red", fg="white", width=50, font=('times', 14, 'bold'))
        Notification.place(x=200, y=400)
        return

    if not os.path.exists(TRAINING_IMAGE_FOLDER) or \
       len(os.listdir(TRAINING_IMAGE_FOLDER)) == 0:
        Notification.configure(
            text="No images found! Please take images first.",
            bg="orange", fg="black", width=50, font=('times', 14, 'bold'))
        Notification.place(x=200, y=400)
        return

    Notification.configure(text="Training... Please wait.",
                           bg="yellow", fg="black", width=50,
                           font=('times', 14, 'bold'))
    Notification.place(x=200, y=400)
    window.update()  

    faces, ids = getImagesAndLabels(TRAINING_IMAGE_FOLDER)

    if len(faces) == 0:
        Notification.configure(
            text="No faces detected. Check image quality.",
            bg="red", fg="white", width=50, font=('times', 14, 'bold'))
        Notification.place(x=200, y=400)
        return

    recognizer.train(faces, np.array(ids))
    recognizer.save(TRAINER_MODEL_PATH)

    Notification.configure(
        text=f"Training complete! {len(set(ids))} student(s) trained.",
        bg="olive drab", fg="white", width=50, font=('times', 14, 'bold'))
    Notification.place(x=200, y=400)
    print(f"Model saved to {TRAINER_MODEL_PATH}")



def subjectchoose():
    """Opens a window to pick subject, then runs face-recognition attendance."""

    def fill_attendance():
        subject = subject_entry.get().strip()
        if not subject:
            show_warning("Please enter the Subject name!")
            return

        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
        except AttributeError:
            try:
                recognizer = cv2.face.createLBPHFaceRecognizer()
            except AttributeError:
                status_label.configure(text="Install opencv-contrib-python",
                                       bg="red", fg="white", width=33,
                                       font=('times', 13, 'bold'))
                status_label.place(x=20, y=250)
                return

        if not os.path.exists(TRAINER_MODEL_PATH):
            status_label.configure(text="Model not found! Train first.",
                                   bg="red", fg="white", width=33,
                                   font=('times', 13, 'bold'))
            status_label.place(x=20, y=250)
            return
        recognizer.read(TRAINER_MODEL_PATH)

        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

        
        try:
            df = pd.read_csv(STUDENT_DETAILS_CSV)
            if 'Enrollment' not in df.columns:
                df = pd.read_csv(STUDENT_DETAILS_CSV,
                                 names=['Enrollment', 'Name', 'Date', 'Time'])
        except Exception:
            df = pd.DataFrame(columns=['Enrollment', 'Name', 'Date', 'Time'])

        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            show_warning("Cannot open camera.")
            return

        font       = cv2.FONT_HERSHEY_SIMPLEX
        attendance = pd.DataFrame(columns=['Enrollment', 'Name', 'Date', 'Time'])
        end_time   = time.time() + 20  

        while time.time() < end_time:
            ret, frame = cam.read()
            if not ret:
                continue

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)

            for (x, y, w, h) in faces:
                student_id, confidence = recognizer.predict(gray[y: y + h, x: x + w])

                if confidence < 70:  
                    name_match   = df.loc[df['Enrollment'] == student_id]['Name'].values
                    student_name = name_match[0] if len(name_match) > 0 else "Unknown"

                    ts   = time.time()
                    now_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    now_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    attendance.loc[len(attendance)] = [student_id, student_name,
                                                       now_date, now_time]

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv2.putText(frame, f"{student_id} - {student_name}",
                                (x, y - 10), font, 0.8, (255, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 25, 255), 3)
                    cv2.putText(frame, "Unknown", (x, y - 10),
                                font, 0.8, (0, 25, 255), 2)

            attendance = attendance.drop_duplicates(subset=['Enrollment'], keep='first')
            remaining  = int(end_time - time.time())
            cv2.putText(frame, f"Time left: {remaining}s  (ESC to stop)",
                        (10, 30), font, 0.7, (255, 255, 255), 2)
            cv2.imshow('Filling Attendance...', frame)
            if cv2.waitKey(30) & 0xFF == 27:
                break

        cam.release()
        cv2.destroyAllWindows()

        ts       = time.time()
        date_str = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        time_str = datetime.datetime.fromtimestamp(ts).strftime('%H-%M-%S')
        csv_filename = f"Attendance/{subject}_{date_str}_{time_str}.csv"

        attendance = attendance.drop_duplicates(subset=['Enrollment'], keep='first')
        attendance.to_csv(csv_filename, index=False)

        status_label.configure(
            text=f"Done! {len(attendance)} student(s) marked present.",
            bg="Green", fg="white", width=33, font=('times', 13, 'bold'))
        status_label.place(x=20, y=250)
        print(f"Attendance saved: {csv_filename}")
        show_csv_table(csv_filename, title=f"Attendance - {subject}")

    subject_window = tk.Toplevel(window)
    subject_window.title("Select Subject")
    subject_window.geometry('580x320')
    subject_window.configure(background='grey80')
    subject_window.grab_set()

    status_label = tk.Label(subject_window, text="", bg="grey80",
                            fg="white", width=33, height=2,
                            font=('times', 13, 'bold'))

    tk.Label(subject_window, text="Enter Subject : ", width=15, height=2,
             fg="black", bg="grey", font=('times', 15, 'bold')).place(x=30, y=100)
    subject_entry = tk.Entry(subject_window, width=20, bg="white",
                             fg="black", font=('times', 23))
    subject_entry.place(x=250, y=105)

    tk.Button(subject_window, text="Fill Attendance", fg="white",
              command=fill_attendance, bg="SkyBlue1", width=20, height=2,
              activebackground="white",
              font=('times', 15, 'bold')).place(x=250, y=160)



def manually_fill():
    """Opens a window to manually enter student attendance (no camera)."""

    def open_manual_form():
        subject = subject_entry.get().strip()
        if not subject:
            show_warning("Please enter the Subject name!")
            return
        subject_window.destroy()

        ts           = time.time()
        date_str     = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d')
        time_str     = datetime.datetime.fromtimestamp(ts).strftime('%H_%M_%S')
        csv_filename = f"Attendance/Manually Attendance/{subject}_{date_str}_{time_str}.csv"
        manual_records = []  

        form_window = tk.Toplevel(window)
        form_window.title(f"Manual Attendance - {subject}")
        form_window.geometry('880x470')
        form_window.configure(background='grey80')
        form_window.grab_set()

        tk.Label(form_window, text="Enrollment No:", width=15, height=2,
                 fg="black", bg="grey", font=('times', 15)).place(x=30, y=100)
        tk.Label(form_window, text="Student Name:", width=15, height=2,
                 fg="black", bg="grey", font=('times', 15)).place(x=30, y=200)

        enr_entry  = tk.Entry(form_window, width=20, bg="white",
                              fg="black", font=('times', 23))
        enr_entry.place(x=290, y=105)

        name_entry = tk.Entry(form_window, width=20, bg="white",
                              fg="black", font=('times', 23))
        name_entry.place(x=290, y=205)

        status_lbl = tk.Label(form_window, text="", bg="grey80",
                              width=33, height=2, font=('times', 14, 'bold'))

        def add_record():
            enrollment   = enr_entry.get().strip()
            student_name = name_entry.get().strip()
            if not enrollment or not student_name:
                show_warning("Please fill both Enrollment and Name!")
                return
            record_date  = datetime.datetime.now().strftime('%Y-%m-%d')
            record_time  = datetime.datetime.now().strftime('%H:%M:%S')
            manual_records.append([enrollment, student_name, record_date, record_time])
            status_lbl.configure(text=f"Added: {student_name}", bg="SkyBlue1", fg="black")
            status_lbl.place(x=180, y=310)
            enr_entry.delete(0, 'end')
            name_entry.delete(0, 'end')

        def save_csv():
            if not manual_records:
                show_warning("No records to save! Add at least one student.")
                return
            with open(csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Enrollment', 'Name', 'Date', 'Time'])
                writer.writerows(manual_records)
            status_lbl.configure(text="CSV saved!", bg="Green", fg="white")
            status_lbl.place(x=180, y=310)
            print(f"Manual attendance saved: {csv_filename}")
            show_csv_table(csv_filename, title=f"Manual Attendance - {subject}")

        tk.Button(form_window, text="Clear",
                  command=lambda: enr_entry.delete(0, 'end'),
                  fg="white", bg="black", width=13, height=1,
                  font=('times', 12, 'bold')).place(x=690, y=105)
        tk.Button(form_window, text="Clear",
                  command=lambda: name_entry.delete(0, 'end'),
                  fg="white", bg="black", width=13, height=1,
                  font=('times', 12, 'bold')).place(x=690, y=205)
        tk.Button(form_window, text="Add Student", command=add_record,
                  fg="black", bg="SkyBlue1", width=20, height=2,
                  font=('times', 15, 'bold')).place(x=170, y=360)
        tk.Button(form_window, text="Save to CSV", command=save_csv,
                  fg="black", bg="SkyBlue1", width=20, height=2,
                  font=('times', 15, 'bold')).place(x=480, y=360)

    subject_window = tk.Toplevel(window)
    subject_window.title("Enter Subject Name")
    subject_window.geometry('580x320')
    subject_window.configure(background='grey80')
    subject_window.grab_set()

    tk.Label(subject_window, text="Enter Subject : ", width=15, height=2,
             fg="black", bg="grey80", font=('times', 15, 'bold')).place(x=30, y=100)
    subject_entry = tk.Entry(subject_window, width=20, bg="white",
                             fg="black", font=('times', 23))
    subject_entry.place(x=250, y=105)
    tk.Button(subject_window, text="Continue", command=open_manual_form,
              fg="black", bg="SkyBlue1", width=20, height=2,
              font=('times', 15, 'bold')).place(x=250, y=160)



def admin_panel():
    """Admin login → shows all registered students from CSV."""
    ADMIN_USERNAME = "admin"
    ADMIN_PASSWORD = "admin123"

    login_window = tk.Toplevel(window)
    login_window.title("Admin Login")
    login_window.geometry('880x420')
    login_window.configure(background='grey80')
    login_window.grab_set()

    status_lbl = tk.Label(login_window, text="", bg="grey80", fg="white",
                          width=40, height=2, font=('times', 14, 'bold'))

    def do_login():
        if username_entry.get().strip() == ADMIN_USERNAME and \
           password_entry.get().strip() == ADMIN_PASSWORD:
            login_window.destroy()
            if not os.path.exists(STUDENT_DETAILS_CSV):
                show_warning("No student data found yet.\nPlease take images first.")
                return
            show_csv_table(STUDENT_DETAILS_CSV, title="Registered Students")
        else:
            status_lbl.configure(text="Incorrect username or password!",
                                 bg="red", fg="white")
            status_lbl.place(x=120, y=350)

    tk.Label(login_window, text="Username : ", width=15, height=2,
             fg="black", bg="grey", font=('times', 15, 'bold')).place(x=30, y=50)
    tk.Label(login_window, text="Password : ", width=15, height=2,
             fg="black", bg="grey", font=('times', 15, 'bold')).place(x=30, y=150)

    username_entry = tk.Entry(login_window, width=20, bg="white",
                              fg="black", font=('times', 23))
    username_entry.place(x=290, y=55)

    password_entry = tk.Entry(login_window, width=20, show="*",
                              bg="white", fg="black", font=('times', 23))
    password_entry.place(x=290, y=155)

    tk.Button(login_window, text="Clear",
              command=lambda: username_entry.delete(0, 'end'),
              fg="white", bg="black", width=10, height=1,
              font=('times', 13, 'bold')).place(x=690, y=55)
    tk.Button(login_window, text="Clear",
              command=lambda: password_entry.delete(0, 'end'),
              fg="white", bg="black", width=10, height=1,
              font=('times', 13, 'bold')).place(x=690, y=155)
    tk.Button(login_window, text="Log In", fg="black", bg="SkyBlue1",
              width=20, height=2, command=do_login,
              font=('times', 15, 'bold')).place(x=290, y=250)



def show_csv_table(csv_path, title="Data"):
    """Opens a new window and shows a CSV file as a neat grid table."""
    try:
        table_window = tk.Toplevel(window)
        table_window.title(title)
        table_window.configure(background='grey80')

        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            for row_index, row in enumerate(reader):
                for col_index, cell_value in enumerate(row):
                    bg_color = "SkyBlue1" if row_index == 0 else "white"
                    tk.Label(table_window, text=cell_value, width=15,
                             height=1, fg="black", bg=bg_color,
                             font=('times', 13, 'bold'),
                             relief=tk.RIDGE).grid(row=row_index, column=col_index,
                                                   padx=1, pady=1)
    except FileNotFoundError:
        show_warning(f"File not found:\n{csv_path}")
    except Exception as e:
        show_warning(f"Could not open file:\n{str(e)}")



def clear():
    txt.delete(0, 'end')

def clear1():
    txt2.delete(0, 'end')

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        window.destroy()

window.protocol("WM_DELETE_WINDOW", on_closing)





tk.Label(window, text="Face Recognition Attendance System",
         bg="green", fg="white", width=50, height=2,
         font=('times', 28, 'bold')).place(x=80, y=20)


tk.Label(window, text="Enter Enrollment : ", width=20, height=2,
         fg="black", bg="grey", font=('times', 15, 'bold')).place(x=200, y=200)
txt = tk.Entry(window, validate="key", width=20, bg="white",
               fg="black", font=('times', 25))
txt['validatecommand'] = (txt.register(digits_only), '%P', '%d')
txt.place(x=550, y=210)


tk.Label(window, text="Enter Name : ", width=20, fg="black",
         bg="grey", height=2, font=('times', 15, 'bold')).place(x=200, y=300)
txt2 = tk.Entry(window, width=20, bg="white", fg="black", font=('times', 25))
txt2.place(x=550, y=310)


tk.Button(window, text="Clear", command=clear, fg="white", bg="black",
          width=10, height=1, font=('times', 15, 'bold')).place(x=950, y=210)
tk.Button(window, text="Clear", command=clear1, fg="white", bg="black",
          width=10, height=1, font=('times', 15, 'bold')).place(x=950, y=310)


Notification = tk.Label(window, text="System Ready",
                        bg="Green", fg="white", width=15,
                        height=3, font=('times', 15))


tk.Button(window, text="Check Registered Students", command=admin_panel,
          fg="black", bg="SkyBlue1", width=22, height=1,
          font=('times', 14, 'bold')).place(x=980, y=410)


tk.Button(window, text="Take Images", command=take_img,
          fg="black", bg="SkyBlue1", width=20, height=3,
          font=('times', 15, 'bold')).place(x=90, y=500)

tk.Button(window, text="Train Images", command=trainimg,
          fg="black", bg="SkyBlue1", width=20, height=3,
          font=('times', 15, 'bold')).place(x=390, y=500)

tk.Button(window, text="Automatic Attendance", command=subjectchoose,
          fg="black", bg="SkyBlue1", width=20, height=3,
          font=('times', 15, 'bold')).place(x=690, y=500)

tk.Button(window, text="Manually Fill Attendance", command=manually_fill,
          fg="black", bg="SkyBlue1", width=20, height=3,
          font=('times', 15, 'bold')).place(x=990, y=500)

tk.Button(window, text="Exit", command=on_closing,
          fg="black", bg="tomato", width=10, height=2,
          font=('times', 15, 'bold')).place(x=590, y=630)


window.mainloop()
