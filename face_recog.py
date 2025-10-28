import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pandas as pd

# Path to the folder containing student images
path = r"C:\Users\shaky\OneDrive\Desktop\SmartAttendance\images"

# Lists to store images and student names
images = []
studentNames = []

# Read all images from the folder
if not os.path.exists(path):
    raise FileNotFoundError(f"The folder '{path}' does not exist. Please check the path.")

myList = os.listdir(path)
print("Detected student images:", myList)

for imgName in myList:
    curImg = cv2.imread(os.path.join(path, imgName))
    if curImg is None:
        print(f"Warning: Could not read {imgName}, skipping.")
        continue
    images.append(curImg)
    studentNames.append(os.path.splitext(imgName)[0])

# Function to find encodings for known faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if encodes:
            encodeList.append(encodes[0])
        else:
            print("Warning: No face found in one of the images.")
    return encodeList

# Function to mark attendance in CSV file
def markAttendance(name):
    filename = 'attendance.csv'
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write("Name,DateTime\n")

    with open(filename, 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%Y-%m-%d %H:%M:%S')
            f.writelines(f'{name},{dtString}\n')
            print(f'Attendance marked for {name} at {dtString}')

# Step 1: Encode known faces
print("Encoding student faces...")
encodeListKnown = findEncodings(images)
print("Encoding complete")  # No emoji

# Step 2: Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise Exception("Could not open webcam. Make sure your camera is connected.")

# Threshold for face distance to reduce false positives
THRESHOLD = 0.5  # Adjust between 0.4-0.6 for better accuracy


while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame from webcam.")
        break

    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        # Check distance threshold
        if matches[matchIndex] and faceDis[matchIndex] < THRESHOLD:
            name = studentNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            # Scale back to original size
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam - Press Enter to Exit', img)
    if cv2.waitKey(1) == 13:  
        break

cap.release()
cv2.destroyAllWindows()
