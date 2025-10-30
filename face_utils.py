import cv2
import numpy as np
import face_recognition
import os

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


def load_student_images(path):
    images = []
    studentNames = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Folder '{path}' not found.")

    myList = os.listdir(path)
    for imgName in myList:
        curImg = cv2.imread(os.path.join(path, imgName))
        if curImg is not None:
            images.append(curImg)
            studentNames.append(os.path.splitext(imgName)[0])
    return images, studentNames
