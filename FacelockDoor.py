import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import serial
import time
import pyttsx3

data_path = 'C:\\Users\\jatot\\AppData\\Local\\Programs\\Python\\Images'
face_cascade_path = 'C:\\Users\\jatot\\AppData\\Local\\Programs\\Python\\Python38\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml'

face_classifier = cv2.CascadeClassifier(face_cascade_path)

def load_training_data(data_path):
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    Training_data, Labels = [], []

    for i, filename in enumerate(onlyfiles):
        image_path = join(data_path, filename)
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if images is not None:  # Check if the image was loaded successfully
            Training_data.append(np.asarray(images, dtype=np.uint8))
            Labels.append(i)

    Labels = np.asarray(Labels, dtype=np.int32)
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(Training_data), np.asarray(Labels))
    print("Training complete")
    return model

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty("voice", voices[0].id)
engine.setProperty("rate", 140)
engine.setProperty("volume", 1000)

def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return img, []
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))

    return img, roi

model = load_training_data(data_path)

cap = cv2.VideoCapture(0)
x = c = d = m = 0

confidence = 0  # Initialize confidence before the loop

while True:
    ret, frame = cap.read()

    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)
        
        if result[1] < 500:
            confidence = int((1 - (result[1]) / 300) * 100)
            display_string = str(confidence)
            cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 0))

        if confidence >= 83:
            cv2.putText(image, "unlocked", (250, 450), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 255))
            cv2.imshow('face', image)
            x += 1
        else:
            cv2.putText(image, "locked", (500, 500), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 255))
            cv2.imshow('face', image)
            c += 1
    except Exception as e:
        cv2.putText(image, "Face not found", (250, 450), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 255))
        cv2.imshow('face', image)
        d += 1
    
    if cv2.waitKey(1) == 13 or x == 10 or c == 30 or d == 20:
        break

cap.release()
cv2.destroyAllWindows()

if x >= 5:
    speak("Face recognition complete. It is matching with the database. Welcome Sucharan , The door is opening for 5 seconds.")
    m = 1
    ard = serial.Serial('com5', 9600)
    time.sleep(2)
    var = 'a'
    c = var.encode()
    
    ard.write(c)
    time.sleep(4)
elif c == 30:
    speak("Face Is Not Matching")
elif d == 20:
    speak("The face is not found. Please try again.")

if m == 1:
    speak("The door is closing.")
