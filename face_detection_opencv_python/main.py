import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    # Get cam videoflow
    _, image = cap.read()
    image = cv2.resize(image, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
    image = cv2.flip(image, 1)

    # Face recognition
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    for (x, y, w, h) in faces:
        # Make rectangles
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        # Add image coordinates on rectangles
        x_center, y_center = x + w / 2, y + h / 2
        cv2.putText(image, f'{x_center, y_center}', (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    cv2.imshow('Image', image)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
