"""
🎯 Face Detection using OpenCV
================================
This script uses your webcam to detect faces in real time.
It draws a green rectangle around every face it finds.

HOW IT WORKS:
- We use a pre-trained model called "Haar Cascade"
- It scans the image in small windows looking for face patterns
- When it finds a face, it draws a box around it

BEGINNER TIP:
- Press 'q' on your keyboard to quit the program
"""

import cv2  # OpenCV library for computer vision

# ✅ STEP 1: Load the face detector model
# OpenCV comes with a pre-trained face detector — we just load it!
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ✅ STEP 2: Open your webcam
# 0 = default webcam. Use 1 if you have an external camera.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam. Make sure your webcam is connected.")
    exit()

print("✅ Webcam opened! Press 'q' to quit.")
print("📷 Face detection is running...")

# ✅ STEP 3: Loop — read frames from the webcam one by one
while True:
    ret, frame = cap.read()  # Read one frame from the webcam

    if not ret:
        print("❌ Failed to grab frame.")
        break

    # ✅ STEP 4: Convert to grayscale
    # Face detection works better on grayscale (black & white) images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ✅ STEP 5: Detect faces in the grayscale image
    # scaleFactor: how much image size reduces at each scale (1.1 = 10% smaller each time)
    # minNeighbors: how many neighbors each face must have (higher = fewer false detections)
    # minSize: minimum face size to detect
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    # ✅ STEP 6: Draw a rectangle around each detected face
    for (x, y, w, h) in faces:
        # Draw green rectangle: (x, y) is top-left, (x+w, y+h) is bottom-right
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add a label above the rectangle
        cv2.putText(frame, f'Face', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ✅ STEP 7: Show face count on screen
    cv2.putText(frame, f'Faces found: {len(faces)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    # ✅ STEP 8: Show the frame in a window
    cv2.imshow('Face Detection - Press Q to Quit', frame)

    # ✅ STEP 9: Wait for 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Quitting...")
        break

# ✅ STEP 10: Release camera and close windows
cap.release()
cv2.destroyAllWindows()
print("✅ Program ended cleanly.")
