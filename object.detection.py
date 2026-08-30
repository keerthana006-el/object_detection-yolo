from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open the webcam
cap = cv2.VideoCapture(0)

# Set camera resolution to HD (1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Optional: increase camera frame rate
cap.set(cv2.CAP_PROP_FPS, 30)

# Create a resizable window
cv2.namedWindow("YOLO Object Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Object Detection", 1000, 700)

while True:
    # Capture frame
    ret, frame = cap.read()

    if not ret:
        print("Cannot access camera")
        break

    # Run YOLO detection
    results = model(frame)

    # Draw detection boxes and labels
    annotated_frame = results[0].plot()

    # Display camera
    cv2.imshow("YOLO Object Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release camera
cap.release()
cv2.destroyAllWindows()