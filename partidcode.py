## Particle ID Code
# Contact: Cal Kaufman, clairekm@umich.edu
# Goal:  Live video particle trail identification for a Peltier Cloud Chamber
# Tools: Ultralytics (image analysis), Supervisely (data labeling)

import cv2
from ultralytics import YOLO

model = YOLO('runs/detect/train6/weights/best.pt')  # Trained model

# Open the webcam (0 = default camera)
cap = cv2.VideoCapture(0)
    # Debugging tip: If this code runs but you only see a black screen, try using camera '1'

# Check if webcam is opened
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()  # This automatically draws boxes & labels on the live video

    # Display the frame
    cv2.imshow("YOLOv8 Live Detection", annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
