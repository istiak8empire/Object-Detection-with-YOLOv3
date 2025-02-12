import cv2
import numpy as np

# Define paths to pre-trained model files (modify if needed)
classFile = 'coco.names'
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

# Read class names from file
classNames = []
with open(classFile,'rt') as f:
  classNames = f.read().rstrip('\n').split('\n')

# Load the pre-trained YOLOv3 model
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320, 320)  # Adjust input size if needed (based on your model)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Capture video from webcam (modify index if using multiple webcams)
cap = cv2.VideoCapture(0)

while True:
  # Capture frame-by-frame
  ret, frame = cap.read()

  # Optionally, rotate the frame if needed (modify based on your webcam orientation)
  # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # Uncomment for vertical webcam

  # Perform object detection
  classIds, confs, bbox = net.detect(frame, confThreshold=0.5)

  # Draw bounding boxes and labels for detected objects
  if len(classIds) != 0:
    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
      cv2.rectangle(frame, box, color=(0, 255, 0), thickness=3)
      cv2.putText(frame, classNames[classId - 1], (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

  # Display the resulting frame
  cv2.imshow('Real-time Object Detection', frame)

  # Exit if 'q' key is pressed
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Release capture resources
cap.release()
cv2.destroyAllWindows()
