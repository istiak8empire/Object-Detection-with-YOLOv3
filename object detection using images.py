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

# Specify the image path (replace with your image file)
imagePath = 'IMG_20240523_122228.jpg'  # Modify this path to your image

# Read the image
img = cv2.imread(imagePath)

# Optionally, rotate the image if needed (modify based on your image orientation)
# img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # Uncomment for vertical image

# Perform object detection
classIds, confs, bbox = net.detect(img, confThreshold=0.5)

# Draw bounding boxes and labels for detected objects
if len(classIds) != 0:
  for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
    cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)
    cv2.putText(img, classNames[classId - 1], (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

# Display the image with detected objects
cv2.imshow('Object Detection', img)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()
