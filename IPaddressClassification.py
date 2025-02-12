import cv2 # opencv
import urllib.request # to open and read URL
import numpy as np

# OBJECT CLASSIFICATION PROGRAM FOR VIDEO FROM IP ADDRESS 

url = 'http://192.168.159.76/cam-hi.jpg'
#url = 'http://192.168.1.5/cam-lo.jpg'
#url = 'http://192.168.1.5/cam-hi.jpg'

winName = 'ESP32 CAMERA'
cv2.namedWindow(winName,cv2.WINDOW_AUTOSIZE)
#scale_percent = 80 # percent of original size    # for image processing

classNames = []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
#net.setInputSize(480,480)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while(1):
    imgResponse = urllib.request.urlopen(url) # open the URL
    imgNp = np.array(bytearray(imgResponse.read()),dtype=np.uint8)
    img = cv2.imdecode(imgNp,-1) # decode the image

    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) # rotate image vertically
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to black and white

    classIds, confs, bbox = net.detect(img,confThreshold=0.5)
    print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0,255,0), thickness=3) # draw rectangle around detected object
            cv2.putText(img, classNames[classId-1], (box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

    cv2.imshow(winName, img) # display the image

    # wait for ESC key to exit the program
    key = cv2.waitKey(5) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()
