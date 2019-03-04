# import the necessary packages
from imutils.video import VideoStream
from time import gmtime, strftime
import numpy as np
import argparse
import imutils
import time
import cv2
import requests
import sys

# model settings
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


# IP-CAM settings
CAM_IP = "1.2.3.4"
USERNAME = "instar"
PASSWORD = "instar"
width, height = (1280,720)


# pushover credentials
token = ""
user = ""
msg_type = "Alert"

def pushover(image_path):
    r = requests.post("https://api.pushover.net/1/messages.json", data = {
        "token": token,
        "user": user,
        "message": msg_type
    },
    files = {
        "attachment": (image_path, open(image_path, "rb"), "image/jpeg")
    })


print("[USAGE] instar_recog.py [-h] [-p PROTOTXT] [-m MODEL] [-c CONFIDENCE]\n")

def main():

    # parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", default="MobileNetSSD_deploy.prototxt.txt",
        help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", default="MobileNetSSD_deploy.caffemodel",
        help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.8,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # load serialized model from disk
    print("[INFO] loading classification model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


    print("[INFO] starting video stream...")

    #local webcam 
    vs = cv2.VideoCapture(1)

    # remote ip cam (instar)
    '''
    try:
        vs = cv2.VideoCapture("http://" + CAM_IP  + "/cgi-bin/hi3510/mjpegstream.cgi?-chn=11&-usr=" + USERNAME + "&-pwd=" + PASSWORD)
        time.sleep(2.0)
        fps = FPS().start()
        rval, frame = vs.read()
    except:
        print("[FAILURE] Could not connect to camera!")
        sys.exit ()
    ''' 

    print("[INFO] loop, press q to stop the script")

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        try:
            rval, frame = vs.read()
            frame = imutils.resize(frame, width=400)

            # grab the frame dimensions and convert it to a blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                0.007843, (300, 300), 127.5)
        except:
            continue

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the                
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > args["confidence"]:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
 
                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx],
                    confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                # Write file to disk
                '''
                date = strftime("%Y-%m-%d %H:%M:%S", gmtime())
                cv2.imwrite("alert_" + date + ".jpg", frame)
                '''

                # Send file over pushover
                '''
                pushover(image_path)
                '''
                
        # debug the output        
        frame = cv2.resize(frame,(width,height))
        cv2.imshow("Frame", frame)        


        # if the q key was pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
 
    print("[INFO] stopping the script")

    # do a bit of cleanup
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
