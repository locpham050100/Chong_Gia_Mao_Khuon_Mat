# CACH DUNG LENH
# python face_extract.py --input videos/real.mp4 --output dataset/real
# python face_extract.py --input videos/fake.mp4 --output dataset/fake

import numpy as np
import argparse
import cv2
import os

# Cac tham so dau vao(đối số)
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
                help="path to input video")
ap.add_argument("-o", "--output", type=str, required=True,
                help="path to output directory of cropped faces")
ap.add_argument("-d", "--detector", type=str, default='face_detector',
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip", type=int, default=1,
                help="# of frames to skip before applying face detection")
args = vars(ap.parse_args())

# Load model ssd nhan dien mat
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
                              "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# mo duong dan den tep video va khoi chay
# khoi tao 2 bien khung hinh doc va luu
# Doc file video input
vs = cv2.VideoCapture(args["input"])
read = 0
saved = 0

# Lap qua cac frame cua video
while True:
    # lay khung tu tep dang doc
    (grabbed, frame) = vs.read()
    # Neu khong doc duoc frame thi thoat
    if not grabbed:
        break
    # tang so khung hinh len them 1
    read += 1
    # kiem tra co nen xu ly khung hinh nay khong
    if read % args["skip"] != 0:
        continue

    # Chuyen tu frame thanh blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # Phat hien cac khuon mat trong frame
    net.setInput(blob)
    detections = net.forward()

    # Neu tim thay it nhat 1 khuon mat
    if len(detections) > 0:
        # Tim khuon  mat to nhat trong anh
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        # Neu muc do nhan dien > threshold
        if confidence > args["confidence"]:
            # tính tọa độ (x, y) của khung
            # Tach khuon mat va ghi ra file
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]

            # ghi khung vao disk
            p = os.path.sep.join([args["output"],
                                  args["input"].split('/')[1] + "{}.png".format(saved)])
            cv2.imwrite(p, face)
            saved += 1
            print("[INFO] saved {} to disk".format(p))

# don dep
vs.release()
cv2.destroyAllWindows()
