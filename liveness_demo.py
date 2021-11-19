# cách dùng
# python liveness_demo.py
import imutils
from imutils.video import VideoStream
from keras.models import load_model
from keras.preprocessing.image import img_to_array

import argparse
import os
import pickle
import time
import cv2
import numpy as np

# môi trường opencv
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;0"

# Cai dat cac tham so dau vao/ra (đối số)
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default='liveness.model',
                help="path to trained model")
ap.add_argument("-l", "--le", type=str, default='le.pickle',
                help="path to label encoder")
ap.add_argument("-d", "--detector", type=str, default='face_detector',
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Load model nhan dien khuon mat
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
                              "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Load model nhan dien fake/real và nhãn từ disk
print("[INFO] loading liveness detector...")
model = load_model(args["model"])
le = pickle.loads(open(args["le"], "rb").read())

#  Doc video tu webcam
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# chờ 2s
time.sleep(2.0)
#lặp các khung từ video stream
while True:
    # Doc anh tu webcam
    #lấy khung hình từ video và thay đổi kích thước
    # chiều rộng tối đa 600 pixel
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    # Chuyen thanh blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # Phat hien khuon mat
    net.setInput(blob)
    detections = net.forward()

    # Loop qua cac khuon mat
    for i in range(0, detections.shape[2]):
        #lấy độ tin cậy được liên kết với dự đoán
        confidence = detections[0, 0, i, 2]

        # Neu conf lon hon threshold(lọc ra các ảnh yếu)
        if confidence > args["confidence"]:
            #tính toạ độ x,y cho box giới hạn cho khuôn mặt
            # trích xuất ROI khuôn mặt
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            #đảm bảo khuôn mặt không nằm ngoài khung
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            # Lay vung khuon mat
            face = frame[startY:endY, startX:endX]
            face = cv2.resize(face, (32, 32))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            # Dua vao model de nhan dien fake/real
            preds = model.predict(face)[0]
            j = np.argmax(preds)
            label = le.classes_[j]

            # Ve hinh chu nhat quanh mat
            label = "{}: {:.4f}".format(label, preds[j])
            if j == 0:
                # Neu la fake thi ve mau do
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
            else:
                # Neu real thi ve mau xanh
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)
    # hiện thị khung đầu ra và đợi 1 lần ấn phím
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Bam 'q' de thoat
    if key == ord("q"):
        break

# dọn dẹp
cv2.destroyAllWindows()
vs.stop()
