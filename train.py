#Cách sử dụng:
# python train.py --dataset dataset --model liveness.model --le le.pickle
# thêm matplotlib để các cố liệu có thể được lưu trong nền
import matplotlib
matplotlib.use("Agg")

from model.livenessnet import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os

# các tham số đầu vào/ra (đối số)
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=True,
                help="path to label encoder")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output loss/accuracy plot")
args = vars(ap.parse_args())
# khởi tạo giá trị ban đầu cho epochs để đào tạo
# vận tốc, kích thước batch, số lượng
INIT_LR = 1e-4
BS = 8
EPOCHS = 50
# lấy hình ảnh trong dataset và tạo danh sách dữ liệu hình ảnh
# và hình ảnh trong các lớp
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []
# vòng lặp qua tất cả hình ảnh thông qua path
for imagePath in imagePaths:
    #lấy nhãn lớp từ tệp, tải hình ảnh đó và thay đổi kích thước
    # kích thước 32x32 pixel cố định, bỏ qua phần tỉ lệ khung hình
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32))
    # cập nhật lại dự liệu hình ảnh và nhãn tương ứng
    data.append(image)
    labels.append(label)
# chuyển đổi dữ liệu thành mảng Numpy, sau đó xử lý trước bằng cách chia tỉ lệ
# tất cả cường độ pixel có cường độ pixel từ [0,1]
data = np.array(data, dtype="float") / 255.0
# mã hoá nhãn(dạng chuỗi thành dạng số)
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, 2)
# Phân vùng dữ liệu
# 75% để train, 25% để thử nghiệm
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.25, random_state=42)
# tạo thêm hình ảnh đào tạo để tăng dữ liệu
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                         horizontal_flip=True, fill_mode="nearest")
# khởi chạy trình tối ưu và mô hình
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = LivenessNet.build(width=32, height=32, depth=3,
                          classes=len(le.classes_))
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])
# đào tạo mạng
print("[INFO] training network for {} epochs...".format(EPOCHS))
H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS),
              validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
              epochs=EPOCHS)
# đánh giá mạng
print("[INFO] evaluating network...")
predictions = model.predict(x=testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=le.classes_))
#lưu mạng vào disk
print("[INFO] serializing network to '{}'...".format(args["model"]))
model.save(args["model"], save_format="h5")
# lưu nhãn mã hoá vào disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()
# vẽ biểu đồ về tính chính xác và sự mất mát trong quá trình train
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
