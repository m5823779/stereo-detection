import os
import cv2
import time
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import InceptionResNetV2


# Check GPU
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def build_model():
    model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
    model.trainable = False
    output = keras.layers.GlobalAveragePooling2D()(model.output)
    model = keras.models.Model(inputs=model.inputs, outputs=output)
    # model.summary()
    return model


def get_pressprocess_info():
    preprocess_input = keras.applications.inception_resnet_v2.preprocess_input
    img_size = (299, 299)
    return preprocess_input, img_size


def euclidean_distance(a, b):
    dist = np.linalg.norm(a - b)
    return dist


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', '--input', type=str, default="./test.mp4", help='Input video path')
parser.add_argument('-f', '--frame_rate', type=int, default=30, help='Inference frame rate (0-60)')
parser.add_argument('-s', '--SBS', action='store_true', default=False, help='If input video content ground is SBS input true')
args = parser.parse_args()

if (args.frame_rate >= 60):
    frame_rate = 59
else:
    frame_rate = args.frame_rate

preprocess_input, input_size = get_pressprocess_info()
model = build_model()

cap = cv2.VideoCapture(args.input)

total_frame_num = 0
sbs_frame_num = 0
non_sbs_frame_num = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("\nCan't receive frame (stream end?). Exiting ...")
        print(f'SBS Frame : {sbs_frame_num}' )
        print(f'Non SBS Frame : {non_sbs_frame_num}')
        break

    if (total_frame_num % (60 - frame_rate) == 0):
        img = cv2.resize(frame, (1280, 720))

        left = img[0:, 0:int(img.shape[1] / 2)]
        right = img[0:, int(img.shape[1] / 2):]

        left = cv2.resize(left, input_size)
        right = cv2.resize(right, input_size)

        left_ = np.expand_dims(left, axis=0)
        right_ = np.expand_dims(right, axis=0)

        left_tensor = preprocess_input(left_)
        right_tensor = preprocess_input(right_)

        left_enc = model.predict(left_tensor)[0]
        right_enc = model.predict(right_tensor)[0]

        dis = euclidean_distance(left_enc, right_enc)

        if (0.5 <= dis <= 10):
            result = 'SBS'
            cv2.putText(img, result, (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            sbs_frame_num += 1
        else:
            result = 'Non SBS'
            cv2.putText(img, result, (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            non_sbs_frame_num += 1

        cv2.imshow('frame', img)
        if cv2.waitKey(1) == ord('q'):
            break

        if args.SBS:
            accuracy = sbs_frame_num / (sbs_frame_num + non_sbs_frame_num)
        else:
            accuracy = non_sbs_frame_num / (sbs_frame_num + non_sbs_frame_num)

        print(f'\rAccuracy: {round(accuracy * 100, 1)} %', end=' ')
    total_frame_num += 1

cap.release()
cv2.destroyAllWindows()
