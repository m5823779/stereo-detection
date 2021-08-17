import os
import cv2
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


preprocess_input, input_size = get_pressprocess_info()
model = build_model()

cap = cv2.VideoCapture('./input.mp4')

while cap.isOpened():
    ret, frame = cap.read()
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

    if dis <= 10:
        result = 'SBS'
        cv2.putText(img, result, (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    else:
        result = 'Non SBS'
        cv2.putText(img, result, (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('frame', img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
