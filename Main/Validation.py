import tensorflow as tf
import numpy as np
import cv2
from keras.models import load_model
import matplotlib.pyplot as plt

import os
import cv2

import ValidationImageGenerator

# set path and Image Generator
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
VALIDATION_SET_DIR = os.path.join(ROOT_DIR, 'validation_set')
GENERATED_TRAINING_SET_DIR = os.path.join(ROOT_DIR, 'Generated_training_set')

testGenerator = ValidationImageGenerator.ImageGenerate(VALIDATION_SET_DIR, GENERATED_TRAINING_SET_DIR, batch_size=1)

# load keras model
model = load_model('first_try.h5')
model.summary()

# 전체 테스트셋의 정확도
print("-- Evaluate --")
scores = model.evaluate_generator(testGenerator, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# 테스트셋의 예상 라벨값을 보여준다
print("-- Predict --")
output = model.predict_generator(testGenerator, steps=len(testGenerator))
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(testGenerator.class_indices)
print(output)

# 개별 이미지의 예측값 보여줌
testIm = plt.imread(os.path.join(ROOT_DIR, 'validation_set', 'Black', '4840111.png_cutted+resized (1).jpg'))
plt.imshow(testIm)
plt.waitforbuttonpress()

testIm_reshape = testIm.reshape((1, 30, 30, 3))
print(testGenerator.class_indices)
print("Predict:", model.predict(testIm_reshape))