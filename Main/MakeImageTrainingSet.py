from keras import applications
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Dropout, Flatten, Dense
import os
from keras import backend as K

K.set_image_dim_ordering('th')

import imageGenerator

# ROOT_DIR = Root of this project (Anime_Pupil_Color_Detection)
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
TRAINING_SET_DIR = os.path.join(ROOT_DIR, 'image_cut')
GENERATED_TRAINING_SET_DIR = os.path.join(ROOT_DIR, 'Generated_training_set')


# 생성된 Training Set
trainingGenerator = imageGenerator.ImageGenerate(TRAINING_SET_DIR,GENERATED_TRAINING_SET_DIR, Traincount=5, Debug=0)


# padding 옵션은 에러때문에 추가함. 지워야할수도?
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(3, 30, 30)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # 이전 CNN 레이어에서 나온 3차원 배열은 1차원으로 뽑아줍니다
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


model.fit_generator(trainingGenerator,
                    steps_per_epoch=10)

model.save_weights('first_try.h5')





