from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os

# Validation 을 위한 DataGenerator
# 이미지 재조정기능 삭제

datagen = ImageDataGenerator(
    data_format="channels_last")


def ImageGenerate(TRAIN_DIR, GENE_DIR, batch_size=1, Traincount=20):

    # 넘겨줄 이미지의 수
    train_generator = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(30, 30),
        batch_size=batch_size,
        class_mode='binary'
    )

    return train_generator