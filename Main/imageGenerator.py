# This file is for generating training set with keras.ImageDataGenerator.
# WARNING: 테스트 중일땐 img_cut에 절대 많은 수의 사진을 넣지 말것! 몇천장 지우는데도 한세월임.
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

# rotation_range: 이미지 회전 범위 (degrees)
#
# width_shift, height_shift: 그림을 수평 또는 수직으로 랜덤하게 평행 이동시키는 범위 (원본 가로, 세로 길이에 대한 비율 값)
#
# rescale: 원본 영상은 0-255의 RGB 계수로 구성되는데, 이 같은 입력값은 모델을 효과적으로 학습시키기에 너무 높습니다 (통상적인 learning rate를 사용할 경우).
# 그래서 이를 1/255로 스케일링하여 0-1 범위로 변환시켜줍니다. 이는 다른 전처리 과정에 앞서 가장 먼저 적용됩니다.
# -> 이 프로젝트는 색을 구분하는 프로젝트이기때문에 단순 Object Detection이 아님. 따라서 나는 1/10 정도로만 스케일링.... 했는데 그냥 1/255로 스케일링 한거랑 변화가 없길래 놔둠.
#
# shear_range: 임의 전단 변환 (shearing transformation) 범위 ( 기울이기)
#
# zoom_range: 임의 확대/축소 범위
#
# horizontal_flip: True로 설정할 경우, 50% 확률로 이미지를 수평으로 뒤집습니다. 원본 이미지에 수평 비대칭성이 없을 때 효과적입니다. 즉, 뒤집어도 자연스러울 때 사용하면 좋습니다.
#
# fill_mode 이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식
#
# 위 설명을 참고하여 아래의 값을 잘 조정해서 데이터셋을 만들어야 함.
# 참고로 shift 관련 기능도 있는데 테스트해보니 우리한테 부적절한 것으로 보여 아예 파라미터에서 뺐음.

datagen = ImageDataGenerator(
    rotation_range=40,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')


def ImageGenerate(TRAIN_DIR, GENE_DIR, Traincount=20, Debug=0):
    # Traincount = 이미지 하나당 몇개의 변형 이미지를 생성할 것인가?
    # Debug = 생성된 이미지의 단순 확인을 위해서인가?

    if Debug is 1:
        TrainingList = os.listdir(TRAIN_DIR)
        for originalImg in TrainingList:
            img = load_img(os.path.join(TRAIN_DIR, originalImg))  # PIL 이미지
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            i = 1
            for batch in datagen.flow(x, batch_size=1, save_to_dir=GENE_DIR, save_prefix='gen',
                                      save_format='jpeg'):
                i += 1
                if i > Traincount:
                    break
        return None

    # 넘겨줄 이미지의 수
    train_generator = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(30, 30),
        batch_size=10,
        class_mode='categorical'
    )

    return train_generator
