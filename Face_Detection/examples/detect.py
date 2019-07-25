import cv2
import sys
import os.path

#def detect(filename, cascade_file = "../lbpcascade_animeface.xml"):
def detect(filename, cascade_file = "C:/Users/unlea/OneDrive/Desktop/DeepLearning_Study-master/Anime_Pupil_Color_Detection/Face_Detection/lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
    for (x, y, w, h) in faces:
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2) # 아직 확실치않아 주석처리
        cutted = image[y:y+h, x:x+w] # 이미지를 자르는 부분

        cv2.imshow("AnimeFaceDetect", image)
        cv2.imshow("cutted", cutted) # 자른 이미지도 보여주기
        cv2.waitKey(0)
        cv2.imwrite(filename + "_cutted.jpg", cutted)

#if len(sys.argv) != 2:
#    sys.stderr.write("usage: detect.py <filename>\n")
#    sys.exit(-1)

img_path = 'C:/Users/unlea/OneDrive/Desktop/DeepLearning_Study-master/Anime_Pupil_Color_Detection/Face_Detection/examples/image_original'    
#detect(sys.argv[1])

print(os.listdir(img_path))
filelist = os.listdir(img_path)
for file in filelist:
    print(file)
    detect(img_path + '/' + file)
