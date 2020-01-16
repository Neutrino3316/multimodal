import os
import cv2
from mtcnn.mtcnn import MTCNN
from tqdm import tqdm
# outer_path = './data/ImageData/training/_0bg1TLPP-I.004'
# filelist = os.listdir(outer_path)  # 列举图片
# detector = MTCNN()
# for item in filelist:
#     src = os.path.join(os.path.abspath(outer_path), item)
#     input_img = cv2.imread(src)
#     output = os.path.join('after_MTCNN', item)
#     detected = detector.detect_faces(input_img)
#     if len(detected) > 0:  # 大于0则检测到人脸
#         x1, y1, w, h = detected[0]['box']
#         x2 = x1 + w
#         y2 = y1 + h
#         image = input_img[y1:y2, x1:x2]
#         cv2.imwrite(output, image)

def sep_face(data_path, outer_path):
    filelist = os.listdir(data_path)
    detector = MTCNN()
    for file in tqdm(filelist):
        imagefile = data_path + '/' + file
        output_file = outer_path + '/' + file
        try:
            if not os.path.exists(output_file):
                os.makedirs(output_file)
        except OSError:
            print('Error: Creating directory of data')
        images = os.listdir(imagefile)
        for image in images:
            src = os.path.join(os.path.abspath(imagefile), image)
            # print(src)
            input_img = cv2.imread(src)
            output = os.path.join(output_file, image)
            detected = detector.detect_faces(input_img)
            if len(detected) > 0:
                x1, y1, w, h = detected[0]['box']
                x2 = x1 + w
                y2 = y1 + h
                image = input_img[y1:y2, x1:x2]
                cv2.imwrite(output, image)

if __name__ == "__main__":
    sep_face('./data/ImageData/test', './after_MTCNN/test')
    sep_face('./data/ImageData/validation', './after_MTCNN/validation')
    sep_face('./data/ImageData/training', './after_MTCNN/training')