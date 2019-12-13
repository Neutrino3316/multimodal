import cv2
import numpy as np
import os
import zipfile


def save_image(image, addr, file_name, num):
    address = addr + str(file_name) + '/frame' + str(num) + '.jpg'
    cv2.imwrite(address, image)


# Runnin a loop throught all the zipped training file to extract all video and then extract 100 frames from each.
for i in range(14, 26):
    if i < 10:
        zipfilename = 'test80_0' + str(i) + '.zip'
    else:
        zipfilename = 'test80_' + str(i) + '.zip'
    # Accessing the zipfile i
    archive = zipfile.ZipFile('data/train_data/' + zipfilename, 'r')
    zipfilename = zipfilename.split('.zip')[0]

    # Extracting all videos in it and saving it all to the new folder with same name as zipped one
    archive.extractall('data/unzippedData/' + zipfilename)

    # Running a loop over all the videos in the zipped file and extracting 100 frames from each
    for file_name in archive.namelist():
        cap = cv2.VideoCapture('data/unzippedData/' + zipfilename + '/' + file_name)

        file_name = (file_name.split('.mp4'))[0]
        # Creating folder to save all the 100 frames from the video
        try:
            if not os.path.exists('data/ImageData/testData/' + file_name):
                os.makedirs('data/ImageData/testData/' + file_name)
        except OSError:
            print('Error: Creating directory of data')
        c = 1
        count = 1
        timeF = 5
        success, frame = cap.read()
        while success:
            if c % timeF == 0:
                save_image(frame, 'data/ImageData/testData/', str(file_name), count)
                count += 1
            c += 1
            success, frame = cap.read()

        # Print the file which is done
        print(zipfilename, ':', file_name)
# path = './data/unzippedData'
# files = os.listdir(path)
# for file in files:
#     video_path = os.path.join(path, file)
#     all_videos = os.listdir(video_path)
#     for video_name in all_videos:
#         cap = cv2.VideoCapture('data/unzippedData/' + file + '/' + video_name)
#         video_name = (video_name.split('.mp4'))[0]
#         try:
#             if not os.path.exists('data/ImageData/trainingData/' + video_name):
#                 os.makedirs('data/ImageData/trainingData/' + video_name)
#         except OSError:
#             print('Error: Creating directory of data')
#         c = 1
#         count = 1
#         timeF = 5
#         success, frame = cap.read()
#         while success:
#             if c % timeF == 0:
#                 save_image(frame, 'data/ImageData/trainingData/', str(video_name), count)
#                 count += 1
#             c += 1
#             success, frame = cap.read()
#     print(file, ':', video_name)
