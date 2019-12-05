import os
import pickle
# import pdb
# # /data/ccb/Pers_Detect/dataset/val-annotation-e
#
# # with open("/data/ccb/Pers_Detect/dataset/val-transcription/transcription_validation.pkl", "rb") as f:
# #     annotations = pickle.load(f, encoding='iso-8859-1')
# # # annotations = pickle.load("/data/ccb/Pers_Detect/dataset/val-annotation-e/annotation_validation.pkl")
# # pdb.set_trace()
# # print(annotations)
# # # pickle.loads
#
# # /data/ccb/Pers_Detect/dataset/
# import zipfile
# files = zipfile.ZipFile("/data/ccb/Pers_Detect/dataset/train-1.zip", "r")
#
# for file in files.namelist():
#     print(file)
#
# content = files.read(files.namelist()[0])
# print(type(content))


# path = "../dataset/trainset/"
# dir = os.listdir(path)
# mp3_files = filter(lambda file: file.split(".")[-1] == "mp3", dir)  # filter out files in mp3 format
# mp3_files_train = list(mp3_files)
# filenames_train = []
# for file in mp3_files_train:
#     filename = file.split(".")[0]
#     filenames_train.append(filename)
#
# path = "../dataset/testset/"
# dir = os.listdir(path)
# mp3_files = filter(lambda file: file.split(".")[-1] == "mp3", dir)  # filter out files in mp3 format
# mp3_files_test = list(mp3_files)
# filenames_test = []
# for file in mp3_files_test:
#     filename = file.split(".")[0]
#     filenames_test.append(filename)
#
# count = 0
# filenames = []
# for file in filenames_test:
#     if file in filenames_train:
#         count += 1
#         filenames.append(file)
#
# # print(filenames_test)
# print(filenames)
# print(count)
# print(len(filenames_train), len(filenames_test))


with open("../dataset/preprocessed/preproc_audio_test.pkl", "rb") as f:
    trainset = pickle.load(f)
# with open(os.path.join(data_path, "preprocessed/preproc_audio_valid.pkl"), "rb") as f:
#     validset = pickle.load(f)
# with open(os.path.join(data_path, "preprocessed/preproc_audio_test.pkl"), "rb") as f:
#     testset = pickle.load(f)

shapes = []
count = 0
for dt in trainset:
    shapes.append(dt.feature.shape[1])
    if dt.feature.shape[1] != 611:
        count += 1
print(max(shapes))
print(count)