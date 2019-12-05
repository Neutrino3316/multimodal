import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

# f = open("../snapshots/exp2/analysis_of_result.txt", "w", encoding="utf-8")

with open("../snapshots/exp_unif_labels/true_predictions.json", "r") as f:
    predictions = json.load(f)

with open("../dataset/annotations/annotation_test.pkl", "rb") as f:
    annotaions_test = pickle.load(f, encoding='iso-8859-1')

with open("../dataset/annotations/annotation_training.pkl", "rb") as f:
    annotaions_training = pickle.load(f, encoding='iso-8859-1')


labels_names = list(annotaions_test.keys())
print(labels_names)
labels_names.remove("interview")
ground_truth = dict()
for key in labels_names:
    value = annotaions_test[key]
    for k, v in value.items():
        if k not in ground_truth:
            ground_truth[k] = dict()
        ground_truth[k][key] = v

# ## analyse whether overlapping affect predicting
# train_person_id = [key.split(".")[0] for key in annotaions_training['extraversion'].keys()]
# predictions_unique_id = list(predictions.keys())
# gt_nonoverlap, gt_overlap = dict(), dict()
# for key, value in ground_truth.items():
#     person_id = key.split(".")[0]
#     assert (key in predictions_unique_id)
#     if person_id in train_person_id:
#         gt_overlap[key] = value
#     else:
#         gt_nonoverlap[key] = value
#
# n_nonoverlap, n_overlap = len(gt_nonoverlap.keys()), len(gt_overlap.keys())
# # f.write("{:4d} non-overlap instances, {:4d} overlap instances in total.".format(n_nonoverlap, n_overlap))
# print("{:4d} non-overlap instances, {:4d} overlap instances in total.".format(n_nonoverlap, n_overlap))
#
# error_overlap = []
# for key, value in gt_overlap.items():
#     preds = predictions[key]
#     preds = np.array(list(preds.values()))
#     gt = np.array(list(value.values()))
#     error = np.mean(np.abs(preds - gt))
#     error_overlap.append(error)
#
# error_nonoverlap = []
# for key, value in gt_nonoverlap.items():
#     preds = predictions[key]
#     preds = np.array(list(preds.values()))
#     gt = np.array(list(value.values()))
#     error = np.mean(np.abs(preds - gt))
#     error_nonoverlap.append(error)
#
# error_overlap, error_nonoverlap = np.array(error_overlap), np.array(error_nonoverlap)
# assert error_nonoverlap.shape[0] == n_nonoverlap
# mean_error_overlap, mean_error_nonoverlap = np.mean(error_overlap), np.mean(error_nonoverlap)
# # f.write("Average error of non-overlap instances: {:.4f}, "
# #         "average error of overlap instances: {:.4f}".format(mean_error_nonoverlap, mean_error_overlap))
# print("Average error of non-overlap instances: {:.4f}, "
#         "average error of overlap instances: {:.4f}".format(mean_error_nonoverlap, mean_error_overlap))

# # f.close()

# ## distribution of errors
# plt.figure()
# plt.subplot(3, 1, 1)
# plt.hist(error_overlap, color='b')
# plt.ylabel("Frequency")
# plt.xlim(0, 0.5)
# plt.ylim(0, 800)
# plt.title("Histogram of overlapping instances' errors")
# plt.tight_layout()
# plt.subplot(3, 1, 2)
# plt.hist(error_nonoverlap, color='b')
# plt.ylabel("Frequency")
# plt.xlim(0, 0.5)
# plt.ylim(0, 800)
# plt.title("Histogram of non-overlapping instances' errors")
# plt.tight_layout()
# plt.subplot(3, 1, 3)
# plt.hist(list(error_overlap) + list(error_nonoverlap), color='b')
# plt.ylabel("Frequency")
# plt.xlabel("Value")
# plt.xlim(0, 0.5)
# plt.ylim(0, 800)
# plt.title("Histogram of all instances' errors")
# plt.tight_layout()
# plt.savefig("../snapshots/exp2/err_hist.png")


## distribution of predictions and groundtruth
preds, gt = dict(), dict()
for name in labels_names:
    preds[name], gt[name] = [], []
    for unique_id in ground_truth.keys():
        preds[name].append(predictions[unique_id][name])
        gt[name].append(ground_truth[unique_id][name])
# histogram plot
for name in labels_names:
    assert len(preds[name]) == len(gt[name]) == 2000
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.hist(preds[name], bins=np.arange(0, 1, 0.05), color='b')
    plt.ylabel("Count")
    plt.title(f"Prediction of {name}")
    plt.xlim(0, 1)
    plt.ylim(0, 500)
    plt.tight_layout()
    plt.subplot(2, 1, 2)
    plt.hist(gt[name], bins=np.arange(0, 1, 0.05), color='b')
    plt.ylabel("Count")
    plt.xlabel("values")
    plt.title(f"Ground truth of {name}")
    plt.xlim(0, 1)
    plt.ylim(0, 500)
    plt.tight_layout()
    plt.savefig(f"../snapshots/exp_unif_labels/{name}_hist.png")

# ## set all predictions to 0.5, get the accuracy
# accuracy = np.zeros(len(labels_names))
# for i, name in enumerate(labels_names):
#     accuracy[i] = np.mean(np.abs(np.array(gt[name])-0.5))
# print(f"Aaverage accuracy {1 - np.mean(accuracy)} if all predictions are set to 0.5")
