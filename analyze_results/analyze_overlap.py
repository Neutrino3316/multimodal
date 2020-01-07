## analyse whether overlapping affect predicting

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pdb

with open("../snapshots/tri_gelu_preres/predictions.json", "r") as f:
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

train_person_id = [key.split(".")[0] for key in annotaions_training['extraversion'].keys()]
predictions_unique_id = list(predictions.keys())
gt_nonoverlap, gt_overlap = dict(), dict()
for key, value in ground_truth.items():
    person_id = key.split(".")[0]
    # pdb.set_trace()
    assert (key[:-4] in predictions_unique_id)
    if person_id in train_person_id:
        gt_overlap[key] = value
    else:
        gt_nonoverlap[key] = value

n_nonoverlap, n_overlap = len(gt_nonoverlap.keys()), len(gt_overlap.keys())
print("{:4d} non-overlap instances, {:4d} overlap instances in total.".format(n_nonoverlap, n_overlap))

error_overlap = []
for key, value in gt_overlap.items():
    key = key[:-4]
    preds = predictions[key]
    preds = np.array(list(preds.values()))
    gt = np.array(list(value.values()))
    error = np.mean(np.abs(preds - gt))
    error_overlap.append(error)

error_nonoverlap = []
for key, value in gt_nonoverlap.items():
    key = key[:-4]
    preds = predictions[key]
    preds = np.array(list(preds.values()))
    gt = np.array(list(value.values()))
    error = np.mean(np.abs(preds - gt))
    error_nonoverlap.append(error)

error_overlap, error_nonoverlap = np.array(error_overlap), np.array(error_nonoverlap)
assert error_nonoverlap.shape[0] == n_nonoverlap
mean_error_overlap, mean_error_nonoverlap = np.mean(error_overlap), np.mean(error_nonoverlap)
print("Average error of non-overlap instances: {:.4f}, "
        "average error of overlap instances: {:.4f}".format(mean_error_nonoverlap, mean_error_overlap))