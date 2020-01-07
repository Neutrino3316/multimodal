import matplotlib.pyplot as plt
import pickle
import json
import os
import numpy as np

def get_ground_truth_distr():
    with open("../dataset/annotations/annotation_test.pkl", "rb") as f:
        annotations_test = pickle.load(f, encoding='iso-8859-1')
    annotations = dict()
    for label_type, value in annotations_test.items():
        annotations[label_type] = list(value.values())
    return annotations

def get_prediction_distr(filedir):
    filename = os.path.join(filedir, "predictions.json")
    with open(filename, "r") as f:
        predictions = json.load(f)
    labels_names = list(predictions.values())[0].keys()
    preds = dict()
    for label in labels_names:
        preds[label] = []
    for utter_id, value in predictions.items():
        for label, val in value.items():
            preds[label].append(val)
    return preds

def get_histograms(annotations, preds, filedir):
    for label in preds.keys():
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.hist(preds[label], bins=np.arange(0, 1, 0.05), color='b')
        plt.ylabel("Count")
        plt.title(f"Prediction of {label}")
        plt.xlim(0, 1)
        plt.ylim(0, 500)
        plt.tight_layout()
        plt.subplot(2, 1, 2)
        plt.hist(annotations[label], bins=np.arange(0, 1, 0.05), color='b')
        plt.ylabel("Count")
        plt.xlabel("values")
        plt.title(f"Ground truth of {label}")
        plt.xlim(0, 1)
        plt.ylim(0, 500)
        plt.tight_layout()
        filename = os.path.join(filedir, f"{label}_hist.png")
        plt.savefig(filename)

if __name__ == "__main__":
    filedir = "../snapshots/tri_gelu_preres/"
    annotations = get_ground_truth_distr()
    predictions = get_prediction_distr(filedir)
    get_histograms(annotations, predictions, filedir)