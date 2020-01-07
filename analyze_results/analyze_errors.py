import pickle
import math
import json
import matplotlib.pyplot as plt
import numpy as np
import pdb
import os

import pandas as pd


def get_errors(annotaions, predictions):
    error_larger_than_01, error_larger_than_005, error_less_than_005 = dict(), dict(), dict()
    labels_names = list(annotaions.keys())
    labels_names.remove("interview")
    for label_type in labels_names:
        value = annotaions[label_type]
        error_larger_than_01[label_type], error_larger_than_005[label_type], error_less_than_005[label_type] = [], [], []
        for utter_id, val in value.items():
            utter_id = utter_id[:-4]
            pred = predictions[utter_id][label_type]
            if abs(pred - val) >= 0.1:
                error_larger_than_01[label_type].append(val)
            elif abs(pred - val) >= 0.05:
                error_larger_than_005[label_type].append(val)
            else:
                error_less_than_005[label_type].append(val)
        print(f"{label_type}: >=0.1 {len(error_larger_than_01[label_type])} \
            / >=0.05 <0.1 {len(error_larger_than_005[label_type])} / <0.05 {len(error_less_than_005[label_type])}")

    errors = dict()
    for label_type in labels_names:
        errors[label_type] = [error_larger_than_01[label_type], error_larger_than_005[label_type], error_less_than_005[label_type]]
    
    return errors

# pdb.set_trace()

def get_counts(data):
    counts = np.zeros((20, 3))
    for i, lower in enumerate(np.arange(0, 1, 0.05)):
        higher = lower + 0.05
        for j in range(3):
            tmp_data = np.array(data[j])
            tmp_data = tmp_data * (tmp_data >= lower) * (tmp_data < higher)
            tmp_count = np.sum(tmp_data)
            counts[i][j] = tmp_count
    return counts


def get_hist(errors, filedir):
    # fig, axes = plt.subplots(5, 1, figsize=(20, 10))
    for i, label_type in enumerate(errors.keys()):
        df = pd.DataFrame(get_counts(errors[label_type]), columns=['>=0.1','>=0.05 <0.1','<0.05'])
        # df.plot(kind='bar', ax = axes[i], grid = True, color=['r', 'g', 'b'], stacked=True)
        df.plot(kind='bar', grid = True, color=['r', 'g', 'b'], stacked=True)

        filename = os.path.join(filedir, f"{label_type}_errhist.png")
        plt.savefig(filename)


if __name__ == "__main__":
    with open("../dataset/annotations/annotation_test.pkl", "rb") as f:
        annotaions = pickle.load(f, encoding='iso-8859-1')

    filedir = "../snapshots/tri_gelu_preres/"
    with open(os.path.join(filedir, "predictions.json"), "r") as f:
        predictions = json.load(f)

    errors = get_errors(annotaions, predictions)

    get_hist(errors, filedir)