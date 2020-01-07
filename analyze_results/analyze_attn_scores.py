import pickle
import pdb

with open("../snapshots/tri_gelu_preres/attn_scores.pkl", "rb") as f:
    scores = pickle.load(f)

pdb.set_trace()