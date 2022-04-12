"""
Author: Colin Rioux
Convert staqc pkl file to a single col tsv of code for use with DP
"""
import pickle as pkl
import pandas as pd

PAIR = "sc_nl.pkl"
OUT = "test.tsv"

with open(PAIR, 'rb') as f:
    data = pkl.load(f)

with open(OUT, 'w') as f:
    for code in data["sc"]:
        f.write(code)
        f.write("\n")
