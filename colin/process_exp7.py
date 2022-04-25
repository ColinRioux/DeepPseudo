"""
Author: Colin Rioux
Combine results of exp7
"""
import os
import pandas as pd
import csv
import pickle as pkl

SPLIT = "../data/staqc/dp-pair.csv"
GEN = "generated.csv"
TMP = "t.csv"

df1 = pd.read_csv(SPLIT)
df2 = pd.read_csv(GEN)

combo = pd.concat([df1, df2], axis=1)

old = None
data = {
    "sc": [],
    "nl": [],
    "ps": []
}

sc = []
ps = []
for _, row in combo.iterrows():
    if old is None:
        old = row['nl']
    
    if old == row['nl']:
        sc.append(row['sc'])
        ps.append(row['ps'])
    else:
        data['sc'].append("\\n".join(str(x) for x in sc))
        data['nl'].append(old)
        data['ps'].append("\\n".join(str(x) for x in ps))
        sc = []
        nl = []
        ps = []
        sc.append(row['sc'])
        ps.append(row['ps'])
    old = row['nl']
    

combo2 = pd.DataFrame(data)
combo2.to_csv(TMP, index=False, quoting=2)
