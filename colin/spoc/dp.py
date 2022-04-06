"""
Author: Colin Rioux
Convert grouped spoc into code\tnl tsvs
*Should be run after group
"""
import glob
import pandas as pd
import string
import random
import argparse
import os

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--data_path', default='./data/in/')
arg_parser.add_argument('--out_path', default='./data/uniq/')
args = arg_parser.parse_args()

files = glob.glob(os.path.join(args.data_path, "*.tsv"))

for file in files:
    fname = file.split('/')[-1].split(',')[0]
    """ Skip non-grouped files """
    if "grouped-" not in fname:
        continue
    if "eval" in fname:
        fname = str("dp-eval.tsv")
    elif "test" in fname:
        fname = str("dp-test.tsv")
    else:
        fname = str("dp-train.tsv")
    df = pd.read_csv(file, sep="\t")
    data = []
    
    for index, row in df.iterrows():
        d = {}
        d["code"] = row["sc"]
        d["nl"] = row["ps"]
        data.append(d)
 
    final_df = pd.DataFrame(data)
    final_df.to_csv(os.path.join(args.out_path, fname), sep="\t", index=False)
