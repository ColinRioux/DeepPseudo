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
from pathlib import Path

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--data_path', default='./data/in/')
arg_parser.add_argument('--out_path', default='./data/uniq/')
args = arg_parser.parse_args()

files = glob.glob(os.path.join(args.data_path, "*.csv"))

for file in files:
    # fname = file.split('/')[-1].split(',')[0]
    fname = Path(file).stem
    """ Skip non-grouped files """
    if "grouped-" not in fname:
        continue
    if "eval" in fname:
        fname = str("valid.csv")
    elif "test" in fname:
        fname = str("test.csv")
    else:
        fname = str("train.csv")
    df = pd.read_csv(file)
    data = []
    
    for index, row in df.iterrows():
        d = {}
        d["sc"] = row["sc"]
        d["ps"] = row["ps"]
        data.append(d)
 
    final_df = pd.DataFrame(data)
    final_df.to_csv(os.path.join(args.out_path, fname), index=False)
