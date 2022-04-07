"""
Author: Colin Rioux
Process results of exp2
"""
import glob
import os
import pandas as pd
import argparse
import csv

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--data_path', required=True)
arg_parser.add_argument('--result_path', required=True)
args = arg_parser.parse_args()

""" Although its stored as a csv, it is a tsv """
folder = glob.glob(os.path.join(args.result_path, "*.csv"))

for f in folder:
  fname = f.split('/')[-1].split(',')[0]
  if "pred" in fname:
    pred_file = f
  else:
    true_file = f

test_data = os.path.join(args.data_path, "test.tsv")
df = pd.read_csv(test_data, sep="\t")
data = {"sc":[], "ps":[]}

with open(pred_file, 'r') as pred:
 lines = pred.readlines()
 i = 0
 for line in lines:
  data["ps"].append(line)
  data["sc"].append(df.at[i, "code"])
  i += 1

out_df = pd.DataFrame(data)
out_df.to_csv('exp2.tsv', sep='\t', index=False)
