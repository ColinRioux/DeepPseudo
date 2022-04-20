"""
Author: Colin Rioux
Combine results of exp3
"""
import os
import pandas as pd
import argparse
import csv
import pickle as pkl

arg_parser = argparse.ArgumentParser()
# arg_parser.add_argument('--data_path', required=True)
arg_parser.add_argument('--result_path', required=True)
args = arg_parser.parse_args()

""" Although its stored as a csv, it is a tsv """
result = os.path.join(args.result_path, "generated.tsv")
data = './staqc/sc_nl.pkl'
with open(data, 'rb') as f:
  data_dic = pkl.load(f)

result_df = pd.read_csv(result, sep='\t')
result_df['ps'] = result_df['ps'].apply(lambda x: "'" + str(x) + "'")

data_df = pd.DataFrame(data_dic)
data_df['sc'] = data_df['sc'].apply(lambda x: "'" + x.strip('"').strip("'") + "'")
data_df['nl'] = data_df['nl'].apply(lambda x: "'" + x.strip('"').strip("'") + "'")

combo = pd.concat([data_df, result_df], axis=1)

combo.to_csv("exp3.tsv", sep='\t', quoting=3, index=False)