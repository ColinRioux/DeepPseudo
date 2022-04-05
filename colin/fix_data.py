"""
Author: Colin Rioux
Fix input_data such that problem ids are unique
Assumes order of occurrence is unique
"""
import glob
import pandas as pd
import string
import random
import argparse
import os

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--data_path', default='./data/in/')
args = arg_parser.parser_args()

files = glob.glob(os.path.join(args.data_path, "*.tsv"))

for file in files:
    fname = file.split('/')[-1].split(',')[0]
    df = pd.read_csv(file, sep="\t")
    data = {}
    probs = {}

    for index, row in df.iterrows():
        worker_id, prob_id = int(row['workerid']), row['probid']
        if prob_id not in probs:
            probs[prob_id] = prob_id

        if prob_id not in data:
            data[probs[prob_id]] = {} 

        if worker_id in data[probs[prob_id]] and int(row['line']) == 0:
            probs[prob_id] = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
            data[probs[prob_id]] = {}
            data[probs[prob_id]][worker_id] = []
        elif worker_id not in data[probs[prob_id]]:
            data[probs[prob_id]][worker_id] = []
        row['probid'] = probs[prob_id]
        data[probs[prob_id]][worker_id].append(row)

    dfs = []
    for prob_id, problem in data.items():
        for worker_id, worker in problem.items():
            for w in worker:
                ndf = w
                #ndf = pd.DataFrame.from_dict(w, orient='index')
                dfs.append(ndf)

    final_df = pd.concat(dfs, axis=1).T.reset_index(drop=True)
    final_df.to_csv(str('./data/unique/uniq-') + fname, sep="\t")
