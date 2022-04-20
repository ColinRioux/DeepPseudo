"""
Author: Colin Rioux
Group SPoC pseudo code and source code together
*Should be run after fix_data
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

files = glob.glob(os.path.join(args.data_path, "*.tsv"))

for file in files:
    # fname = file.split('/')[-1].split(',')[0]
    fname = Path(file).stem
    """ Skip already grouped files """
    if "grouped-" in fname:
        continue
    fname = str("grouped-") + fname
    df = pd.read_csv(file, sep="\t")
    df = df.fillna('')
    data = {}
    
    """
    Group programs together
    """
    for index, row in df.iterrows():
        worker_id, prob_id = int(row['workerid']), row['probid']

        if prob_id not in data:
            data[prob_id] = {} 

        if worker_id not in data[prob_id]:
            data[prob_id][worker_id] = { "ps": [], "sc": [], "workerid": worker_id, "probid": prob_id, "subid": row['subid'] }
        
        data[prob_id][worker_id]["ps"].append(row['text'])
        # data[prob_id][worker_id]["sc"].append("".join(["\\t"*int(row['indent'])]) + row['code'])
        data[prob_id][worker_id]["sc"].append(row['code'])

    
    """
    Compress codes and texts
    """
    for prob_id, problem in data.items():
        for worker_id, worker in problem.items():
            worker["ps"] = [x for x in worker["ps"] if x]
            # worker["ps"] = "\\n".join(worker["ps"])
            worker["ps"] = "; ".join(worker["ps"])
            # worker["ps"] = '"' + worker["ps"].strip("'").strip('"') + '"'
            worker["ps"] = worker["ps"].strip("'").strip('"')
            worker["sc"] = [x for x in worker["sc"] if x]
            # worker["sc"] = "\\n".join(worker["sc"])
            worker["sc"] = " ".join(worker["sc"])
            # worker["sc"] = '"' + worker["sc"].strip("'").strip('"') + '"'
            worker["sc"] = worker["sc"].strip("'").strip('"')
    
    data_l = []
    for prob_id, problem in data.items():
        for worker_id, worker in problem.items():
            data_l.append(worker)
    
    final_df = pd.DataFrame(data_l)
    final_df.to_csv(os.path.join(args.out_path, fname + '.csv'), index=False)
