"""
Author: Colin Rioux
Convert staqc single-code problems into sc,nl dictionary saved as a csv
"""
import pandas as pd
import pickle as pkl

CODE = "../../../staqc/annotation_tool/data/code_solution_labeled_data/source/python_how_to_do_it_qid_by_classifier_unlabeled_single_code_answer_qid_to_code.pickle"

QUESTION = "../../../staqc/annotation_tool/data/code_solution_labeled_data/source/python_how_to_do_it_qid_by_classifier_unlabeled_single_code_answer_qid_to_title.pickle"

with open(CODE, 'rb') as f:
    ob = pkl.load(f)
    code_df = pd.DataFrame(ob, index=[0])

with open(QUESTION, 'rb') as f:
    ob = pkl.load(f)
    ques_df = pd.DataFrame(ob, index=[0])

data = {"sc": [], "nl": []}
for col in code_df.columns:
    data["sc"].append(repr(code_df.loc[0, col]))
    data["nl"].append(repr(ques_df.loc[0, col]))

data_df = pd.DataFrame(data)

data_df['sc'] = data_df['sc'].apply(lambda x: x.strip('"').strip("'"))
data_df['nl'] = data_df['nl'].apply(lambda x: x.strip('"').strip("'"))
data_df.to_csv("pair.csv", index=False, quoting=1)
