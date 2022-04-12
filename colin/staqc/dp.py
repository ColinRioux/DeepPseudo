"""
Author: Colin Rioux
Convert staqc single-code problems into code,nl pairs
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

#print(code_df.head())
#print(ques_df.head())

data = {"sc": [], "nl": []}
for col in code_df.columns:
    data["sc"].append(repr(code_df.loc[0, col]))
    data["nl"].append(repr(ques_df.loc[0, col]))

with open('sc_nl.pkl', 'wb') as f:
    pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)

with open('sc_nl.pkl', 'rb') as f:
    unserialized_data = pkl.load(f)

print(data == unserialized_data)
