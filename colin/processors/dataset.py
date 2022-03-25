"""
Dataset Base Class (the template other processors should use)

Author: Colin Rioux
"""
import typing
import os.path as path

from ../problem import Problem
import pandas as pd

class Dataset:
    """
    A basic dataset processor for csv based datasets
    - A subclass can be used to override these methods if the dataset does not follow the conventional csv format

    @property id:str the id of the dataset
    @property data_path:str the file path to the train, val, and test sets
    @property train:pd.DataFrame the training set
    @property val:pd.DataFrame the validation set
    @property test:pd.DataFrame the test set
    """
    def __init__(self, dsid: str, data_path: str, sep: str = ",") -> Dataset:
        """
        @param dsid:str the name of the dataset
        @param data_path:str the file path to the train, val, and test sets
        @param sep:str the separator for the csv file
        """
        self.id = dsid
        self.data_path = data_path
        self.train = pd.read_csv(path.join(data_path, "train.csv"), sep=sep)
        self.val = pd.read_csv(path.join(data_path, "val.csv"), sep=sep)
        self.test = pd.read_csv(path.join(data_path, "test.csv"), sep=sep)


    def toProblem(self, t: str) -> Problem:
        """
        Converts the train, val, or test dataset to a Problem tree

        @param t:str "train", "val", or "test"
        """
        df = self.train if t == "train" else self.val if t == "val" else self.test
        print(df.head())
