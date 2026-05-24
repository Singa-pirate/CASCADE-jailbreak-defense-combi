from abc import ABC, abstractmethod
import pandas as pd
from datasets import load_dataset

class SingletonDatasetBase(ABC):
    @classmethod
    @abstractmethod
    def load(cls):
        pass
    
    @classmethod
    @abstractmethod
    def load_goals(cls):
        pass
    
    @classmethod
    @abstractmethod
    def load_targets(cls):
        pass


class DatasetBase(ABC):
    def __init__(self, goal_column_name=None, target_column_name=None):
        self.goal_column_name = goal_column_name
        self.target_column_name = target_column_name
    
    @abstractmethod
    def load_goals(self):
        pass
    
    @abstractmethod
    def load_targets(self):
        pass


class CsvDatasetBase(DatasetBase):
    def __init__(self, csv_file_path, goal_column_name, target_column_name):
        super().__init__(goal_column_name, target_column_name)
        self.csv_file_path = csv_file_path
        self.df = pd.read_csv(csv_file_path)

    def load_goals(self):
        if self.goal_column_name not in self.df.columns:
            raise Exception(f"Column '{self.goal_column_name}' not found in the CSV file.")
        return self.df[self.goal_column_name].tolist()
    
    def load_targets(self):
        if self.target_column_name is None:
            return ["Sure, here is" for _ in range(len(self.df))]
        if self.target_column_name not in self.df.columns:
            raise Exception(f"Column '{self.target_column_name}' not found in the CSV file.")
        return self.df[self.target_column_name].tolist()


class HfDatasetBase(DatasetBase):
    def __init__(self, dataset_name, goal_column_name, target_column_name, subset_name=None, split_name=None):
        super().__init__(goal_column_name, target_column_name)
        args = [dataset_name]
        kwargs = {}
        if subset_name:
            args.append(subset_name)
        if split_name:
            kwargs['split'] = split_name
        self.ds = load_dataset(*args, **kwargs)

    def load_goals(self):
        if self.goal_column_name not in self.ds.column_names:
            raise Exception(f"Column '{self.goal_column_name}' not found in the dataset.")
        return self.ds[self.goal_column_name]
    
    def load_targets(self):
        if self.target_column_name is None:
            return ["Sure, here is" for _ in range(len(self.ds))] # for case where the dataset doesn't have a target column, return a default response
        if self.target_column_name not in self.ds.column_names:
            raise Exception(f"Column '{self.target_column_name}' not found in the dataset.")
        return self.ds[self.target_column_name]
