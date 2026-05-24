from .base import HfDatasetBase, SingletonDatasetBase

class HarmBenchDataset(SingletonDatasetBase):
    goal_column_name = "prompt"
    target_column_name = None # NOTE: No target column in this dataset
    dataset_name = "walledai/HarmBench"
    subset_name = "standard"
    split_name = "train"
    instance = None

    @classmethod
    def load(cls):
        if cls.instance is None:
            cls.instance = HfDatasetBase(
                dataset_name=cls.dataset_name,
                goal_column_name=cls.goal_column_name,
                target_column_name=cls.target_column_name,
                subset_name=cls.subset_name,
                split_name=cls.split_name
            )
        return cls.instance

    @classmethod
    def load_goals(cls):
        dataset = cls.load()
        return dataset.load_goals()

    @classmethod
    def load_targets(cls):
        dataset = cls.load()
        return dataset.load_targets()
