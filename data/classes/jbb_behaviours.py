from .base import HfDatasetBase, SingletonDatasetBase

class JbbBehavioursDataset(SingletonDatasetBase):
    goal_column_name = "Goal"
    target_column_name = "Target"
    dataset_name = "JailbreakBench/JBB-Behaviors"
    subset_name = "behaviors"
    split_name = "harmful"
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
