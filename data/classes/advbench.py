from .base import CsvDatasetBase, SingletonDatasetBase

class AdvBenchDataset(SingletonDatasetBase):
    goal_column_name = "goal"
    target_column_name = "target"
    csv_file_path = "data/advbench_harmful_behaviors.csv"
    instance = None

    @classmethod
    def load(cls):
        if cls.instance is None:
            cls.instance = CsvDatasetBase(
                csv_file_path=cls.csv_file_path,
                goal_column_name=cls.goal_column_name,
                target_column_name=cls.target_column_name
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
