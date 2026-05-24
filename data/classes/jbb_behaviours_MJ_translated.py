from .base import CsvDatasetBase, SingletonDatasetBase

class JbbBehavioursMJTranslatedDataset(SingletonDatasetBase):
    goal_column_name = "en"
    target_column_name = None # NOTE: No target column in this dataset
    csv_file_path = "data/jbb_behaviours_MJ_translated.csv"
    instance = None
    target_languages = ['en', 'zh-CN', 'it', 'vi', 'ar', 'ko', 'th', 'bn', 'sw', 'jv']

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
    def load_goals_translation_dict(cls):
        dataset = cls.load()
        data = dataset.df.to_dict(orient='records')
        translations_dict = {row['en']: {lang: row[lang] for lang in cls.target_languages} for row in data}
        return translations_dict

    @classmethod
    def load_targets(cls):
        print("Warning: This dataset does not have target columns.")
        return None
