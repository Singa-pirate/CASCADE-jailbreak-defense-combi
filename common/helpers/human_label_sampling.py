import os, random
import pandas as pd


def generate_dataset_for_human_label(folder1, folder2, folder3, output_folder_name, num_samples_per_technique=50):
    # for all subfolders of folder1, folder2 and folder3,
    # the subfolders contain parquet files that ends with "jbb.parquet"
    # for a filename in subfolder of folder1, the corresponding filname is subfolder of folder2 is to replace "oss-jb-judge" with "ft-roberta-judge"
    # the corresponding filname in subfolder of folder3 is to replace "oss-jb-judge" with "harmbench-judge"
    # for each set of 3 corresponding files, first read the parquet files, then check they have same number of rows
    # and the they all contain columns "goal", "response" and "jailbroken"
    # if a file contains "variant" and "rep_idx" columns, must sort the df by these 2 columns before sampling

    # for each subfolder, combine such pq files into a single df for each judge; this df contains result of the technique on the judge
    # then, obtain a 4th df that only contains "jailbroken" column, which is the majority vote of the 3 dfs
    # then, obtain the number of positive samples (jailbroken=True) and negative samples in the 4th df
    # ideally, sample half of num_samples_per_technique from positive and half from negative, but if one of them is not enough, then sample as many as possible from that category and fill the rest from the other category
    # after sampling the row indices, obtain the corresponding rows from the 3 dfs
    # build up a sampled df for each of the 3 judges containing "goal", "resposne", "jailbroken", "evaluator_output" columns
    # also build up a separate df containing "goal", "response", "jailbroken" columns, the jailbroken column is empty for later human label
    # in the end, save the 4 dfs to output_folder_name, the filename for the 3 judge dfs is same as judge name and is in parquet format, the filename for the human label df is "human_label.csv" in csv format
    # the output folder should be created if it does not exist

    random.seed(42)

    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)
    judge_1_output_folder = os.path.join(output_folder_name, "oss-jb-judge")
    judge_2_output_folder = os.path.join(output_folder_name, "ft-roberta-judge")
    judge_3_output_folder = os.path.join(output_folder_name, "harmbench-judge")
    human_label_output_folder = os.path.join(output_folder_name, "human_label")
    if not os.path.exists(judge_1_output_folder):
        os.makedirs(judge_1_output_folder)
    if not os.path.exists(judge_2_output_folder):
        os.makedirs(judge_2_output_folder)
    if not os.path.exists(judge_3_output_folder):
        os.makedirs(judge_3_output_folder)
    if not os.path.exists(human_label_output_folder):
        os.makedirs(human_label_output_folder)

    total_number_of_samples = 0

    for subfolder in os.listdir(folder1):
        subfolder_path1 = os.path.join(folder1, subfolder)
        subfolder_path2 = os.path.join(folder2, subfolder)
        subfolder_path3 = os.path.join(folder3, subfolder)
        combined_df1 = pd.DataFrame()
        combined_df2 = pd.DataFrame()
        combined_df3 = pd.DataFrame()
        if not os.path.isdir(subfolder_path1):
            continue
        for filename in os.listdir(subfolder_path1):
            if not filename.endswith("jbb.parquet"):
                continue
            file_name2 = filename.replace("oss-jb-judge", "ft-roberta-judge")
            file_name3 = filename.replace("oss-jb-judge", "harmbench-judge")
            file_path1 = os.path.join(subfolder_path1, filename)
            file_path2 = os.path.join(subfolder_path2, file_name2)
            file_path3 = os.path.join(subfolder_path3, file_name3)
            if not os.path.exists(file_path2) or not os.path.exists(file_path3):
                print(f"Corresponding files for {file_path1} do not exist, skipping.")
                continue
            df1 = pd.read_parquet(file_path1)
            df2 = pd.read_parquet(file_path2)
            df3 = pd.read_parquet(file_path3)
            if len(df1) != len(df2) or len(df1) != len(df3):
                print(f"Files {file_path1}, {file_path2}, and {file_path3} do not have the same number of rows, skipping.")
                continue
            required_columns = ["goal", "response", "jailbroken"]
            if not all(col in df1.columns for col in required_columns) or not all(col in df2.columns for col in required_columns) or not all(col in df3.columns for col in required_columns):
                print(f"One of the files {file_path1}, {file_path2}, and {file_path3} does not contain the required columns, skipping.")
                continue
            if "variant" in df1.columns and "rep_idx" in df1.columns:
                df1 = df1.sort_values(by=["variant", "rep_idx"]).reset_index(drop=True)
                df2 = df2.sort_values(by=["variant", "rep_idx"]).reset_index(drop=True)
                df3 = df3.sort_values(by=["variant", "rep_idx"]).reset_index(drop=True)
            # combine the required columns and the "evaluator_output" column for each judge into combined_df
            combined_df1 = pd.concat([combined_df1, df1[["goal", "response", "jailbroken", "evaluator_output"]]], ignore_index=True)
            combined_df2 = pd.concat([combined_df2, df2[["goal", "response", "jailbroken", "evaluator_output"]]], ignore_index=True)
            combined_df3 = pd.concat([combined_df3, df3[["goal", "response", "jailbroken", "evaluator_output"]]], ignore_index=True)
        # now we have combined dfs for each subfolder, do sampling based on majority vote
        majority_vote_jailbroken = (combined_df1["jailbroken"] & combined_df2["jailbroken"]) | (combined_df1["jailbroken"] & combined_df3["jailbroken"]) | (combined_df2["jailbroken"] & combined_df3["jailbroken"])
        num_positive_samples = majority_vote_jailbroken.sum()
        num_negative_samples = len(majority_vote_jailbroken) - num_positive_samples
        if num_positive_samples < num_samples_per_technique // 2:
            num_positive_samples_to_sample = num_positive_samples
            num_negative_samples_to_sample = num_samples_per_technique - num_positive_samples_to_sample
        elif num_negative_samples < num_samples_per_technique // 2:
            num_negative_samples_to_sample = num_negative_samples
            num_positive_samples_to_sample = num_samples_per_technique - num_negative_samples_to_sample
        else:
            num_positive_samples_to_sample = num_samples_per_technique // 2
            num_negative_samples_to_sample = num_samples_per_technique // 2

        positive_sample_indices = majority_vote_jailbroken[majority_vote_jailbroken].index
        negative_sample_indices = majority_vote_jailbroken[~majority_vote_jailbroken].index
        sampled_positive_indices = random.sample(list(positive_sample_indices), num_positive_samples_to_sample) if num_positive_samples_to_sample > 0 else []
        sampled_negative_indices = random.sample(list(negative_sample_indices), num_negative_samples_to_sample) if num_negative_samples_to_sample > 0 else []
        sampled_indices = sampled_positive_indices + sampled_negative_indices
        random.shuffle(sampled_indices)
        sampled_df1 = combined_df1.loc[sampled_indices].reset_index(drop=True)
        sampled_df2 = combined_df2.loc[sampled_indices].reset_index(drop=True)
        sampled_df3 = combined_df3.loc[sampled_indices].reset_index(drop=True)
        human_label_df = sampled_df1[["goal", "response"]].copy()
        human_label_df["jailbroken"] = ""
        # save into corresponding output folders
        sampled_df1.to_parquet(os.path.join(judge_1_output_folder, f"{subfolder}__sampled.parquet"), index=False)
        sampled_df2.to_parquet(os.path.join(judge_2_output_folder, f"{subfolder}__sampled.parquet"), index=False)
        sampled_df3.to_parquet(os.path.join(judge_3_output_folder, f"{subfolder}__sampled.parquet"), index=False)
        human_label_df.to_csv(os.path.join(human_label_output_folder, f"{subfolder}__human_label.csv"), index=False)

        total_number_of_samples += len(sampled_indices)
        print(f"Success for {file_path1}, sampled {num_positive_samples_to_sample} positive samples and {num_negative_samples_to_sample} negative samples.")
        print(f"Total number of samples so far: {total_number_of_samples}")

def human_label_csv_to_parquet(human_label_csv_folder):
    for filename in os.listdir(human_label_csv_folder):
        if not filename.endswith(".csv"):
            continue
        csv_path = os.path.join(human_label_csv_folder, filename)
        df = pd.read_csv(csv_path)
        if df["jailbroken"].dtype == "int64":
            df["jailbroken"] = df["jailbroken"].apply(lambda x: True if x == 1 else False)
        parquet_path = os.path.join(human_label_csv_folder, filename.replace(".csv", ".parquet"))
        df.to_parquet(parquet_path, index=False)

if __name__ == "__main__":
    # folder1 = "../../outputs/results/asr_final/oss_jb_judge"
    # folder2 = "../../outputs/results/asr_final/ft_roberta_judge"
    # folder3 = "../../outputs/results/asr_final/harmbench_judge"
    # output_folder_name = "../../outputs/results/human_label_sampling"
    # generate_dataset_for_human_label(folder1, folder2, folder3, output_folder_name)
    human_label_csv_folder = "../../outputs/results/human_label_sampling/human_label-subset"
    human_label_csv_to_parquet(human_label_csv_folder)
