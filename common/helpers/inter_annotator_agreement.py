import pandas as pd
import os
from sklearn.metrics import cohen_kappa_score

def get_jailbroken_labels(pq_file):
    if not os.path.exists(pq_file):
        raise FileNotFoundError(f"File not found: {pq_file}")
    
    df = pd.read_parquet(pq_file)
    
    if 'jailbroken' not in df.columns:
        raise ValueError("The file must contain a 'jailbroken' column")
    
    if df['jailbroken'].isnull().any():
        raise ValueError("The 'jailbroken' column must not contain null values")
    
    if 'variant' in df.columns and 'rep_idx' in df.columns:
        df = df.sort_values(by=['variant', 'rep_idx']).reset_index(drop=True)
    
    return df['jailbroken'].tolist()

def calculate_cohen_kappa(pq_file_1, pq_file_2):
    # check both files exist
    if not os.path.exists(pq_file_1):
        raise FileNotFoundError(f"File not found: {pq_file_1}")
    if not os.path.exists(pq_file_2):
        raise FileNotFoundError(f"File not found: {pq_file_2}")

    df1 = pd.read_parquet(pq_file_1)
    df2 = pd.read_parquet(pq_file_2)

    # check both files have the same number of rows
    if len(df1) != len(df2):
        raise ValueError("Both files must have the same number of rows")
    
    labels1 = get_jailbroken_labels(pq_file_1)
    labels2 = get_jailbroken_labels(pq_file_2)
    kappa = cohen_kappa_score(labels1, labels2)

    return kappa


def calculate_cohen_kappa_for_final_results(folder1, folder2, output_folder, judge_name_1, judge_name_2):
    os.makedirs(output_folder, exist_ok=True)
    subfolders1 = [f.name for f in os.scandir(folder1) if f.is_dir()]
    subfolders2 = [f.name for f in os.scandir(folder2) if f.is_dir()]
    # for each subfolder of folder1, find the corresponding subfolder in folder2
    # for each pq file ending with 'jbb.parquet' in the subfolder of folder1, the corresponding pq file in subfolder of folder2 should have same name, with 'oss-jb-judge' replaced by 'ft-roberta-judge
    # calculate Cohen's kappa between the 2 pq files
    # in output folder, use a csv file with same name as the subfolders, to save the Cohen's kappa score for each pq file pair
    # then, at the end of the csv file, save the total Cohen's kappa score for all pq file pairs in that subfolder combined (note: not simple average, but calculate Cohen's kappa score using all rows from all pq files combined)
    # then, in a final csv file in the output folder, save total Cohen's kappa score for all data combined
    total_combined_labels1 = []
    total_combined_labels2 = []
    for subfolder in subfolders1:
        if subfolder not in subfolders2:
            print(f"Subfolder {subfolder} not found in {folder2}, skipping...")
            continue
        
        pq_files_1 = [f for f in os.listdir(os.path.join(folder1, subfolder)) if f.endswith('jbb.parquet')]
        pq_files_2 = [f for f in os.listdir(os.path.join(folder2, subfolder)) if f.endswith('jbb.parquet')]
        
        kappa_scores = []

        combined_labels1 = []
        combined_labels2 = []
        
        for pq_file_1 in pq_files_1:
            corresponding_file_2 = pq_file_1.replace(judge_name_1, judge_name_2)
            if corresponding_file_2 not in pq_files_2:
                print(f"Corresponding file {corresponding_file_2} not found in {folder2}/{subfolder}, skipping...")
                continue
            
            pq_file_1_path = os.path.join(folder1, subfolder, pq_file_1)
            pq_file_2_path = os.path.join(folder2, subfolder, corresponding_file_2)
            kappa = calculate_cohen_kappa(pq_file_1_path, pq_file_2_path)
            kappa_scores.append((pq_file_1, kappa))

            combined_labels1.extend(get_jailbroken_labels(pq_file_1_path))
            combined_labels2.extend(get_jailbroken_labels(pq_file_2_path))
        
        # calculate total Cohen's kappa for the combined data
        total_kappa = cohen_kappa_score(combined_labels1, combined_labels2)
        total_combined_labels1.extend(combined_labels1)
        total_combined_labels2.extend(combined_labels2)
        
        # save kappa scores to csv
        output_csv_path = os.path.join(output_folder, f"{subfolder}_kappa_scores.csv")
        with open(output_csv_path, 'w') as f:
            f.write("pq_file,kappa_score\n")
            for pq_file, kappa in kappa_scores:
                f.write(f"{pq_file},{kappa}\n")
            f.write(f"Total,{total_kappa}\n")
    # calculate total Cohen's kappa for all combined data
    overall_kappa = cohen_kappa_score(total_combined_labels1, total_combined_labels2)
    # save overall kappa to a final csv file
    overall_output_csv_path = os.path.join(output_folder, "overall_kappa_score.csv")
    with open(overall_output_csv_path, 'w') as f:
        f.write("overall_kappa_score\n")
        f.write(f"{overall_kappa}\n")


# Should be used within outputs/results/human_label_sampling
def calculate_cohen_kappa_with_human_label(folder1, human_folder2, output_folder):
    os.makedirs(output_folder, exist_ok=True)
        
    judge_pq_files = [f for f in os.listdir(folder1) if f.endswith('parquet')]
    human_pq_files = [f for f in os.listdir(human_folder2) if f.endswith('parquet')]

    total_combined_labels1 = []
    total_combined_labels2 = []
    
    for judge_pq_file in judge_pq_files:
        corresponding_human_file = judge_pq_file.replace('sampled', 'human_label')
        if corresponding_human_file not in human_pq_files:
            print(f"Corresponding human file {corresponding_human_file} not found in {human_folder2}, skipping...")
            continue
        
        judge_pq_path = os.path.join(folder1, judge_pq_file)
        human_pq_path = os.path.join(human_folder2, corresponding_human_file)
        
        kappa = calculate_cohen_kappa(judge_pq_path, human_pq_path)

        total_combined_labels1.extend(get_jailbroken_labels(judge_pq_path))
        total_combined_labels2.extend(get_jailbroken_labels(human_pq_path))
        
        output_txt_path = os.path.join(output_folder, f"{judge_pq_file.replace('.parquet', '')}_kappa_score.txt")
        with open(output_txt_path, 'w') as f:
            f.write(f"{kappa}\n")

    overall_kappa = cohen_kappa_score(total_combined_labels1, total_combined_labels2)
    overall_output_txt_path = os.path.join(output_folder, "overall_kappa_score.txt")
    with open(overall_output_txt_path, 'w') as f:
        f.write(f"{overall_kappa}\n")

def calculate_accurity_with_human_label(folder1, human_folder2, output_folder):
    os.makedirs(output_folder, exist_ok=True)
        
    judge_pq_files = [f for f in os.listdir(folder1) if f.endswith('parquet')]
    human_pq_files = [f for f in os.listdir(human_folder2) if f.endswith('parquet')]

    total_correct = 0
    total_count = 0
    
    for judge_pq_file in judge_pq_files:
        corresponding_human_file = judge_pq_file.replace('sampled', 'human_label')
        if corresponding_human_file not in human_pq_files:
            print(f"Corresponding human file {corresponding_human_file} not found in {human_folder2}, skipping...")
            continue
        
        judge_pq_path = os.path.join(folder1, judge_pq_file)
        human_pq_path = os.path.join(human_folder2, corresponding_human_file)
        
        labels_judge = get_jailbroken_labels(judge_pq_path)
        labels_human = get_jailbroken_labels(human_pq_path)

        correct = sum(1 for j, h in zip(labels_judge, labels_human) if j == h)
        count = len(labels_judge)

        total_correct += correct
        total_count += count

        accuracy = correct / count if count > 0 else 0

        output_txt_path = os.path.join(output_folder, f"{judge_pq_file.replace('.parquet', '')}_accuracy.txt")
        with open(output_txt_path, 'w') as f:
            f.write(f"{accuracy}\n")

    overall_accuracy = total_correct / total_count if total_count > 0 else 0
    overall_output_txt_path = os.path.join(output_folder, "overall_accuracy.txt")
    with open(overall_output_txt_path, 'w') as f:
        f.write(f"{overall_accuracy}\n")

if __name__ == "__main__":
    # folder1 = '../../outputs/results/asr_final/ft_roberta_judge'
    # folder2 = '../../outputs/results/asr_final/harmbench_judge'
    # output_folder = '../../outputs/results/inter_annotator_agreement/roberta_harmbench'
    # judge_name_1 = 'ft-roberta-judge'
    # judge_name_2 = 'harmbench-judge'
    # calculate_cohen_kappa_for_final_results(folder1, folder2, output_folder, judge_name_1, judge_name_2)
    folder1_list = [
        '../../outputs/results/human_label_sampling/oss-jb-judge-subset',
        '../../outputs/results/human_label_sampling/ft-roberta-judge-subset',
        '../../outputs/results/human_label_sampling/harmbench-judge-subset',
        '../../outputs/results/human_label_sampling/shieldlm-judge-subset',
        '../../outputs/results/human_label_sampling/majority_vote-subset'
    ]
        
    human_folder2 = '../../outputs/results/human_label_sampling/human_label-subset'
    output_folder_list = [
        '../../outputs/results/inter_annotator_agreement/human_oss_subset',
        '../../outputs/results/inter_annotator_agreement/human_roberta_subset',
        '../../outputs/results/inter_annotator_agreement/human_harmbench_subset',
        '../../outputs/results/inter_annotator_agreement/human_shieldlm_subset',
        '../../outputs/results/inter_annotator_agreement/human_majority_subset'
    ]
    for folder1, output_folder in zip(folder1_list, output_folder_list):
        calculate_accurity_with_human_label(folder1, human_folder2, output_folder)
