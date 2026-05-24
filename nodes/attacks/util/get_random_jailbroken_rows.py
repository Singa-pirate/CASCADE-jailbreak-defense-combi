import pandas as pd

def get_random_jailbroken_rows(parquet_path, n=20, random_state=42):
    df = pd.read_parquet(parquet_path).head(100)
    # jailbroken & response length > 30
    jailbroken_df = df[(df['jailbroken'] == True) & (df['response'].str.len() > 30)]
    
    if len(jailbroken_df) < n:
        print(f"Warning: Only {len(jailbroken_df)} jailbroken rows available after filtering, returning all of them.")
        return jailbroken_df
    
    return jailbroken_df.sample(n=n, random_state=random_state)

if __name__ == "__main__":
    file = "outputs/results/20251019-1-collect-data-for-jailbroken-and-defended-samples/prefix__vicuna-7b__oss-jb-judge_jbb.parquet"
    results = get_random_jailbroken_rows(file, n=25)
    # after manual review, remove rows that are not actually jailbroken
    # results = results.drop(index=results.index[remove_rows])
    results.to_parquet("nodes/attacks/util/jailbroken_samples.parquet")
