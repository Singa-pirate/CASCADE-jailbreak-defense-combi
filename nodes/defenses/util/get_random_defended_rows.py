import pandas as pd

def get_random_defended_rows(parquet_path, n=20, random_state=42):
    df = pd.read_parquet(parquet_path).head(50)
    # not jailbroken & response length > 300 & response starts with "I'm sorry"
    defended_df = df[~df["jailbroken"] & (df["response"].str.len() > 300) & df["response"].str.startswith("I'm sorry")]
    
    if len(defended_df) < n:
        print(f"Warning: Only {len(defended_df)} defended rows available, returning all of them.")
        return defended_df
    return defended_df.sample(n=n, random_state=random_state)

if __name__ == "__main__":
    file = "outputs/results/asr/20260316-1-icd-regen-advbench-defense-examples/nothing__VICUNA_7B__oss-jb-judge_advbench-short_LLM_outputs.parquet"
    results = get_random_defended_rows(file)
    results.to_parquet("nodes/defenses/util/defended_samples.parquet")
