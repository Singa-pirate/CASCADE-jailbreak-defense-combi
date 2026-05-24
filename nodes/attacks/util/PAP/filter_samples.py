import json

def filter_samples(file_path='./sampling_example.json', save_path='./sampling_example_filtered.json'):
    with open(file_path, 'r') as f:
        data = json.load(f)

    filtered_data = {}

    for goal, samples in data.items():
        filtered_data[goal] = []
        for sample in samples:
            sample_ans = list(sample.values())[0]
            reject_strings = ["I cannot", "I can't", "I can\u2019t", "I won't"]
            if not any(sample_ans.startswith(reject_str) for reject_str in reject_strings):
                filtered_data[goal].append(sample)
        print(f"Filtered {len(samples)} samples to {len(filtered_data[goal])} samples.")
    
    with open(save_path, 'w') as f:
        json.dump(filtered_data, f, indent=4)

if __name__ == "__main__":
    filter_samples()