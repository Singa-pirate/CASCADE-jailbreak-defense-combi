import json

def get_I_GCG_best_adv_suffixes(folder_path: str):
    suffixes = {}
    for i in range(1, 101):
        file_path = f"{folder_path}/result_{i}.json"
        with open(file_path, 'r') as f:
            data = json.load(f)
            prompt = data[0]['user_prompt']

            # find lowest loss
            min_loss = float('inf')
            best_suffix = ""
            for entry in data:
                loss = float(entry['loss'])
                if loss < min_loss:
                    min_loss = loss
                    best_suffix = entry['adv_suffix']
            
            suffixes[prompt] = best_suffix
    
    # save to json
    output_path = "./I_GCG_best_adv_suffixes_vicuna.json"
    with open(output_path, 'w') as f:
        json.dump(suffixes, f, indent=4)
    print(f"Saved best adversarial suffixes to {output_path}")

if __name__ == "__main__":
    folder_path = "./I-GCG_log_vicuna"
    get_I_GCG_best_adv_suffixes(folder_path)
