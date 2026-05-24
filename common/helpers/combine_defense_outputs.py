import json
import os
import sys
from itertools import combinations
import time
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from common.constants import REJECTION_MESSAGE

DEFAULT_UTILITY_OUTPUT_FOLDER = "../../data/alpaca_eval_model_outputs"
FILTER_DEFENSES = ["PPL_FILTER", "WINDOWED_PPL_FILTER", "PROMPT_GUARD", "WILDGUARD_INPUT", "OSS_SAFEGUARD_20B", "OPENAI_MODERATION_INPUT", "LLAMA_GUARD", "WILDGUARD_OUTPUT", "XGUARD_REASON", "GUARD_REASONER", "OPENAI_MODERATION_OUTPUT"]

def combine_filter_defenses_utility_output(model_name, defense_name_list, utility_output_folder=DEFAULT_UTILITY_OUTPUT_FOLDER, subfolder="combined"):
    generator_name = f"{model_name}__{'-'.join(defense_name_list)}"
    output_filename = f"{generator_name}.json"
    output_folder = os.path.join(utility_output_folder, subfolder)
    os.makedirs(output_folder, exist_ok=True)
    output_filepath = os.path.join(output_folder, output_filename)
    if os.path.exists(output_filepath):
        print(f"Combined defense utility output already exists at {output_filepath}. Skipping.")
        return
    filter_defense_list = []
    for d in defense_name_list:
        if d not in FILTER_DEFENSES:
            print(f"Warning: {d} is not recognized as a filter defense. Skipping.")
            continue
        filter_defense_list.append(d)
    
    filter_defense_utility_output_filepaths = []
    for filter_defense in filter_defense_list:
        filename = f"{model_name}__{filter_defense}.json"
        filepath = os.path.join(utility_output_folder, filename)
        if not os.path.exists(filepath):
            print(f"Error: Utility output file for {filter_defense} not found at {filepath}. Skipping.")
            continue
        filter_defense_utility_output_filepaths.append(filepath)

    if not filter_defense_utility_output_filepaths:
        print("Error: No valid filter defense utility output files found. Exiting.")
        return
    
    print("Combining utility outputs from the following filter defenses:")
    for filepath in filter_defense_utility_output_filepaths:
        print(f" - {filepath}")
    
    outputs = [json.loads(open(filepath, "r").read()) for filepath in filter_defense_utility_output_filepaths]
    combined_outputs = []

    # check all output lengths are the same
    output_lengths = [len(output) for output in outputs]
    if len(set(output_lengths)) != 1:
        print(f"Error: Output lengths are not the same across filter defenses: {output_lengths}. Exiting.")
        return
    
    for i in range(output_lengths[0]):
        combined_data = outputs[0][i].copy()
        combined_data["generator"] = generator_name
        # check if any output rejects
        if any(output[i]["output"] == REJECTION_MESSAGE for output in outputs):
            combined_data["output"] = REJECTION_MESSAGE
        combined_outputs.append(combined_data)
    
    with open(output_filepath, "w") as f:
        json.dump(combined_outputs, f, indent=4)
    print(f"Combined defense utility output saved to {output_filepath}.")


if __name__ == "__main__":
    model_name = "..."
    # filters_to_combine = ["PPL_FILTER", "PROMPT_GUARD", "WILDGUARD_INPUT", "OSS_SAFEGUARD_20B", "LLAMA_GUARD", "WILDGUARD_OUTPUT", "XGUARD_REASON", "GUARD_REASONER"]
    filters_to_combine = ["PPL_FILTER", "PROMPT_GUARD", "WILDGUARD_INPUT", "OSS_SAFEGUARD_20B"]
    # filters_to_combine = ["LLAMA_GUARD", "WILDGUARD_OUTPUT", "XGUARD_REASON", "GUARD_REASONER"]
    config_file_output_path = "../../config_new.yaml"
    combination_number = 2
    for defense_name_list in combinations(filters_to_combine, combination_number):
        combine_filter_defenses_utility_output(model_name, defense_name_list, subfolder=f"combined_{combination_number}")
        with open(config_file_output_path, "a") as f:
            f.write(f'''
  - name: "{model_name}__{'-'.join(defense_name_list)}"
    defense_sequence_before_llm: []
    target_llm: LLM.PLACEHOLDER
    defense_sequence_after_llm: []''')
