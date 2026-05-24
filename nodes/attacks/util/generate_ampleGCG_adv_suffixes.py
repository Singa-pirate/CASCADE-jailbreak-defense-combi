from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import json
from tqdm import tqdm

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))

from data.classes.jbb_behaviours import JbbBehavioursDataset

model_name = "osunlp/AmpleGCG-plus-llama2-sourced-vicuna-7b13b-guanaco-7b13b"
num_beams = 200
device = "cuda:0"

# From paper: AmpleGCG: Learning a Universal and Transferable Generative Model of Adversarial Suffixes for Jailbreaking Both Open and Closed LLMs
class AmpleGCG():
    def __init__(self):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.gen_kwargs = {"pad_token_id":self.tokenizer.pad_token_id, "eos_token_id":self.tokenizer.eos_token_id, "bos_token_id":self.tokenizer.bos_token_id}
        self.gen_config = {"do_sample":False, "max_new_tokens":20, "min_new_tokens":20, "diversity_penalty":1.0, "num_beams":num_beams, "num_beam_groups":1, "num_return_sequences":num_beams}
        self.gen_config = GenerationConfig(**self.gen_kwargs,**self.gen_config)
    
    def generate_adversarial_suffix(self, prompt: str, num_suffixes: int = 5):
        input_ids = self.tokenizer(prompt, return_tensors='pt', padding= True).to(device)
        output = self.model.generate(**input_ids, generation_config = self.gen_config)
        output = output[:,input_ids["input_ids"].shape[-1]:]
        adv_suffixes = self.tokenizer.batch_decode(output,skip_special_tokens= True)
        # NOTE: by right, should try all suffixes generated
        return adv_suffixes[:num_suffixes]
    
if __name__ == "__main__":
    dataset = JbbBehavioursDataset()
    goals = dataset.load_goals()
    ample_gcg = AmpleGCG()
    adv_suffixes_dict = {}
    for goal in tqdm(goals):
        adv_suffixes = ample_gcg.generate_adversarial_suffix(goal, num_suffixes=5)
        adv_suffixes_dict[goal] = adv_suffixes
    with open("ampleGCG_adv_suffixes.json", "w") as f:
        json.dump(adv_suffixes_dict, f, indent=4)
