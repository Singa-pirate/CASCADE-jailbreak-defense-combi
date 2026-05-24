import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from ..base import DefenseNode
from data.classes.jbb_behaviours import JbbBehavioursDataset
from common.labels import Defense

# code adapted from https://github.com/JailbreakBench/jailbreakbench/blob/main/src/jailbreakbench/defenses/perplexity_filter.py
# code from paper: JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models
# method from paper: Baseline Defenses for Adversarial Attacks Against Aligned Language Models

perplexity_model_path = "meta-llama/Llama-2-7b-chat-hf"

class PerplexityFilterNode(DefenseNode):

    def __init__(self):
        super().__init__()
        self.ppl_tokenizer = AutoTokenizer.from_pretrained(perplexity_model_path)
        self.ppl_model = AutoModelForCausalLM.from_pretrained(
            perplexity_model_path,
            device_map="auto",
        )
        self.cn_loss = torch.nn.CrossEntropyLoss(reduction="none")
        self.label = Defense.PERPLEXITY_FILTER

        goals = JbbBehavioursDataset.load_goals()
        self.threshold = self.max_log_perplexity(goals)

    def tokenize(self, sequence: str, as_pt: bool = True) -> torch.Tensor:
        """Return the input_ids of the input sequence."""

        if not as_pt:
            return self.ppl_tokenizer.encode(sequence)
        return self.ppl_tokenizer.encode(sequence, return_tensors="pt").cuda()

    def log_perplexity(self, sequence: str) -> float:
        """Get the log perplexity of a sequence."""

        input_ids = self.tokenize(sequence)
        with torch.no_grad():
            loss = self.ppl_model(input_ids, labels=input_ids).loss
        return loss.item()

    def max_log_perplexity(self, sequences: list[str]) -> float:
        """Get maximum log perplexity of a sequence."""

        return max([self.log_perplexity(s) for s in sequences])

    def log_prob(self, sequence: str):
        """Get the log probabilities of the token."""

        input_ids = self.tokenize(sequence)
        with torch.no_grad():
            logits = self.ppl_model(input_ids, labels=input_ids).logits
        logits = logits[:, :-1, :].contiguous()
        input_ids = input_ids[:, 1:].contiguous()
        log_probs = self.cn_loss(logits.view(-1, logits.size(-1)), input_ids.view(-1))
        return log_probs


    def defend_and_update_state(self, messages: list[dict], prompt: str, response: str) -> dict:
        log_probs = self.log_prob(prompt)
        mean_perplexity = log_probs.mean()
        if mean_perplexity > self.threshold:
            # reject due to high perplexity
            return {
                "rejected": True,
                "rejector": self.label,
                "reject_reason": f"High perplexity detected in prompt: {mean_perplexity:.2f}. Threshold: {self.threshold:.2f}",
            }
        else:
            return {}
