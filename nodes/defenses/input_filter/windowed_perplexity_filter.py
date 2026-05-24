import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..base import DefenseNode
from data.classes.jbb_behaviours import JbbBehavioursDataset
from common.labels import Defense

# code adapted from https://github.com/JailbreakBench/jailbreakbench/blob/main/src/jailbreakbench/defenses/perplexity_filter.py
# code from paper: JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models
# method from paper: Baseline Defenses for Adversarial Attacks Against Aligned Language Models

perplexity_model_path = "meta-llama/Llama-2-7b-chat-hf"
WINDOW_SIZE = 10 # tokens, as per paper

class WindowedPerplexityFilterNode(DefenseNode):

    def __init__(self):
        super().__init__()
        self.ppl_tokenizer = AutoTokenizer.from_pretrained(perplexity_model_path)
        self.ppl_model = AutoModelForCausalLM.from_pretrained(
            perplexity_model_path,
            device_map="auto",
        )
        self.cn_loss = torch.nn.CrossEntropyLoss(reduction="none")
        self.window_size = WINDOW_SIZE
        self.label = Defense.WINDOWED_PERPLEXITY_FILTER

        goals = JbbBehavioursDataset.load_goals()
        self.threshold = self.max_window_log_perplexity(goals)

    def tokenize(self, sequence: str, as_pt: bool = True) -> torch.Tensor:
        """Return the input_ids of the input sequence."""

        if not as_pt:
            return self.ppl_tokenizer.encode(sequence)
        return self.ppl_tokenizer.encode(sequence, return_tensors="pt").cuda()

    def windowed_log_perplexities(self, sequence: str) -> torch.Tensor:
        """Get mean log perplexity for each sliding token window in a sequence."""

        log_probs = self.log_prob(sequence)
        num_tokens = log_probs.numel()
        if num_tokens == 0:
            return torch.tensor([0.0], device=log_probs.device)

        if num_tokens <= self.window_size:
            return log_probs.mean().unsqueeze(0)

        windows = [
            log_probs[start : start + self.window_size].mean()
            for start in range(0, num_tokens, self.window_size)
        ]
        return torch.stack(windows)

    def max_window_log_perplexity(self, sequences: list[str]) -> float:
        """Get maximum windowed log perplexity across all sequences."""

        max_per_sequence = [self.windowed_log_perplexities(s).max().item() for s in sequences]
        return max(max_per_sequence)

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
        prompt_max_window_perplexity = self.windowed_log_perplexities(prompt).max().item()
        if prompt_max_window_perplexity > self.threshold:
            # reject due to high perplexity
            return {
                "rejected": True,
                "rejector": self.label,
                "reject_reason": f"High windowed perplexity detected in prompt: {prompt_max_window_perplexity:.2f}. Threshold: {self.threshold:.2f}",
            }
        else:
            return {}
