from enum import Enum

# This file defines consistent labels used across the project
# Labels should be (1) descriptive, (2) succinct, (3) close to original name

class LLM(str, Enum):
    VICUNA_7B = "vicuna-7b-v1.5"
    VICUNA_13B = "vicuna-13b-v1.5"
    LLAMA2_7B = "llama2-7b"
    LLAMA2_13B = "llama2-13b"
    OSS_20B = "gpt-oss-20b"
    OSS_120B = "gpt-oss-210b"

    def __str__(self):
        return self.value

class Attack(str, Enum):
    RAW = "raw"
    PREFIX = "affirmative-prefix"
    GCG = "greedy-coordinate-gradient"
    REFUSAL_SUPPRESSION = "refusal-suppression"
    ENCODING_B64 = "base64-encoding"
    ICA = "in-context-attack"

    def __str__(self):
        return self.value
    
class Defense(str, Enum):
    PERPLEXITY_FILTER = "perplexity-filter"
    WINDOWED_PERPLEXITY_FILTER = "windowed-perplexity-filter"
    PROMPT_GUARD = "llama-prompt-guard-2"
    WILDGUARD_INPUT = "wildguard-input-filter"
    OSS_SAFEGUARD_20B = "oss-safeguard-20b-input-filter"
    OPENAI_MODERATION_INPUT = "openai-moderation-input-filter"

    ICD = "in-context-defense"
    ICAR = "in-context-abrupt-refusal"

    LLAMA_GUARD = "llama-guard-3"
    WILDGUARD_OUTPUT = "wildguard-output-filter"
    XGUARD_REASON = "xguard-reasoning-filter"
    GUARD_REASONER = "guard-reasoner-filter"
    OPENAI_MODERATION_OUTPUT = "openai-moderation-output-filter"

    def __str__(self):
        return self.value

class Evaluator(str, Enum):
    KEYWORD_REJECTION = "keyword-rejection-evaluator"
    OSS_JB_JUDGE = "oss-jailbreak-evaluator"
    FT_ROBERTA_JUDGE = "ft-roberta-jailbreak-evaluator"
    HARMBENCH_JUDGE = "harmbench-jailbreak-evaluator"
    SHIELDLM_JUDGE = "shieldlm-jailbreak-evaluator"

    def __str__(self):
        return self.value

class Dataset(str, Enum):
    ADVBENCH = "advbench"
    JBB_BEHAVIOURS = "jbb-behaviours"

    def __str__(self):
        return self.value

class Experiment(str, Enum):
    LINEAR = "linear"
    SINGLE_LOOP = "single-loop"
    TAP = "tree-attack-pruning"
    UTILITY = "utility"

    def __str__(self):
        return self.value
