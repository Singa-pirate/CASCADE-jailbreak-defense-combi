from nodes.llms.vicuna_7b import Vicuna7B
from nodes.llms.vicuna_13b import Vicuna13B
from nodes.llms.llama2_7b import LlamaTwo7B
from nodes.llms.llama2_13b import LlamaTwo13B
from nodes.llms.llama2_70b import LlamaTwo70B
from nodes.llms.oss_20b import Oss20B
from nodes.llms.oss_120b import Oss120B
from nodes.llms.gemma2_9b import GemmaTwo9B
from nodes.llms.llama3_8b import LlamaThree8B
from nodes.llms.llama3_70b import LlamaThree70B
from nodes.llms.chatgpt_3_point_5_turbo_0125 import ChatGPT3Point5Turbo0125
from nodes.llms.chatgpt_4o import ChatGPT4o
from nodes.llms.chatgpt_5 import ChatGPT5
from nodes.llms.chatgpt_5_point_4_mini import ChatGPT5Point4Mini
from nodes.llms.claude_sonnet_4 import ClaudeSonnet4
from nodes.llms.gemini_2_point_5_flash import Gemini2Point5Flash
from nodes.llms.gemini_3_flash import Gemini3Flash
from nodes.llms.placeholder import PlaceholderLLM

from nodes.attacks.offline.raw import RawPromptNode
from nodes.attacks.offline.prefix import PrefixInjectionNode
from nodes.attacks.offline.gcg import GCGNode
from nodes.attacks.offline.ample_gcg import AmpleGCGNode
from nodes.attacks.offline.refusal_suppression import RefusalSuppressionNode
from nodes.attacks.offline.encoding_b64 import EncodingB64Node
from nodes.attacks.offline.in_context_attack import InContextAttackNode
from nodes.attacks.offline.ignore_previous import IgnorePreviousNode
from nodes.attacks.offline.developer_mode import DeveloperModeNode
from nodes.attacks.offline.dan import DanNode
from nodes.attacks.offline.aim import AimNode
from nodes.attacks.offline.wikipedia import WikipediaNode
from nodes.attacks.offline.repeat_prompt import RepeatPromptNode
from nodes.attacks.offline.flip import FlipAttackNode
from nodes.attacks.offline.code_attack import CodeAttackNode
from nodes.attacks.offline.emoji import EmojiAttackNode
from nodes.attacks.offline.multi_jail import MultiJailNode
from nodes.attacks.offline.sequential_break import SequentialBreakNode
from nodes.attacks.offline.llm_adaptive_attacks import LLMAdaptiveAttacksNode
from nodes.attacks.offline.I_GCG import IGCGNode
from nodes.attacks.offline.dsn import DSNNode

from nodes.attacks.online.tap_attacker import TapAttackerNode
from nodes.attacks.online.renellm_attacker import ReNeLLMAttackerNode
from nodes.attacks.online.gptfuzzer_attacker import GPTFuzzerAttackerNode
from nodes.attacks.online.pap_attacker import PAPAttackerNode

from nodes.defenses.input_filter.perplexity_filter import PerplexityFilterNode
from nodes.defenses.input_filter.windowed_perplexity_filter import WindowedPerplexityFilterNode
from nodes.defenses.input_filter.prompt_guard import PromptGuardNode
from nodes.defenses.input_filter.wildguard_input import WildGuardInputNode
from nodes.defenses.input_filter.oss_safeguard_20b import OssSafeguard20BNode
from nodes.defenses.input_filter.openai_moderation_input import OpenAIModerationInputNode
from nodes.defenses.input_modification.in_context_defense import InContextDefenseNode
from nodes.defenses.input_modification.in_context_defense_abrupt_refusal import InContextDefenseAbruptRefusalNode
from nodes.defenses.input_modification.in_context_defense_abrupt_refusal_random import InContextDefenseAbruptRefusalRandomNode
from nodes.defenses.input_modification.self_reminder import SelfReminderNode
from nodes.defenses.input_modification.smooth_llm import SmoothLLM
from nodes.defenses.input_modification.defensive_prompt_patch import DPPNode
from nodes.defenses.input_modification.prompt_adversarial_tuning import PATNode
from nodes.defenses.input_modification.robust_prompt_optimization import RPONode
from nodes.defenses.output_filter.llama_guard import LlamaGuardNode
from nodes.defenses.output_filter.wildguard_output import WildGuardOutputNode
from nodes.defenses.output_filter.xguard_reason import XGuardReasonNode
from nodes.defenses.output_filter.guard_reasoner import GuardReasonerNode
from nodes.defenses.output_filter.openai_moderation_output import OpenAIModerationOutputNode
from nodes.defenses.util.util_multi_jail_translate_back import MultiJailTranslateBackNode

from nodes.evaluators.keyword_rejection import KeywordRejectionNode
from nodes.evaluators.oss_jb_judge_20b import OssJbJudge20BNode
from nodes.evaluators.oss_jb_judge_120b import OssJbJudge120BNode
from nodes.evaluators.llama_guard_jb_judge import LlamaGuardJbJudge
from nodes.evaluators.gemini_jb_judge import GeminiJbJudge
from nodes.evaluators.ft_roberta_judge import FtRoBERTaJudgeNode
from nodes.evaluators.harmbench_judge import HarmBenchJudgeNode
from nodes.evaluators.shieldlm_judge import ShieldLMJudgeNode

from data.classes.advbench import AdvBenchDataset
from data.classes.advbench_one_entry import AdvBenchOneEntryDataset
from data.classes.jbb_behaviours import JbbBehavioursDataset
from data.classes.harmbench import HarmBenchDataset

def label_to_class(label: str):
    mapping = {
        "LLM.VICUNA_7B": Vicuna7B,
        "LLM.VICUNA_13B": Vicuna13B,
        "LLM.LLAMA2_7B": LlamaTwo7B,
        "LLM.LLAMA2_13B": LlamaTwo13B,
        "LLM.LLAMA2_70B": LlamaTwo70B,
        "LLM.OSS_20B": Oss20B,
        "LLM.OSS_120B": Oss120B,
        "LLM.GEMMA2_9B": GemmaTwo9B,
        "LLM.LLAMA3_8B": LlamaThree8B,
        "LLM.LLAMA3_70B": LlamaThree70B,
        "LLM.CHATGPT_3.5_turbo": ChatGPT3Point5Turbo0125,
        "LLM.CHATGPT_4o": ChatGPT4o,
        "LLM.CHATGPT_5": ChatGPT5,
        "LLM.CHATGPT_5.4_MINI": ChatGPT5Point4Mini,
        "LLM.CLAUDE_SONNET_4": ClaudeSonnet4,
        "LLM.GEMINI_2.5_FLASH": Gemini2Point5Flash,
        "LLM.GEMINI_3_FLASH": Gemini3Flash,
        "LLM.PLACEHOLDER": PlaceholderLLM,

        # Template-based attacks
        'Attack.RAW': RawPromptNode,
        'Attack.IGNORE_PREVIOUS': IgnorePreviousNode,

        'Attack.ENCODING_B64': EncodingB64Node,

        'Attack.ICA': InContextAttackNode,

        'Attack.PREFIX': PrefixInjectionNode,
        'Attack.REFUSAL_SUPPRESSION': RefusalSuppressionNode,
        'Attack.WIKIPEDIA': WikipediaNode,

        'Attack.DEVELOPER_MODE': DeveloperModeNode,
        'Attack.DAN': DanNode,
        'Attack.AIM': AimNode,

        'Attack.FLIP': FlipAttackNode,
        'Attack.CODE_ATTACK': CodeAttackNode,
        'Attack.SEQUENTIAL_BREAK': SequentialBreakNode,

        'Attack.EMOJI': EmojiAttackNode,

        'Attack.MULTI_JAIL': MultiJailNode,

        # LLM-based attacks
        'Attack.TAP_ATTACKER': TapAttackerNode,
        'Attack.ReNeLLM_ATTACKER': ReNeLLMAttackerNode,
        'Attack.GPTFuzzer_ATTACKER': GPTFuzzerAttackerNode,
        'Attack.PAP_ATTACKER': PAPAttackerNode,

        # Optim-based attacks
        'Attack.GCG': GCGNode,
        'Attack.AmpleGCG': AmpleGCGNode,
        'Attack.REPEAT_PROMPT': RepeatPromptNode,
        'Attack.LLM_ADAPTIVE_ATTACKS': LLMAdaptiveAttacksNode,
        'Attack.I_GCG': IGCGNode,
        'Attack.DSN': DSNNode,
        
        # Input filters
        "Defense.PPL_FILTER": PerplexityFilterNode,
        "Defense.WINDOWED_PPL_FILTER": WindowedPerplexityFilterNode,
        "Defense.PROMPT_GUARD": PromptGuardNode,
        "Defense.WILDGUARD_INPUT": WildGuardInputNode,
        "Defense.OSS_SAFEGUARD_20B": OssSafeguard20BNode,
        "Defense.OPENAI_MODERATION_INPUT": OpenAIModerationInputNode,

        # Input modification
        "Defense.SELF_REMINDER": SelfReminderNode,
        "Defense.SMOOTH_LLM": SmoothLLM,
        "Defense.ICD": InContextDefenseNode,
        "Defense.ICD_AR": InContextDefenseAbruptRefusalNode,
        "Defense.ICD_ARR": InContextDefenseAbruptRefusalRandomNode,
        "Defense.DPP": DPPNode,
        "Defense.PAT": PATNode,
        "Defense.RPO": RPONode,

        # Output filters
        "Defense.LLAMA_GUARD": LlamaGuardNode,
        "Defense.WILDGUARD_OUTPUT": WildGuardOutputNode,
        "Defense.XGUARD_REASON": XGuardReasonNode,
        "Defense.GUARD_REASONER": GuardReasonerNode,
        "Defense.OPENAI_MODERATION_OUTPUT": OpenAIModerationOutputNode,

        "Defense.MULTI_JAIL_TRANSLATE_BACK": MultiJailTranslateBackNode,

        "Evaluator.KEYWORD_REJECTION": KeywordRejectionNode,
        "Evaluator.OSS_JB_JUDGE_20B": OssJbJudge20BNode,
        "Evaluator.OSS_JB_JUDGE_120B": OssJbJudge120BNode,
        "Evaluator.LLAMA_GUARD_JB_JUDGE": LlamaGuardJbJudge,
        "Evaluator.GEMINI_JB_JUDGE": GeminiJbJudge,
        "Evaluator.FT_ROBERTA_JUDGE": FtRoBERTaJudgeNode,
        "Evaluator.HARMBENCH_JUDGE": HarmBenchJudgeNode,
        "Evaluator.SHIELDLM_JUDGE": ShieldLMJudgeNode,

        "Dataset.ADVBENCH": AdvBenchDataset,
        "Dataset.ADVBENCH_ONE_ENTRY": AdvBenchOneEntryDataset,
        "Dataset.JBB_BEHAVIOURS": JbbBehavioursDataset,
        "Dataset.HARMBENCH": HarmBenchDataset,
    }
    return mapping.get(label, None)
