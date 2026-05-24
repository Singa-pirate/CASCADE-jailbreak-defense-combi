from common.label_to_class import label_to_class
from common.state import JailBreakGraphState
from common.constants import DEFAULT_SEED
from nodes.llms.placeholder import PlaceholderLLM

# represents target LLM node in the graph
class TargetModelNode:
    def __init__(self, model_label, temperature=None, seed=None, **kwargs):
        model_class = label_to_class(model_label)
        self.model = model_class(**kwargs)
        # model temperature defaults to 1.0; can set custom temperature via optional configs
        if temperature is not None:
            if type(temperature) in [int, float] and 0 <= temperature <= 2 and hasattr(self.model, "temperature"):
                self.model.temperature = temperature
        if seed is not None:
            if type(seed) == int and hasattr(self.model, "seed"):
                self.model.seed = seed
        if temperature == 0 and seed is None:
            # add missing seed
            if hasattr(self.model, "seed"):
                self.model.seed = DEFAULT_SEED
    
    def __call__(self, state: JailBreakGraphState) -> JailBreakGraphState:
        if state["rejected"]:
            return {}  # No-op if already rejected
        
        messages = state["messages"].copy()
        messageHistory = state["messageHistoryForTargetModel"].copy()

        # For a fresh start, log system prompt to messages
        if len(messages) == 0 and self.model.system_prompt is not None:
            messages.append({"role": "system", "content": self.model.system_prompt})

        messages.append({"role": "user", "content": state["prompt"]})
        messageHistory.append({"role": "user", "content": state["prompt"]})

        if isinstance(self.model, PlaceholderLLM):
            return {
                "messages": messages,
                "messageHistoryForTargetModel": messageHistory,
            }

        response, n_input_tokens, n_output_tokens, max_output_tokens_reached = self.model.run_inference(messageHistory, count_tokens=True)
        new_total_input_tokens = state.get("total_input_tokens", 0) + n_input_tokens
        new_total_output_tokens = state.get("total_output_tokens", 0) + n_output_tokens

        messages.append({"role": "assistant", "content": response})
        # do not add to history yet, wait for evaluator to decide if it's rejected
        # handle history elsewhere (e.g. in evaluator node / in experiment loop)

        return {
            "messages": messages,
            "messageHistoryForTargetModel": messageHistory,
            "n_input_tokens": n_input_tokens,
            "n_output_tokens": n_output_tokens,
            "total_input_tokens": new_total_input_tokens,
            "total_output_tokens": new_total_output_tokens,
            "max_output_tokens_reached": max_output_tokens_reached,
            }
