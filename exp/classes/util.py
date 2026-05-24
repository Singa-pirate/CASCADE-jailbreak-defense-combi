import pandas as pd

from common.state import JailBreakGraphState
from common.mapper import langchain_messages_to_dict_format

def create_empty_df():
    return pd.DataFrame(columns=[
        "goal", "prompt", "response",
        "jailbroken", "evaluator_output",
        "rejected", "rejector", "reject_reason",
        "total_input_tokens", "total_output_tokens", "max_output_tokens_reached",
        "iteration", "target_LLM_query_count", "score", "target",
        "messages","messageHistoryForAttacker",
        "goal_id", "repetition"
    ])

# Save a list of JailBreakGraphState to a parquet file
def save_states(file_path: str, states_after_AnD: list[JailBreakGraphState]):
    df = create_empty_df()
    for state in states_after_AnD:
        state["messages"] = langchain_messages_to_dict_format(state.get("messages", [])) # for consistency
        df = pd.concat([df, pd.DataFrame([{
            "goal": state.get("goal", ""),
            "prompt": state.get("prompt", ""),
            "response": state["messages"][-1]["content"] if len(state.get("messages", [])) > 0 else "",
            "jailbroken": state.get("jailbroken", False),
            "evaluator_output": state.get("evaluator_output", ""),
            "rejected": state.get("rejected", False),
            "rejector": state.get("rejector", ""),
            "reject_reason": state.get("reject_reason", ""),
            "total_input_tokens": state.get("total_input_tokens", 0),
            "total_output_tokens": state.get("total_output_tokens", 0),
            "max_output_tokens_reached": state.get("max_output_tokens_reached", False),
            "iteration": state.get("iteration", None),
            "target_LLM_query_count": state.get("target_LLM_query_count", None),
            "score": state.get("score", None),
            "target": state.get("target", None),
            "messages": str(state.get("messages", [])),
            # optional, only for online attacks
            "messageHistoryForAttacker": str(state.get("messageHistoryForAttacker", [])),
            "goal_id": state.get("goal_id", None),
            "repetition": state.get("repetition", None),
        }])], ignore_index=True)
    df.to_parquet(file_path)
    print(f"\n--- States saved to {file_path} ---", flush=True)

# Load a list of JailBreakGraphState from a parquet file
def load_states(file_path: str) -> list[JailBreakGraphState]:
    df = pd.read_parquet(file_path)
    states = []
    for _, row in df.iterrows():
        state = JailBreakGraphState(
            goal=row.get("goal", ""),
            prompt=row.get("prompt", ""),
            messages=eval(row.get("messages", "[]")),
            rejected=row.get("rejected", False),
            rejector=row.get("rejector", ""),
            reject_reason=row.get("reject_reason", ""),
            total_input_tokens=row.get("total_input_tokens", 0),
            total_output_tokens=row.get("total_output_tokens", 0),
            max_output_tokens_reached=row.get("max_output_tokens_reached", False),
            iteration=row.get("iteration", None),
            target_LLM_query_count=row.get("target_LLM_query_count", None),
            score=row.get("score", None),
            target=row.get("target", None),
            messageHistoryForAttacker=eval(row.get("messageHistoryForAttacker", "[]")),
            goal_id=row.get("goal_id", None),
            repetition=row.get("repetition", None)
        )
        states.append(state)
    print(f"\n--- Loaded states from {file_path} ---", flush=True)
    return states
