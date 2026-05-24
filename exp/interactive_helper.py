from pprint import pprint
from langgraph.graph import StateGraph
import logging

from common.state import JailBreakGraphState

def greet():
    logging.info("\nInteractive experiment setup complete, you can now enter your prompts.\nType 'quit', 'exit', or 'q' to exit.")

def wait_input():
    user_input = input("\nEnter your prompt:\n")
    if user_input.lower() in ["quit", "exit", "q"]:
        logging.info("Goodbye!")
        return None
    return user_input

def stream_graph_updates(graph: StateGraph, init_state: JailBreakGraphState, print_final_state=False) -> JailBreakGraphState:
    for mode, chunk in graph.stream(init_state, stream_mode=["updates", "values"]):
        # mode is either "update" or "value" string
        if mode == "updates":
            # for "update" mode, chunk is a dict of node_name -> state
            for name, value in chunk.items():
                # for attack node, print changed prompt
                if value and "prompt" in value and value["prompt"]:
                    logging.info(f"\nAttack node '{name}' changed prompt to:")
                    logging.info(f"\n{value['prompt']}\n")
                
                # for defense node, print new assistant message
                if value and len(value.get("messages", [])) > 0:
                    last_message = value["messages"][-1]
                    if last_message and last_message.get("role", "") == "assistant":
                        logging.info("Assistant: " + value["messages"][-1]["content"])
                
                # for defense node, print new rejection
                if value and value.get("rejected", False):
                    logging.info("\n=== REJECTED ===")
                    logging.info(f"{value.get('rejector', 'Unknown rejector')}: {value.get('reject_reason', 'unknown reason')}\n")
        else:
            # for "value" mode, chunk is the full state
            state = chunk
    
    if print_final_state:
        logging.info("\n--- Final state ---")
        del state["messages"]  # remove messages for cleaner print
        del state["messageHistoryForTargetModel"]
        pprint(state)
    
    return state
