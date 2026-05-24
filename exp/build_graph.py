from langgraph.graph import StateGraph, START, END
from IPython.display import Image
import logging

from common.label_to_class import label_to_class
from common.state import JailBreakGraphState
from nodes.llms.base import LLMBase
from nodes.llms.target_model_node import TargetModelNode

def _build_graph_from_node_sequence(node_sequence: list[str | dict], add_rejection_helper: bool = False) -> StateGraph:
    current_node = START
    builder = StateGraph(JailBreakGraphState)

    for node in node_sequence:
        label = node["label"] if type(node) is dict else node
        optional_configs = node["optional_configs"] if type(node) is dict else {}
        node_class = label_to_class(label)
        if issubclass(node_class, LLMBase):
            builder.add_node(label, TargetModelNode(model_label=label, **optional_configs))
        else:
            builder.add_node(label, node_class(**optional_configs))
        builder.add_edge(current_node, label)
        logging.info(f"\nAdded node: {label} with configs: {optional_configs}" if optional_configs else f"\nAdded node: {label}")
        current_node = label
    
    if add_rejection_helper:
        # add rejection check: if rejected (by defense after llm), respond with refusal message, to help evaluation
        def rejection_helper_node(state: JailBreakGraphState) -> JailBreakGraphState:
            if state.get("rejected", False):
                return {"messages": [
                    {"role": "assistant", "content": "I'm sorry, I cannot fulfill that request."}
                ]}
            return {}
        
        builder.add_node("rejection_helper", rejection_helper_node)
        builder.add_edge(current_node, "rejection_helper")
        logging.info(f"\nAdded node: rejection_helper")
        current_node = "rejection_helper"
    
    builder.add_edge(current_node, END)
    graph = builder.compile()
    return graph


def _draw_graph(graph: StateGraph, graph_name: str):
    try:
        Image(graph.get_graph().draw_mermaid_png(output_file_path=f"outputs/graphs/{graph_name}.png"))
    except Exception as e:
        logging.error(f"Failed to draw graph {graph_name}, error: {e}")


def build_attack_graph(a_set: dict) -> StateGraph:
    a_name = a_set["name"]
    logging.info(f"\n--- Building attack graph '{a_name}' ---")

    attack_sequence = a_set["attack_sequence"]
    graph = _build_graph_from_node_sequence(attack_sequence, add_rejection_helper=False)
    _draw_graph(graph, a_name)
    logging.info(f"\n--- Attack graph '{a_name}' complete ---")

    return graph


def build_defense_graph(d_set: dict, add_rejection_helper:bool = True) -> StateGraph:
    d_name = d_set["name"]
    logging.info(f"\n--- Building defense graph '{d_name}' ---")

    defense_sequence_before_llm = d_set["defense_sequence_before_llm"]
    target_llm = d_set["target_llm"]
    defense_sequence_after_llm = d_set["defense_sequence_after_llm"]

    node_sequence = defense_sequence_before_llm + ([target_llm] if target_llm else []) + defense_sequence_after_llm # for batch API exp step 2
    graph = _build_graph_from_node_sequence(node_sequence, add_rejection_helper=add_rejection_helper)
    _draw_graph(graph, d_name)
    logging.info(f"\n--- Defense graph {d_name} complete ---")

    return graph


def build_evaluation_graph(e_set: dict) -> StateGraph:
    e_name = e_set["name"]
    logging.info(f"\n--- Building evaluation graph '{e_name}' ---")

    evaluator_sequence = e_set["evaluator_sequence"]
    graph = _build_graph_from_node_sequence(evaluator_sequence, add_rejection_helper=False)
    _draw_graph(graph, e_name)
    logging.info(f"\n--- Evaluation graph {e_name} complete ---")

    return graph
