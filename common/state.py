from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class JailBreakGraphState(TypedDict):
    ### Attack states
    goal: str
    prompt: str

    ### Defense states
    # messages: record of all messages among system, user, assistant
    messages: Annotated[list, add_messages] # note the reducer function will create specific Message types based on role
    # messageHistoryForTargetModel: messages that target model can access
    messageHistoryForTargetModel: list[dict] # list of dicts containing role and content
    rejected: bool
    rejector: str
    reject_reason: str
    jailbroken: bool # init to false, set to true if an evaluator decides it's jailbroken
    evaluator_output: str

    # Fields for online attacks
    iteration: int # current iteration/depth of the attack
    target_LLM_query_count: int
    score: int # jailbroken score to help pruning, 1-10
    target: str # affirmative string that attacker aims to make target LLM produce
    # messageHistoryForAttacker: messages that attacker LLM can access
    messageHistoryForAttacker: list[dict] # list of dicts containing role and content

    # Other misc fields for logging & troubleshooting
    n_input_tokens: int # all token, including system prompt etc. for one round
    n_output_tokens: int # all token, including thinking steps etc. for one round
    total_input_tokens: int # accumulated over multiple rounds
    total_output_tokens: int # accumulated over multiple rounds
    max_output_tokens_reached: bool # if many rounds, effectively only for last round

    # Fields for tracking in batch API experiments
    goal_id: int
    repetition: int
