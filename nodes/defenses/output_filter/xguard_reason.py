import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..base import DefenseNode
from common.labels import Defense
from common.mapper import langchain_messages_to_dict_format


model_id = "Alibaba-AAIG/YuFeng-XGuard-Reason-8B"

# From paper: YuFeng-XGuard: A Reasoning-Centric, Interpretable, and Flexible Guardrail Model for Large Language Models
# Supports custom policy
class XGuardReasonNode(DefenseNode):
    def __init__(self, policy=None, reason_first=True, max_new_tokens=200, confidence_threshold=0.8):
        super().__init__()
        self.label = Defense.XGUARD_REASON
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto").eval()
        self.policy = policy
        self.reason_first = reason_first
        self.max_new_tokens = max_new_tokens
        self.confidence_threshold = confidence_threshold

    def defend_and_update_state(self, messages: list[dict], prompt: str, response: str) -> dict:
        if len(messages) == 0:
            return {} # no change to state
        result = self.infer(messages, prompt)
        risk_score = result['risk_score'] # dict of risk codes to probabilities
        if risk_score == {}:
            return {} # no risk category identified, do not reject
        most_likely_category = max(risk_score, key=risk_score.get)
        if most_likely_category != "Safe-Safe" and risk_score[most_likely_category] >= self.confidence_threshold: # as per paper
            return {
                "rejected": True,
                "rejector": self.label,
                "reject_reason": f"XGuard identified risk category '{most_likely_category}' with confidence {risk_score[most_likely_category]:.2f}",
            }
        else:
            return {} # no change to state
    
    def infer(self, messages, prompt):
        # https://huggingface.co/Alibaba-AAIG/YuFeng-XGuard-Reason-8B

        messages = langchain_messages_to_dict_format(messages)
        if messages[0]['role'] == 'ai' or messages[0]['role'] == 'assistant':
            # to fix special case for batch API results, where user message is missing # TODO fix properly
            messages = [{'role': 'user', 'content': prompt}] + messages

        rendered_query = self.tokenizer.apply_chat_template(messages, policy=self.policy, reason_first=self.reason_first, tokenize=False)
        
        model_inputs = self.tokenizer([rendered_query], return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(**model_inputs, max_new_tokens=self.max_new_tokens, do_sample=False, output_scores=True, return_dict_in_generate=True)

        batch_idx = 0
        input_length = model_inputs['input_ids'].shape[1]

        output_ids = outputs["sequences"].tolist()[batch_idx][input_length:]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        ### parse score ###
        generated_tokens_with_probs = []

        generated_tokens = outputs.sequences[:, input_length:]

        scores = torch.stack(outputs.scores, 1)
        scores = scores.softmax(-1)
        scores_topk_value, scores_topk_index = scores.topk(k=10, dim=-1)

        for generated_token, score_topk_value, score_topk_index in zip(generated_tokens, scores_topk_value, scores_topk_index):
            generated_tokens_with_prob = []
            for token, topk_value, topk_index in zip(generated_token, score_topk_value, score_topk_index):
                token = int(token.cpu())
                if token == self.tokenizer.pad_token_id:
                    continue
                
                res_topk_score = {}
                for ii, (value, index) in enumerate(zip(topk_value, topk_index)):
                    if ii == 0 or value.cpu().numpy() > 1e-4:
                        text = self.tokenizer.decode(index.cpu().numpy())
                        res_topk_score[text] = {
                            "id": str(int(index.cpu().numpy())),
                            "prob": round(float(value.cpu().numpy()), 4),
                        }

                generated_tokens_with_prob.append(res_topk_score)
            
            generated_tokens_with_probs.append(generated_tokens_with_prob)

        score_idx = max(len(generated_tokens_with_probs[batch_idx])-2, 0) if self.reason_first else 0
        id2risk = self.tokenizer.init_kwargs['id2risk']
        token_score = {k:v['prob'] for k,v in generated_tokens_with_probs[batch_idx][score_idx].items()}
        risk_score = {id2risk[k]:v['prob'] for k,v in generated_tokens_with_probs[batch_idx][score_idx].items() if k in id2risk}

        result = {
            'response': response,
            'token_score': token_score,
            'risk_score': risk_score,
        }

        return result
