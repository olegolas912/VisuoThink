import re
import json
import sys
from typing import Callable

from utils_llm import chat_vlm
from dataclasses import dataclass
from utils_misc import tee_stdout, print_message
from copy import deepcopy


VOTE_PROMPT = """
Here are some replies for above instructions, please vote for the best one.

{reply}

Please vote for the best one, first you should explain how you vote, then reply in the json format like: 
```json
{{"best": <index of the best reply>}}
```
"""


@dataclass
class Node:
    response: str
    messages: list
    rollout_result: str = None


class TreeSearch:
    def __init__(self, max_depth: int = 10, temperature: float = 0.8, debug: bool = False, log_file: str = None):
        self.counter = 0 # inner counter for api calls
        self.max_depth = max_depth
        self.temperature = temperature
        self.debug = debug
        self.max_vote_retry = 3
        self.log_file = log_file
        self.vote_template = VOTE_PROMPT
    

    def _chat_vlm(self, prompt, messages):
        """
        Chat with the visual LLM.
        """
        self.counter += 1
        return chat_vlm(prompt, messages, temperature=self.temperature)


    def _output_when_debug(self, message):
        if self.debug:
            message = deepcopy(message)
            message['role'] = 'vote'
            print_message(message)


    def step(self, state, num_samples: int = 3):
        """
        Search the next step, given the current state.
        """
        prompt, state_messages = state
        nodes = []
        for i in range(num_samples):
            model_response, messages = self._chat_vlm(prompt, state_messages)
            nodes.append(Node(response=model_response, messages=messages))
        
        self.current_state = nodes[0]
        return nodes
    

    def text_candidate(self, nodes):
        """
        Organize the candidates in text format.
        """
        replies = [node.response for node in nodes]

        candidate_text = ""
        for i, reply in enumerate(replies):
            candidate_text += f"Reply {i}: \n```\n{reply}\n```\n"
        
        return candidate_text


    def vote(self, nodes):
        """
        Choose the best next action.
        """
        candidate_text = self.text_candidate(nodes)

        context_messages = nodes[0].messages[:-2]
        context_message = nodes[0].messages[-2]
        instruction_text = context_message['content']

        prompt = instruction_text + '\n\n' + self.vote_template.format(reply=candidate_text)
        vote_response, vote_history = self._chat_vlm(prompt, context_messages)

        self._output_when_debug(vote_history[-2])
        self._output_when_debug(vote_history[-1])
        vote_count = 0

        while True:
            try:
                error_message = None

                # Pathaway for Qwen, for it likes to output short and un-enclosed response.
                try:
                    vote_result = json.loads(vote_response)
                    vote_idx = vote_result['best']
                    best_node = nodes[vote_idx]
                    return best_node.response, best_node.messages
                except Exception as e:
                    pass

                pattern = r'```json(.*?)```'
                match = re.search(pattern, vote_response, re.DOTALL)
                if match:
                    vote_response = match.group(1)
                else:
                    error_message = f'Could not find the json pattern in the vote response: {pattern}'
                    raise ValueError
                
                try:
                    vote_result = json.loads(vote_response)
                except Exception as e:
                    error_message = f'Could not parse the vote response as json: {vote_response}'
                    raise ValueError
                
                vote_idx = vote_result['best']
                best_node = nodes[vote_idx]
                return best_node.response, best_node.messages

            except Exception as e:
                if error_message is None:
                    error_message = f'Error in vote response: {e}'
                
                prompt = error_message + '\n\n' + 'Please give me a correct vote response in the json format.'
                vote_response, vote_history = self._chat_vlm(prompt, vote_history)
                self._output_when_debug(vote_history[-2])
                self._output_when_debug(vote_history[-1])
                if vote_count >= self.max_vote_retry:
                    break
                vote_count += 1
        
        return nodes[0].response, nodes[0].messages


    
    def should_terminate(self):
        """
        Check if the search should terminate, the search should terminate when the depth of the reasoning tree is greater than the max depth.
        """
        return len(self.current_state.messages) // 2 >= self.max_depth



VOTE_PROMPT_ROLLOUT = """
Here are some replies and its corresponding rollout results for above instructions, please vote for the best one.
Rollout results: The results you get after several steps of reasoning, MCTS-like rollouts.

{reply}

Please vote for the best one, first you should explain how you vote, then reply in the json format like: 

- You should vote for the reply that leads to the correct answer.
- You should also consider the number of rollouts, the less rollouts, the better.

```json
{{"best": <index of the best reply>}}
```
"""



class TreeSearchWithRollout(TreeSearch):
    """
    Tree search with MCTS-like rollouts.
    """
    def __init__(self, max_depth: int = 10, max_rollout_steps: int = 10, temperature: float = 0.8, debug: bool = False, log_file: str = None, rollout_func: Callable = None):
        super().__init__(max_depth, temperature, debug, log_file)
        self.max_rollout_steps = max_rollout_steps
        self.rollout_count = 0
        self.rollout_func = rollout_func


    def text_candidate(self, nodes):
        """
        Organize the candidates in text format.
        """
        replies = [node.response for node in nodes]
        results = [node.rollout_result for node in nodes]

        candidate_text = ""
        for i, reply in enumerate(replies):
            candidate_text += f"Reply {i}: \n```\n{reply}\n```\n"
            candidate_text += f"Rollout result {i}: \n```\n{results[i]}\n```\n"

        return candidate_text
    

    def _process_rollout_result(self, rollout_result):
        del rollout_result['messages']
        return rollout_result


    def step(self, state, num_samples: int = 3):
        """
        Search the next step, given the current state.
        """
        nodes = super().step(state, num_samples)

        for node in nodes:
            rollout_result = self.rollout(node.messages)
            node.rollout_result = self._process_rollout_result(rollout_result)

        return nodes
    

    def rollout(self, state):
        """
        Rollout the tree search.
        """
        return self.rollout_func(state)



if __name__ == "__main__":
    reponse = r'''
Let me explain how I vote:

Reply 0 provides a clear, structured approach to solving the problem:
1. It properly gathers information and identifies key relationships in the THOUGHT section
2. It correctly breaks down the problem into steps
3. It uses the solve_equation tool appropriately
4. It follows the format requirements by separating THOUGHT and ACTION
5. It provides a final answer in the correct format

Reply 1 incorrectly states that the image is missing, when in fact the image was provided in the problem description.

Based on this analysis, Reply 0 is clearly the better response as it properly addresses the problem and follows all requirements.

```json
{"best": 0}
```
'''

    pattern = r'```json(.*?)```'
    match = re.search(pattern, reponse, re.DOTALL) # 不然不能匹配多行
    if match:
        print(match.group(1))
    else:
        print("No match found")
    import pdb; pdb.set_trace()