import json
import os
import argparse, shutil
import re

from agent import VisualNavigationUserAgent

try:
    from prompt import VisualNavigationPrompt, VisualNavigationCoT, VisualNavigationVoT
    from execution import VisiualNavigationExecutor
except:
    print("Visual Navigation is not supported.")

try:
    from prompt import VisualTilingPrompt
    from execution import VisiualTilingExecutor
except:
    print("Visual Tiling is not supported.")

from utils_misc import print_message, tee_stdout, print_error
from tree_search import TreeSearchWithRollout
from utils_llm import chat_vlm
from config import MAX_REPLY
from copy import deepcopy
from tqdm import tqdm
from pprint import pprint


def aux_step(task_type: str) -> bool:
    return True if task_type in ['geovar2'] else False


def run_task_cot(task_input: str, output_dir: str, task_type: str, verbose: bool = False, run_tag: str = 'cot'):
    """
    Run a task and return the result.

    - task: the task to run, a directory path.
    """
    assert task_type in ["visual-navigation", "visual-tiling"]
    assert run_tag in ['cot', 'vot']

    # create a directory for the task
    task_input = task_input.rstrip('/')
    task_directory = os.path.join(output_dir, os.path.basename(task_input))
    
    # copy the task input to the output directory
    shutil.rmtree(task_directory, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    shutil.copytree(task_input, task_directory, dirs_exist_ok=True)
    log_file_path = os.path.join(task_directory, 'output.log') 

    with tee_stdout(log_file_path):
        if task_type == 'visual-navigation':
            configuration = json.load(open(os.path.join(task_input, "solution.json")))
            if run_tag == 'cot':
                prompt_generator = VisualNavigationCoT(task_input)
            elif run_tag == 'vot':
                prompt_generator = VisualNavigationVoT(task_input)
            else:
                raise ValueError(f"Invalid run_tag: {run_tag}")
        
        else:
            raise ValueError(f"Invalid task_type: {task_type}")
    
        prompt = prompt_generator.prompt
        executor = VisiualNavigationExecutor(configuration, task_directory)
        response, messages = chat_vlm(prompt)

        pattern = "(```json.*?```)"
        # find all json strings in the response
        json_strs = re.findall(pattern, response, re.DOTALL)
        for json_str in json_strs:
            feedback = executor.execute(json_str)
            pprint(feedback)
            messages.append({"role": "user", "content": feedback['message']})
        
        # save messages
        with open(os.path.join(task_directory, "output.json"), "w") as f:
            json.dump(messages, f, indent=4, ensure_ascii=False)



def run_task(task_input: str, output_dir: str, task_type: str, verbose: bool = False, visual: bool = False, tree_span=3, tree_search=False):
    """
    Run a task and return the result.

    - task: the task to run, a directory path.
    """
    assert task_type in ["visual-navigation", "visual-tiling"]

    # create a directory for the task
    task_input = task_input.rstrip('/')
    task_directory = os.path.join(output_dir, os.path.basename(task_input))
    if os.path.exists(os.path.join(task_directory, 'output.json')):
        print_error(f'"{task_directory}" already inferred before, skip.')
        return
    
    # copy the task input to the output directory
    shutil.rmtree(task_directory, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    shutil.copytree(task_input, task_directory, dirs_exist_ok=True)
    log_file_path = os.path.join(task_directory, 'output.log') 
    
    with tee_stdout(log_file_path):
        if task_type == 'visual-navigation':
            # configuration
            configuration = json.load(open(os.path.join(task_input, "solution.json")))
            prompt_generator = VisualNavigationPrompt()
            executor = VisiualNavigationExecutor(configuration, task_directory, visual=visual)
        
        elif task_type == 'visual-tiling':
            configuration = json.load(open(os.path.join(task_input, "problem.json")))
            prompt_generator = VisualTilingPrompt()
            executor = VisiualTilingExecutor(configuration, task_directory, visual=visual)
        
        # agent setup
        agent = VisualNavigationUserAgent(
            prompt_generator = prompt_generator,
            executor = executor,
        )
        init_message = agent.initiate_chat(task_directory)
        messages = []

        def rollout_func(state):
            model_response = state[-1]['content']
            messages = state.copy()
            init_messages = deepcopy(messages)
            reply = None
            terminator = False
            save_state = executor.save_state()
            for i in tqdm(range(MAX_REPLY - (len(messages) // 2)), desc='Rolling Out'):
                reply_dict = agent.receive(model_response)
                reply = reply_dict['content']
                terminator = reply_dict['terminator']
                if terminator:
                    messages.append({"role": "user", "content": reply})
                    break
                model_response, messages = chat_vlm(reply, messages)
            
            exe_state = executor._visualize()
            exe_state_message = f"\nThe current map is <img src='{exe_state}'>"
            
            num_rollouts = (len(messages) - len(init_messages) + 1) // 2
            executor.load_state(save_state)

            pprint(messages[len(init_messages):])

            if not terminator:
                return {'status': False, 'content': 'You have not reached the destination.' + exe_state_message, 'num_rollouts': num_rollouts, 'messages': messages}
            else:
                last_turn_response = messages[-1]['content']
                return {'status': True, 'content': last_turn_response + exe_state_message, 'num_rollouts': num_rollouts, 'messages': messages}

        if tree_search:
            tree_search = TreeSearchWithRollout(max_depth=MAX_REPLY, debug=True if verbose else False, rollout_func=rollout_func)
    
        for i in range(MAX_REPLY):
            if tree_search:
                nodes = tree_search.step((init_message, messages), tree_span)
                model_response, messages = tree_search.vote(nodes)
            else:
                model_response, messages = chat_vlm(init_message, messages)

            if verbose:
                print_message(messages[-2])
                print_message(messages[-1])

            # agent
            reply_dict = agent.receive(model_response)
            reply = reply_dict['content']
            terminator = reply_dict['terminator']
            if terminator:
                messages.append({"role": "user", "content": reply})
                print(messages[-1])
                break
            init_message = reply

        # save the results
        with open(os.path.join(task_directory, "output.json"), "w") as f:
            json.dump(messages, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    TASK_DIR = "visualization-of-thought/dataset/visual-navigation/configurations/level-3/1"
    OUTPUT_DIR = "workspace/outputs/visual-navigation/visual-tree_rs/level-3"
    # run_task(TASK_DIR, OUTPUT_DIR, "visual-navigation", verbose=True, visual=True, tree_search=True)

    # run_task_cot(TASK_DIR, OUTPUT_DIR, "visual-navigation", verbose=True, run_tag='cot')
    run_task_cot(TASK_DIR, OUTPUT_DIR, "visual-navigation", verbose=True, run_tag='vot')
    # TASK_DIR = "visualization-of-thought/dataset/visual-tiling/configurations/level-2/4"
    # OUTPUT_DIR = "workspace/outputs/visual-tiling/without-visual/level-2"
    # run_task(TASK_DIR, OUTPUT_DIR, "visual-tiling", verbose=True, visual=True, tree_search=True)