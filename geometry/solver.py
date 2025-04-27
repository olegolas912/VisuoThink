import json
import os
import argparse, shutil

from agent import GeoProUserAgent
from prompt import GeoPromptVisuoThink
from parse import Parser
from execution import CodeExecutor
from contextlib import redirect_stdout
from utils_misc import tee_stdout, print_message
from utils_llm import chat_vlm
from tqdm import tqdm
from copy import deepcopy

# the max reasoning steps / tree search depth
from config import MAX_REPLY


def aux_step(task_type: str) -> bool:
    return True if task_type in ['geovar2', 'visuothink'] else False
    

def run_geo_task(task_input: str, output_dir: str, task_type: str, verbose: bool = False, rollout_search: bool = False, tree_span: int = 3, search: bool = False):
    """
    Run a task and return the result.

    - task: the task to run, a directory path.
    """
    assert task_type in ["visuothink"]

    # create a directory for the task
    task_input = task_input.rstrip('/')
    task_directory = os.path.join(output_dir, os.path.basename(task_input))

    # copy the task input to the output directory
    shutil.rmtree(task_directory, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    shutil.copytree(task_input, task_directory, dirs_exist_ok=True)
    log_file_path = os.path.join(task_directory, 'output.log')

    
    with tee_stdout(log_file_path):
        if task_type == 'visuothink':
            query = json.load(open(os.path.join(task_input, "ex.json")))

            # load the images
            query['image_path_code'] = os.path.join(output_dir, query['image_path_code'])
            print(query['image_path_code'])
            images = []
            prompt_generator = GeoPromptVisuoThink()
            parser = Parser()
            executor = CodeExecutor(working_dir=task_directory)
        
        # agent setup
        agent = GeoProUserAgent(
            prompt_generator = prompt_generator,
            parser = parser,
            executor = executor,
            step_aux = aux_step(task_type)
        )
        init_message = agent.initiate_chat(query)
        messages = []

        for i in range(MAX_REPLY):
            model_response, messages = chat_vlm(init_message, messages)

            if verbose:
                print_message(messages[-2])
                print_message(messages[-1])

            reply = agent.receive(model_response)
            if reply is None:
                break
            init_message = reply

        # turn off server
        agent.executor.cleanup()

        # save the results
        with open(os.path.join(task_directory, "output.json"), "w") as f:
            json.dump(messages, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    TASK_DIR = "dataset/geometry/Dataset_GeomVerse/test_geomverse_TEST_D2_B100_data_1"
    OUTPUT_DIR = "outputs/geometry/Dataset_GeomVerse/test_geomverse_TEST_D2_B100_data_1"
    run_geo_task(TASK_DIR, OUTPUT_DIR, task_type="visuothink", verbose=True)
