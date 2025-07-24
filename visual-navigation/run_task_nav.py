import os
import sys
import glob
import argparse
from contextlib import redirect_stdout

from navigator import run_task, run_task_cot
from tqdm import tqdm
from config import llm_config, MAX_REPLY
from utils_misc import print_error
def llm_config_model_name(llm_config):
    name = llm_config['config_list'][0]['model']
    if '/' in name:
        return name.split('/')[-1]
    return name


def run_agent(tasks_path="dataset/visual-navigation/configurations/level-3", task_type='visual-navigation', visual=False, tree_search=False, visual_tag=None, maxtry=10, verbose=False):
    print(tasks_path)
    all_task_instances = glob.glob(f"{tasks_path}/*")

    assert len(all_task_instances) > 0, f"No task instances found in {tasks_path}, please run the script under the `VisuoThink`."

    assert visual_tag in ['cot', 'vot', None]
    if visual_tag in ['cot', 'vot']:
        pass
    else:
        visual_tag = "visual" if visual else "without-visual"
        visual_tag = (visual_tag + '-tree_rs') if tree_search else visual_tag

    if maxtry != 10:
        visual_tag = visual_tag + f'-maxtry{maxtry}'
        
    group_tag = os.path.basename(tasks_path.rstrip('/'))

    for task_instance in tqdm(all_task_instances):
        print_error(f"outputs/{llm_config_model_name(llm_config)}/{task_type}/{group_tag}/{visual_tag} {task_instance}")
        try:
            if visual_tag in ['cot', 'vot']:
                run_task_cot(task_instance, f"outputs/{llm_config_model_name(llm_config)}/{task_type}/{group_tag}/{visual_tag}", task_type=task_type, verbose=verbose, run_tag=visual_tag)
            else:
                run_task(task_instance, f"outputs/{llm_config_model_name(llm_config)}/{task_type}/{group_tag}/{visual_tag}", task_type=task_type, verbose=verbose, visual=visual, tree_search=tree_search)
        except Exception as err:
            raise err
            print(err)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--tree_search", action="store_true")
    parser.add_argument("--tasks_path", type=str, default="dataset/visual-navigation/configurations/level-3")
    parser.add_argument("--task_type", type=str, default='visual-navigation')
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--maxtry", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_agent(visual=args.visual, tasks_path=args.tasks_path, task_type=args.task_type, tree_search=args.tree_search, visual_tag=args.run_tag, maxtry=args.maxtry, verbose=args.verbose)    
