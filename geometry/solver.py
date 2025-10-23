import json
import os
import sys
import argparse, shutil
from pathlib import Path

from agent import GeoProUserAgent
from prompt import GeoPromptVisuoThink
from parse import Parser
from execution import CodeExecutor
from contextlib import redirect_stdout
from utils_misc import tee_stdout, print_message
from utils_llm import chat_vlm
from tqdm import tqdm
from copy import deepcopy

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _PROJECT_ROOT / "visual-navigation"
if str(_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(_CONFIG_DIR))

# the max reasoning steps / tree search depth
from config import MAX_REPLY  # noqa: E402


def aux_step(task_type: str) -> bool:
    return True if task_type in ['geovar2', 'visuothink'] else False
    

def run_geo_task(task_input: str, output_dir: str, task_type: str, verbose: bool = False, rollout_search: bool = False, tree_span: int = 3, search: bool = False):
    """
    Run a task and return the result.

    - task: the task to run, a directory path.
    """
    print("\n" + "="*80)
    print("[>>] GEOMETRY SOLVER STARTING")
    print("="*80)
    
    assert task_type in ["visuothink"]

    # create a directory for the task
    task_input = task_input.rstrip('/')
    task_directory = os.path.join(output_dir, os.path.basename(task_input))
    
    print(f"\n[TASK SETUP]")
    print(f"   Input directory:  {task_input}")
    print(f"   Output directory: {task_directory}")
    print(f"   Task type:        {task_type}")
    print(f"   Verbose mode:     {verbose}")

    # copy the task input to the output directory
    shutil.rmtree(task_directory, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    shutil.copytree(task_input, task_directory, dirs_exist_ok=True)
    log_file_path = os.path.join(task_directory, 'output.log')
    print(f"   Log file:         {log_file_path}")

    
    with tee_stdout(log_file_path):
        if task_type == 'visuothink':
            print(f"\n[LOADING TASK DATA]")
            query = json.load(open(os.path.join(task_input, "ex.json")))
            
            print(f"   Problem: {query.get('problem_text', 'N/A')[:100]}...")
            if 'ext_info' in query and 'label' in query['ext_info']:
                print(f"   Expected answer: {query['ext_info']['label']}")

            # load the images
            query['image_path_code'] = os.path.join(output_dir, query['image_path_code'])
            print(f"\n[IMAGE PATH]")
            print(f"   {query['image_path_code']}")
            print(f"   Image exists: {os.path.exists(query['image_path_code'])}")
            
            # Check ATR status
            print(f"\n[CHECKING ATR (Adaptive Token Reduction) STATUS]")
            atr_enable = os.environ.get("ATR_ENABLE", "false").lower() in ("1", "true", "yes")
            if atr_enable:
                print(f"   [+] ATR ENABLED")
                print(f"   Retention: {os.environ.get('ATR_RETENTION', '0.3')}")
                print(f"   Crop: {os.environ.get('ATR_CROP', 'false')}")
                print(f"   [!] NOTE: ATR preprocessing not implemented in this solver version!")
            else:
                print(f"   [-] ATR DISABLED (use enhanced_solver.py for ATR support)")
            
            images = []
            print(f"\n[INITIALIZING COMPONENTS]")
            prompt_generator = GeoPromptVisuoThink()
            print(f"   [OK] Prompt generator initialized")
            parser = Parser()
            print(f"   [OK] Parser initialized")
            executor = CodeExecutor(working_dir=task_directory)
            print(f"   [OK] Code executor initialized (working dir: {task_directory})")
        
        # agent setup
        print(f"\n[INITIALIZING AGENT]")
        agent = GeoProUserAgent(
            prompt_generator = prompt_generator,
            parser = parser,
            executor = executor,
            step_aux = aux_step(task_type)
        )
        print(f"   [OK] GeoProUserAgent created")
        print(f"   Auxiliary steps enabled: {aux_step(task_type)}")
        
        print(f"\n[STARTING CONVERSATION WITH MODEL]")
        init_message = agent.initiate_chat(query)
        print(f"   Initial message length: {len(init_message)} characters")
        messages = []
        
        print(f"\n[REASONING LOOP] (max {MAX_REPLY} iterations)")
        print("-" * 80)

        for i in range(MAX_REPLY):
            print(f"\n[ITERATION {i+1}/{MAX_REPLY}]")
            print(f"   Sending message to VLM...")
            
            model_response, messages = chat_vlm(init_message, messages)
            
            print(f"   [OK] Received response ({len(model_response)} characters)")
            print(f"   Total messages so far: {len(messages)}")

            if verbose:
                print(f"\n   [USER MESSAGE]")
                print_message(messages[-2])
                print(f"\n   [MODEL RESPONSE]")
                print_message(messages[-1])

            print(f"   Processing agent response...")
            reply = agent.receive(model_response)
            
            if reply is None:
                print(f"   [DONE] AGENT COMPLETED (no more steps needed)")
                print(f"   Total iterations used: {i+1}/{MAX_REPLY}")
                break
            else:
                print(f"   [NEXT] Agent requests next step...")
                
            init_message = reply
        else:
            print(f"\n   [WARN] MAX ITERATIONS REACHED ({MAX_REPLY})")

        print(f"\n" + "-" * 80)
        print(f"[REASONING COMPLETE]")
        
        # turn off server
        print(f"\n[CLEANUP]")
        agent.executor.cleanup()
        print(f"   [OK] Executor cleaned up")

        # save the results
        output_json_path = os.path.join(task_directory, "output.json")
        with open(output_json_path, "w") as f:
            json.dump(messages, f, indent=4, ensure_ascii=False)
        print(f"\n[RESULTS SAVED]")
        print(f"   Output JSON: {output_json_path}")
        print(f"   Log file:    {log_file_path}")
        print(f"   Total messages: {len(messages)}")
        
        print(f"\n" + "="*80)
        print(f"[SUCCESS] SOLVER FINISHED SUCCESSFULLY")
        print("="*80 + "\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("VISUOTHINK GEOMETRY SOLVER")
    print("="*80)
    
    # Show configuration
    print("\n[CONFIGURATION]")
    print(f"   Max reasoning steps: {MAX_REPLY}")
    
    # Check for config.py settings
    try:
        from config import MODEL_NAME, llm_config
        print(f"   Model: {MODEL_NAME}")
        if llm_config and 'config_list' in llm_config:
            config = llm_config['config_list'][0] if llm_config['config_list'] else {}
            print(f"   Temperature: {config.get('temperature', 'N/A')}")
            print(f"   API configured: {'YES' if config.get('api_key') and config.get('api_key') != 'YOUR_API_KEY_HERE' else 'NO'}")
    except Exception as e:
        print(f"   [WARN] Could not load config: {e}")
    
    # Environment variables
    print(f"\n[ENVIRONMENT VARIABLES]")
    print(f"   ATR_ENABLE: {os.environ.get('ATR_ENABLE', 'not set')}")
    print(f"   ATR_RETENTION: {os.environ.get('ATR_RETENTION', 'not set')}")
    print(f"   ATR_CROP: {os.environ.get('ATR_CROP', 'not set')}")
    print(f"   MODEL_NAME: {os.environ.get('MODEL_NAME', 'not set')}")
    
    TASK_DIR = "dataset/geometry/Dataset_GeomVerse/test_geomverse_TEST_D2_B100_data_1"
    OUTPUT_DIR = "outputs/geometry/Dataset_GeomVerse/test_geomverse_TEST_D2_B100_data_1"
    
    print(f"\n[TASK TO RUN]")
    print(f"   {TASK_DIR}")
    
    run_geo_task(TASK_DIR, OUTPUT_DIR, task_type="visuothink", verbose=True)
