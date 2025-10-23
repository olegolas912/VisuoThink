"""
Enhanced geometry solver with Adaptive Token Reduction (ATR) integration.

This module extends the original VisuoThink solver to include the paper's
adaptive token reduction method for improved geometry problem solving.
"""

import json
import os
import argparse
import shutil
from contextlib import redirect_stdout
from pathlib import Path
from copy import deepcopy
import logging

# Import original modules
if __package__ is None or __package__ == "":
    import sys
    _PKG_DIR = os.path.dirname(os.path.abspath(__file__))
    if _PKG_DIR not in sys.path:
        sys.path.insert(0, _PKG_DIR)
    
    from agent import GeoProUserAgent
    from prompt import GeoPromptVisuoThink
    from parse import Parser
    from execution import CodeExecutor
    from utils_misc import tee_stdout, print_message
    from utils_llm import chat_vlm
    from metrics import record_task_metrics
else:
    from .agent import GeoProUserAgent
    from .prompt import GeoPromptVisuoThink
    from .parse import Parser
    from .execution import CodeExecutor
    from .utils_misc import tee_stdout, print_message
    from .utils_llm import chat_vlm
    from .metrics import record_task_metrics

from tqdm import tqdm

# Import configuration
try:
    from config import MAX_REPLY, llm_config
except ImportError:
    # Fallback configuration
    MAX_REPLY = 10
    llm_config = {"config_list": [{"model": "gpt-4o", "temperature": 0.0}]}

logger = logging.getLogger(__name__)


def aux_step(task_type: str) -> bool:
    """Determine if auxiliary step is needed based on task type."""
    return True if task_type in ['geovar2', 'visuothink'] else False


class EnhancedGeoPromptVisuoThink(GeoPromptVisuoThink):
    """
    Enhanced prompt generator that incorporates ATR processing information.
    """
    
    def __init__(self, atr_enabled: bool = False, retention_ratio: float = 0.3):
        super().__init__()
        self.atr_enabled = atr_enabled
        self.retention_ratio = retention_ratio
        
    def initial_prompt(self, ex, n_images: int) -> str:
        """Generate initial prompt with ATR processing information."""
        prompt = super().initial_prompt(ex, n_images)
        
        # Add ATR processing information if enabled
        if self.atr_enabled:
            atr_info = self._create_atr_info()
            prompt += atr_info
            
        return prompt
        
    def _create_atr_info(self) -> str:
        """Create ATR processing information section."""
        reduction_pct = (1.0 - self.retention_ratio) * 100
        
        info = f"""
        
# ADAPTIVE TOKEN REDUCTION (ATR) PROCESSING #
This geometry problem image has been processed using the paper's adaptive token reduction method:
- Token retention ratio: {self.retention_ratio:.1%} (reduced {reduction_pct:.1f}% of visual tokens)
- Background noise has been suppressed using learned attention mechanisms
- Only the most informative visual features have been retained
- This processing is task-agnostic and requires no model fine-tuning

Focus on the geometric relationships and shapes that are most relevant to solving this problem.
The image processing has been optimized to emphasize essential geometric elements over background details.
"""
        return info


def run_enhanced_geo_task(task_input: str, 
                        output_dir: str, 
                        task_type: str = "visuothink",
                        verbose: bool = False,
                        enable_atr: bool = True,
                        retention_ratio: float = 0.3,
                        enable_visualization: bool = False,
                        rollout_search: bool = False,
                        tree_span: int = 3,
                        search: bool = False):
    """
    Run a geometry task with enhanced ATR capabilities.
    
    Args:
        task_input: Path to the task input directory
        output_dir: Path to the output directory
        task_type: Type of task (currently only "visuothink" supported)
        verbose: Whether to print verbose output
        enable_atr: Whether to enable ATR processing
        retention_ratio: Ratio of tokens to retain (0.0 to 1.0)
        enable_visualization: Whether to generate processing visualizations
        rollout_search: Whether to enable rollout search (not implemented)
        tree_span: Tree span for search (not implemented)
        search: Whether to enable search (not implemented)
    """
    assert task_type in ["visuothink"], f"Task type {task_type} not supported"
    
    # Create output directory
    task_input = task_input.rstrip('/')
    task_directory = os.path.join(output_dir, os.path.basename(task_input))
    
    # Clean and create directories
    shutil.rmtree(task_directory, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    shutil.copytree(task_input, task_directory, dirs_exist_ok=True)
    log_file_path = os.path.join(task_directory, 'output.log')
    
    with tee_stdout(log_file_path):
        try:
            # Load task data
            query = json.load(open(os.path.join(task_input, "ex.json")))
            query['image_path_code'] = os.path.join(output_dir, query['image_path_code'])
            
            if verbose:
                print(f"[ENHANCED SOLVER] Processing task: {os.path.basename(task_directory)}")
                print(f"[ENHANCED SOLVER] Image path: {query['image_path_code']}")
            
            # Apply ATR preprocessing if enabled
            atr_applied = False
            if enable_atr:
                try:
                    if verbose:
                        print(f"[ENHANCED SOLVER] Applying ATR with {retention_ratio:.1%} retention...")
                    
                    # Set environment variables for ATR
                    os.environ["ATR_ENABLE"] = "true"
                    os.environ["ATR_RETENTION"] = str(retention_ratio)
                    os.environ["ATR_CROP"] = "true" if enable_visualization else "false"
                    
                    # The ATR preprocessing will be applied in the solver
                    atr_applied = True
                    
                    if verbose:
                        reduction_pct = (1.0 - retention_ratio) * 100
                        print(f"[ENHANCED SOLVER] ATR enabled: {reduction_pct:.1f}% token reduction")
                        
                except Exception as e:
                    logger.warning(f"ATR processing failed, falling back to standard processing: {e}")
                    if verbose:
                        print(f"[ENHANCED SOLVER] Warning: ATR processing failed: {e}")
                    enable_atr = False
            
            # Initialize components
            prompt_generator = EnhancedGeoPromptVisuoThink(
                atr_enabled=atr_applied,
                retention_ratio=retention_ratio
            )
            parser = Parser()
            executor = CodeExecutor(working_dir=task_directory)
            
            # Initialize agent
            agent = GeoProUserAgent(
                prompt_generator=prompt_generator,
                parser=parser,
                executor=executor,
                step_aux=aux_step(task_type)
            )
            
            # Generate initial message
            init_message = agent.initiate_chat(query)
            messages = []
            
            if verbose:
                print(f"[ENHANCED SOLVER] Starting reasoning with max {MAX_REPLY} steps")
            
            # Main reasoning loop
            for i in range(MAX_REPLY):
                if verbose:
                    print(f"[ENHANCED SOLVER] Step {i+1}/{MAX_REPLY}")
                
                model_response, messages = chat_vlm(init_message, messages)
                
                if verbose:
                    print_message(messages[-2])
                    print_message(messages[-1])
                
                reply = agent.receive(model_response)
                if reply is None:
                    if verbose:
                        print(f"[ENHANCED SOLVER] Task completed at step {i+1}")
                    break
                init_message = reply
            
            # Cleanup
            agent.executor.cleanup()
            
            # Save results
            output_json_path = os.path.join(task_directory, "output.json")
            with open(output_json_path, "w") as f:
                json.dump(messages, f, indent=4, ensure_ascii=False)
            
            # Record metrics
            config_entry = deepcopy((llm_config.get("config_list") or [{}])[0])
            metrics_info = record_task_metrics(Path(task_directory), messages, config_entry)
            metrics = metrics_info["metrics"]
            
            # Print results
            print(
                "[METRICS] Task {task} | success={success} | correct={correct} | answer={answer} | "
                "ref={reference} | turns={turns} | thoughts={thoughts} | actions={actions}".format(
                    task=os.path.basename(task_directory),
                    success=metrics["success"],
                    correct=metrics["correct"],
                    answer=metrics["final_answer"] or "-",
                    reference=metrics.get("reference"),
                    turns=metrics["turns"],
                    thoughts=metrics["thought_messages"],
                    actions=metrics["action_messages"],
                )
            )
            
            # Add ATR processing information to metrics
            if atr_applied:
                reduction_pct = (1.0 - retention_ratio) * 100
                print(f"[ATR METRICS] Token reduction: {reduction_pct:.1f}%")
                print(f"[ATR METRICS] Retention ratio: {retention_ratio:.1%}")
                print(f"[ATR METRICS] Background suppression: Enabled")
            
            print(f"[METRICS] Saved per-task metrics to {metrics_info['metrics_path']} (history: {metrics_info['history_path']})")
            
        except Exception as e:
            logger.error(f"Error in enhanced geometry task: {e}")
            if verbose:
                print(f"[ENHANCED SOLVER] Error: {e}")
            raise
        finally:
            # Clean up environment variables
            if enable_atr:
                os.environ.pop("ATR_ENABLE", None)
                os.environ.pop("ATR_RETENTION", None)
                os.environ.pop("ATR_CROP", None)


def run_enhanced_geo_task_batch(task_inputs: list,
                              output_base_dir: str,
                              task_type: str = "visuothink",
                              verbose: bool = False,
                              enable_atr: bool = True,
                              retention_ratio: float = 0.3,
                              enable_visualization: bool = False):
    """
    Run multiple geometry tasks with enhanced ATR processing.
    
    Args:
        task_inputs: List of task input directories
        output_base_dir: Base output directory
        task_type: Type of tasks
        verbose: Whether to print verbose output
        enable_atr: Whether to enable ATR processing
        retention_ratio: Token retention ratio
        enable_visualization: Whether to enable visualizations
    """
    results = []
    
    for task_input in tqdm(task_inputs, desc="Processing tasks"):
        try:
            task_name = os.path.basename(task_input)
            output_dir = os.path.join(output_base_dir, task_name)
            
            if verbose:
                print(f"\n[BATCH PROCESSING] Processing task: {task_name}")
            
            run_enhanced_geo_task(
                task_input=task_input,
                output_dir=output_dir,
                task_type=task_type,
                verbose=verbose,
                enable_atr=enable_atr,
                retention_ratio=retention_ratio,
                enable_visualization=enable_visualization
            )
            
            results.append({
                'task': task_name,
                'status': 'success',
                'output_dir': output_dir
            })
            
        except Exception as e:
            logger.error(f"Failed to process task {task_input}: {e}")
            results.append({
                'task': os.path.basename(task_input),
                'status': 'failed',
                'error': str(e)
            })
    
    # Print summary
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    
    print(f"\n[BATCH PROCESSING SUMMARY]")
    print(f"Total tasks: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if enable_atr:
        reduction_pct = (1.0 - retention_ratio) * 100
        print(f"ATR processing enabled with {reduction_pct:.1f}% token reduction")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced VisuoThink Geometry Solver with ATR")
    parser.add_argument("--task_dir", type=str, 
                       default="dataset/geometry/Dataset_GeomVerse/test_geomverse_TEST_D2_B100_data_9",
                       help="Path to the task directory")
    parser.add_argument("--output_dir", type=str,
                       default="outputs/geometry/enhanced_test",
                       help="Path to the output directory")
    parser.add_argument("--task_type", type=str, default="visuothink",
                       help="Type of task to solve")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--disable_atr", action="store_true",
                       help="Disable ATR processing")
    parser.add_argument("--retention_ratio", type=float, default=0.3,
                       help="Token retention ratio (0.0 to 1.0)")
    parser.add_argument("--enable_visualization", action="store_true",
                       help="Enable processing visualizations")
    parser.add_argument("--batch_mode", action="store_true",
                       help="Process multiple tasks in batch mode")
    parser.add_argument("--batch_size", type=int, default=10,
                       help="Number of tasks to process in batch mode")
    
    args = parser.parse_args()
    
    if args.batch_mode:
        # Batch processing mode
        dataset_dir = "dataset/geometry/Dataset_GeomVerse"
        task_dirs = []
        
        for item in os.listdir(dataset_dir):
            item_path = os.path.join(dataset_dir, item)
            if os.path.isdir(item_path) and item.startswith("test_"):
                task_dirs.append(item_path)
        
        # Limit batch size
        task_dirs = task_dirs[:args.batch_size]
        
        run_enhanced_geo_task_batch(
            task_inputs=task_dirs,
            output_base_dir=args.output_dir,
            task_type=args.task_type,
            verbose=args.verbose,
            enable_atr=not args.disable_atr,
            retention_ratio=args.retention_ratio,
            enable_visualization=args.enable_visualization
        )
    else:
        # Single task mode
        run_enhanced_geo_task(
            task_input=args.task_dir,
            output_dir=args.output_dir,
            task_type=args.task_type,
            verbose=args.verbose,
            enable_atr=not args.disable_atr,
            retention_ratio=args.retention_ratio,
            enable_visualization=args.enable_visualization
        )
