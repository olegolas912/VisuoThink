import os, sys
import pickle
import platform
from autogen.coding import CodeBlock, LocalCommandLineCodeExecutor
try:
    from autogen.coding.jupyter import JupyterCodeExecutor, LocalJupyterServer
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False
import ast, re

# add the tools directory to the path
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


# for each dialogue, we will have a new code executor
class CodeExecutor:
    def __init__(
        self, 
        working_dir: str = "",
        use_vision_tools: bool = False,
        ):
        self.working_dir = working_dir
        self.use_vision_tools = use_vision_tools
        self.is_windows = platform.system() == "Windows"
        self.server = None
        
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir, exist_ok=True)
            
        print(f"   [INFO] Using {'LocalCommandLine' if self.is_windows else 'Jupyter'} code executor")
        self._restart()
    

    def _restart(self):
        self.cleanup()

        # set up the executor based on platform
        if self.is_windows:
            # Use LocalCommandLine on Windows (Python subprocess-based execution)
            self.executor = LocalCommandLineCodeExecutor(
                timeout=60,
                work_dir=self.working_dir
            )
        else:
            # Use Jupyter on Linux/Mac
            self.server = LocalJupyterServer()
            self.executor = JupyterCodeExecutor(self.server, output_dir=self.working_dir)

        # initialize the environment
        self.init_env(self.use_vision_tools)
        

    def result_processor(self, result):
        # Change an IPythonCodeResult object to a string, and the list of files
        # If the execution failed, the string is the error message.
        # If the execution is successful, the string is the output of the code execution.
        # In the string, all embeded PIL images are replaced by their file paths, using html img tag.
        # The list of files are the paths of the images.
        
        # process error message
        def parse_error_message(error):
            # Find the index where the list starts, indicated by `['`
            list_start_index = error.find("['")
            
            # The first part before the list is the initial error message
            initial_error = error[:list_start_index].strip()
            
            # The second part is the list of strings, which starts from `['` and goes to the end of the string
            traceback_list_str = error[list_start_index:]
            
            # Use ast.literal_eval to safely evaluate the string representation of the list
            # This method is safer than eval and can handle Python literals properly
            try:
                traceback_list = ast.literal_eval(traceback_list_str)
            except SyntaxError as e:
                print("Error parsing the list: ", e)
                traceback_list = []
                
            # Remove ANSI escape sequences
            ansi_escape = re.compile(r'\x1b\[.*?m')
            traceback_list = [ansi_escape.sub('', line) for line in traceback_list]
            
            return initial_error + "\n\n" + "\n".join(traceback_list)
        
        
        exit_code = result.exit_code
        
        # Handle different result types (Jupyter vs CommandLine)
        file_paths = getattr(result, 'output_files', [])
        output_str = result.output
        output_lines = output_str.split("\n")
        
        if len(file_paths) > 0:
            output_lines = output_lines[:-2*len(file_paths)]
            
        # if execution succeeded, replace PIL images with their file paths
        if exit_code == 0:
            new_str = ""
            image_idx = 0
            
            for line in output_lines:
                if line.startswith("<PIL."):
                    if image_idx < len(file_paths):
                        new_str += f"<img src='{file_paths[image_idx]}'>"
                        image_idx += 1
                else:
                    new_str += line
                new_str += "\n"
            
            # add the remaining images
            for file_idx, file in enumerate(file_paths):
                if file_idx >= image_idx:
                    new_str += f"<img src='{file}'>"
                    new_str += "\n"
                
            return exit_code, new_str, file_paths
        
        # if execution failed, parse the error message
        else:
            error_msg = parse_error_message(output_str)
            return exit_code, error_msg, file_paths
    
    def execute(self, code: str, init_env: bool = False):
        # For Jupyter executor, refresh the kernel client
        if hasattr(self.executor, '_jupyter_kernel_client'):
            self.executor._jupyter_kernel_client = self.executor._jupyter_client.get_kernel_client(self.executor._kernel_id)
        
        execution_result = self.executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(language="python",
                        code=code),
            ]
        )
        ret = self.result_processor(execution_result)
        if not init_env:
            self._restart() # In case some code does not work, just restart the environment after every execution
        return ret
    
    def init_env(self, use_vision_tools):
        # Build initialization code based on executor type
        init_code = "import sys\n"
        init_code += "from PIL import Image\n"
        
        # Only import IPython.display for Jupyter executor (not CommandLine)
        if not self.is_windows:
            init_code += "from IPython.display import display\n"
        
        init_code += f"parent_dir = '{parent_dir}'\n"
        init_code += "if parent_dir not in sys.path:\n"
        init_code += "    sys.path.insert(0, parent_dir)\n"
        init_code += "from tools_geo import *\n"
        init_code += "import math\n"
        
        init_resp = self.execute(init_code, init_env=True)
        print(init_resp[1])


    def cleanup(self):
        if self.server is not None:
            self.server.stop()
