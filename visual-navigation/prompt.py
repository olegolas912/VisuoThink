
############################################################################################################
##### Prompt Generator for Visual Navigation tasks using ReACT agent
############################################################################################################ 


class VisualNavigationCoT:

    def __init__(self, task_path: str) -> None:
        # test example
        configuration_img_path = f'{task_path}/map.png'
        prompt = f"""
Navigation Task: for a provided map, 🏠 is the home as starting point, 🏢 is the office as the destination. ⬜ means the road, 🚧 means the obstacle. There exists one and only one viable route for each map. Each step you choose a direction and move to the end of the continuous road or the destination.

map:
<img src='{configuration_img_path}'>
Starting from 🏠, provide the steps to navigate to 🏢.
Let's think step by step.

Each step should be in json format like:
Step x:
```json
{{"direction": <direction>, "steps": <number of steps>}}
```
"""
        self.prompt = prompt


class VisualNavigationVoT:

    def __init__(self, task_path: str) -> None:
        # test example
        configuration_img_path = f'{task_path}/map.png'
        prompt = f"""
Navigation Task: for a provided map, 🏠 is the home as starting point, 🏢 is the office as the destination. ⬜ means the road, 🚧 means the obstacle. There exists one and only one viable route for each map. Each step you choose a direction and move to the end of the continuous road or the destination.

map:
<img src='{configuration_img_path}'>
Starting from 🏠, provide the steps to navigate to 🏢.
Visualize the state after each reasoning step.

Each step should be in json format like:
Step x:
```json
{{"direction": <direction>, "steps": <number of steps>}}
```
"""
        self.prompt = prompt


class VisualNavigationPrompt:
    def __init__(self) -> None:
        self.continue_prompt = """
Please provide your next THOUGHT and ACTION. Your ACTION should be in json format like:
```json
{"direction": <direction>, "steps": <number of steps>}
```
"""
    
    def initial_prompt(self, task_path: str) -> str:
        initial_prompt = f'''
Navigation Task: for a provided map, 🏠 is the home as starting point, 🏢 is the office as the destination. ⬜ means the road, 🚧 means the obstacle. There exists one and only one viable route for each map. Each step you choose a direction and move to the end of the continuous road or the destination.
'''
        prompt = initial_prompt
        
        # test example
        configuration_img_path = f'{task_path}/map.png'

        prompt += f"Here is the map:\n<img src='{configuration_img_path}'>\n"
        prompt += self.continue_prompt

        return prompt
    

    def get_exec_feedback(self, exit_status, exit_message, exit_file_pths) -> str:
        # if execution fails
        if exit_file_pths:
            assert len(exit_file_pths) == 1, "Only one file path is expected"
            img_path = exit_file_pths[0]
            visual_prompt = f"""
Here is the current state of the map:
🏠 is the home as starting point, 🏢 is the office as the destination. ⬜ means the road, 🚧 means the obstacle, and 🚶 means the current position.
<img src='{img_path}'>
"""
        else:
            visual_prompt = ""

        if not exit_status:
           prompt = f"OBSERVATION: Execution error. Output:\n{exit_message}\nPlease fix the error.\n"
        else:
            prompt = f"OBSERVATION: Execution success. The output is as follows:\n{exit_message}\n"
        prompt += (visual_prompt + self.continue_prompt)
        return prompt

