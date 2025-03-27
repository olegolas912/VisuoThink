import os, sys
import pickle
import json
import ast, re
from PIL import Image, ImageFont
import pilmoji
import matplotlib.pyplot as plt

from copy import deepcopy
from utils_misc import print_error
# add the tools directory to the path
from utils import json_print

emoji_size = 109
spacing = int(emoji_size / 5)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
font = ImageFont.truetype(os.path.join(SCRIPT_DIR, 'NotoColorEmoji.ttf'), size=emoji_size)


def emoji_to_image(emoji_grid:list[list[str]]):
    # Create a new image with white background
    row_num = len(emoji_grid)
    col_num = len(emoji_grid[0])
    img_size = ((emoji_size + spacing) * col_num + spacing, (emoji_size + spacing) * row_num + spacing)  # width, height
    img = Image.new('RGB', img_size, color='black')

    # Draw the emojis onto the image
    draw = pilmoji.Pilmoji(img)
    # Draw each emoji in the grid
    for row_index, row in enumerate(emoji_grid):
        for col_index, emoji in enumerate(row):
            position = (col_index * (emoji_size + spacing) + spacing, row_index * (emoji_size + spacing) + spacing)
            #draw.text(position, emoji, fill="black", spacing=0, font_size=emoji_size[0])
            while True:
                try:
                    draw.text(position, emoji, font=font)
                    break
                except Exception as err:
                    print_error(err)
    # img = img.resize((int(img_size[0] / 3), int(img_size[1] / 3)))
    # Save or display the image
    return img


def emoji_to_imagefile(emoji_grid:list[list[str]], file_path:str):
    img = emoji_to_image(emoji_grid)
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0, transparent=True)


class NavigationParser:
    """Take the json snippet and parse it into a python dict object."""
    def parse(self, response):
        if isinstance(response, dict) and 'content' in response:
            content = response['content']
        else:
            content = response
        
        try:
            try:
                # Direct Parsing Pathaway (DPP), for Qwen2-VL-72B fails to output enclosed actions sometimes.
                content = json.loads(content)
                return {'status': True, 'content': content, 'message': 'Parsing succeeded.', 'file_pths': None}
            except:
                pass
            
            start_pos = content.find("```json")
            if start_pos != -1:
                content = content[start_pos+len("```json"):]

            end_pos = content.find("```")
            if end_pos != -1:
                content = content[:end_pos]
            
            if start_pos == -1 or end_pos == -1:
                return {'status': False, 'content': content, 'message': 'Action is NOT enclosed in ```json``` properly.', 'file_pths': None}
            if len(content) > 0:
                content = json.loads(content)
                return {'status': True, 'content': content, 'message': 'Parsing succeeded.', 'file_pths': None}
            else:
                return {'status': False, 'content': content, 'message': "The content is empty, or it failed to parse the content correctly.", 'file_pths': None}
        except Exception as err:
            return {'status': False, 'content': content, 'message': f"Unexpected {type(err)}: {err}.", 'file_pths': None}
    

# for each dialogue, we will have a new code executor
class VisiualNavigationExecutor:
    def __init__(
        self, 
        configuration: dict,
        working_dir: str,
        visual: bool = False,
        ):
        self.working_dir = working_dir
        self.parser = NavigationParser()
        self.map = configuration['initial_map']
        self.current_pos = configuration['current_pos']
        self.start_pos = configuration['start_pos']
        self.dest_pos = configuration['dest_pos']
        self.image_cnt = 0
        self.visual = visual
        
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir, exist_ok=True)
    

    def _move(self, direction: str, steps: int):
        next_pos = self.current_pos
        if direction == 'up':
            next_pos = (self.current_pos[0], self.current_pos[1] - steps)
        if direction == 'down':
            next_pos = (self.current_pos[0], self.current_pos[1] + steps)
        if direction == 'left':
            next_pos = (self.current_pos[0] - steps, self.current_pos[1])
        if direction == 'right':
            next_pos = (self.current_pos[0] + steps, self.current_pos[1])
        
        self.already_moved_steps += steps
        
        if next_pos[0] < 0 or next_pos[0] >= len(self.map[0]) or next_pos[1] < 0 or next_pos[1] >= len(self.map):
            raise Exception(f"You can not move ## {direction} ## for {self.moved_steps} steps.\nYou are out of the map {next_pos} after moving ## {direction} ## for {self.already_moved_steps} steps.\nAs a result, you are still at {self.saved_current_pos}")
        
        map_value = self.map[next_pos[1]][next_pos[0]]

        if map_value == '🚧':
            raise Exception(f"You can not move ## {direction} ## for {self.moved_steps} steps.\nYou are at a wall after moving ## {direction} ## for {self.already_moved_steps} steps.\nAs a result, you are still at {self.saved_current_pos}")
        
        self.current_pos = next_pos
    

    def _visualize(self):
        map_copy = deepcopy(self.map)
        current_pos = self.current_pos
        map_copy[current_pos[1]][current_pos[0]] = '🚶'

        image_path = os.path.join(self.working_dir, f'image_{self.image_cnt}.png')
        self.image_cnt += 1 
        emoji_to_imagefile(map_copy, image_path)
        return image_path


    def save_state(self):
        state = {
            'current_pos': self.current_pos,
        }
        return state

    def load_state(self, state):
        self.current_pos = state['current_pos']

    
    def execute(self, cmd):
        if isinstance(cmd, str):
            cmd = self.parser.parse(cmd)
            if not cmd['status']:
                return cmd
            cmd = cmd['content']

        self.saved_current_pos = self.current_pos
        try:
            direction = cmd['direction'].lower()
            steps = cmd['steps']

            self.moved_steps = steps 
            self.already_moved_steps = 0

            assert direction in ['up', 'down', 'left', 'right'], f"Direction must be one of ['up', 'down', 'left', 'right'], but got {direction}"
            assert steps > 0, f"Steps must be greater than 0, but got {steps}"

            for i in range(steps):
                self._move(direction, 1)
            
            next_pos = self.current_pos
            map_value = self.map[next_pos[1]][next_pos[0]]
            if map_value == '🏢':
                return {'status': True, 'message': f"You have reached the destination after moving {self.already_moved_steps} steps. TERMINATE.", 'file_pths': None if not self.visual else [self._visualize()]}
            else:
                return {'status': True, 'message': f"You have successfully moved ## {direction} ## for {self.already_moved_steps} steps. You are now at {self.current_pos}.", 'file_pths': None if not self.visual else [self._visualize()]}

        
        except Exception as err:
            self.current_pos = self.saved_current_pos
            return {'status': False, 'message': f"Unexpected {type(err)}: {err}.", 'file_pths': None if not self.visual else [self._visualize()]}



if __name__ == "__main__":
    task_path = 'dataset/visual-navigation/configurations/level-5/2'
    configuration = json.load(open(f'{task_path}/solution.json', 'r'))

    # -- map -- #
    # ⬜⬜⬜🚧🚧🚧🚧
    # ⬜🚧⬜🚧🚧🚧🚧
    # ⬜🚧🏠🚧⬜⬜🏢
    # ⬜🚧🚧🚧⬜🚧🚧
    # ⬜⬜⬜⬜⬜🚧🚧
    # 
    # 🚶

    executor = VisiualNavigationExecutor(configuration, 'workspace/outputs/visual-navigation/level-6/2')
    json_print(executor.execute('nohup python3 -m autogen.coding.jupyter.jupyter_server'))
    json_print(executor.execute('```json\n{"direction": "uu", "steps": 1}\n```'))
    json_print(executor.execute('```json\n{"direction": "up", "stepssss": 1}\n```'))
    json_print(executor.execute('```json\n{"direction": "down", "steps": 1}\n```'))
    json_print(executor.execute('```json\n{"direction": "up", "steps": 3}\n```'))
    json_print(executor.execute('```json\n{"direction": "up", "steps": 2}\n```')) # Only action expected to be executed

    executor.current_pos = (4, 2)
    json_print(executor.execute('```json\n{"direction": "up", "steps": 2}\n```'))
    json_print(executor.execute('```json\n{"direction": "right", "steps": 1}\n```'))
    json_print(executor.execute('```json\n{"direction": "right", "steps": 1}\n```')) # Expected to terminate
