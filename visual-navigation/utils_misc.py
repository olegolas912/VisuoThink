import json
import sys
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image as PILImage
import base64
from io import BytesIO
from functools import wraps
from contextlib import redirect_stdout
from datetime import datetime


class TeeOutput:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = None
        self.stdout = sys.stdout

    def __enter__(self):
        self.file = open(self.file_path, 'w')
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        self.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.stdout.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        if self.file:
            self.file.close()


def tee_stdout(file_path):
    """
    Can be used either as a decorator or context manager to redirect stdout to both file and console
    """
    return TeeOutput(file_path)


def image_to_base64(pil_img):
    # Create a BytesIO buffer to save the image
    buffered = BytesIO()
    # Save the image in the buffer using a format like PNG
    pil_img.save(buffered, format="PNG")
    # Get the byte data from the buffer
    img_byte = buffered.getvalue()
    # Encode the bytes to base64
    img_base64 = base64.b64encode(img_byte)
    # Decode the base64 bytes to string
    return img_base64.decode('utf-8')


def custom_encoder(obj):
    """Custom JSON encoder function that replaces Image objects with '<image>'.
       Delegates the encoding of other types to the default encoder."""
    if isinstance(obj, Image.Image):
        return image_to_base64(obj)
    # Let the default JSON encoder handle any other types
    return json.JSONEncoder().default(obj)


def print_message(message_dict):
    role = message_dict['role']
    message = message_dict['content']

    # print role with color
    COLORS = {
        'assistant': '\033[92m',  # Green
        'user': '\033[94m',      # Blue
        'system': '\033[93m',    # Yellow
        'RESET': '\033[0m'       # Reset color
    }
    
    # Get color for role (default to reset if role not found)
    color = COLORS.get(role, COLORS['RESET'])
    
    # Print role with color and message
    print('-' * 50)
    print(f"{color}[{role}]{COLORS['RESET']}: \n\n{message}\n\n")


def print_error(message):
    message = f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {message}'
    print(f"\033[91m\033[1m{message}\033[0m")

if __name__ == "__main__":
    # As a context manager
    with tee_stdout('output.txt'):
        print("This goes to both console and file")
