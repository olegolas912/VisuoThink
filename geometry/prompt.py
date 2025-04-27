
############################################################################################################
##### Prompt Generator for Geometry tasks using ReACT agent

##### Allowing the agent to plot the auxiliary lines in multiple steps.
############################################################################################################ 


INIT_PROMPT = r'''
Here are some tools you can use:

def solve_equation(equation: str) -> float:
    """
    Solve a math equation about variable x and return the value of x.
    """

def primeter_of_rectangle_with_equilateral_triangle(length_of_shared_side: float | str, length_of_other_side: float | str, result=None) -> float | str:
    """
    Given a rectangle with a equilateral triangle sharing a side. 

    If result is not None, return the primeter equation, otherwise return the value of the primeter.
    """

def area_of_rectangle_with_equilateral_triangle_removed(length_of_shared_side: float | str, length_of_other_side: float | str, result=None) -> float | str:
    """
    Given a rectangle with a equilateral triangle sharing a side. 

    If result is not None, return the area equation, otherwise return the value of the area.
    """

def area_of_rectangle_with_equilateral_triangle_combined(length_of_shared_side: float | str, length_of_other_side: float | str, result=None) -> float | str:
    """
    Given a rectangle with a equilateral triangle sharing a side. 

    If result is not None, return the area equation, other wise return the value of the area.
    """

def primeter_of_rectangle_with_semicircle(length_of_shared_side: float | str, length_of_other_side: float | str, result=None) -> float | str:
    """
    Given a rectangle with a semicircle sharing a side. 

    If result is not None, return the primeter equation, other wise return the value of the primeter.
    """

def area_of_rectangle_with_semicircle_removed(length_of_shared_side: float | str, length_of_other_side: float | str, result=None) -> float | str:
    """
    Given a rectangle with a semicircle sharing a side. 

    If result is not None, return the area equation, other wise return the value of the area.
    """

def area_of_rectangle_with_semicircle_combined(length_of_shared_side: float | str, length_of_other_side: float | str, result=None) -> float | str:
    """
    Given a rectangle with a semicircle sharing a side. 

    If result is not None, return the area equation, other wise return the value of the area.
    """

The jupyter notebook has already executed the following code to import the necessary tools, including the tools above:
```python
from tools_geo import *
```

# REQUIREMENTS #:
1. The generated actions can resolve the given user request # USER REQUEST # perfectly. The user request is reasonable and can be solved. Try your best to solve the request.
2. If you think you got the answer, use ANSWER: <your answer> to provide the answer, and ends with TERMINATE.
3. How to use the tools:
For instance, If the length of the side opposite to the shared side is provided, we have already known the length of equilateral triangle's side length. Then we call `equation = primeter_of_rectangle_with_equilateral_triangle(length_of_shared_side=known_length, length_of_other_side='x', result=known_primeter)`.
Otherwise, if we don't know the length of the side opposite to the shared side, but we know the length of the other side, then we call `equation = primeter_of_rectangle_with_equilateral_triangle(length_of_shared_side='x', length_of_other_side=known_length, result=known_primeter)`.

# USER REQUEST #: Given the geometry diagram <img src='dataset/test_geomverse/test_geomverse_TEST_D2_B100_data_1/1.png'> and the matplotlib code of the geometry:
```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Coordinates
A = (0, 0)
B = (0, 19.54416)
C = (16.97056, 19.54416)
D = (16.97056, 0)
E = (8.48528, 14.69694)
F = (0, 36.51472)
G = (16.97056, 36.51472)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 16))

# Draw the lime polygon (A, B, C, D, E)
lime_polygon = patches.Polygon([A, B, C, D, E], closed=True, color='lime', alpha=0.6)
ax.add_patch(lime_polygon)

# Draw the cyan polygon (B, F, G, C)
cyan_polygon = patches.Polygon([B, F, G, C], closed=True, color='cyan', alpha=0.6)
ax.add_patch(cyan_polygon)

# Draw right-angle markers
def draw_right_angle(ax, p1, p2, p3, size=1, color="gray", alpha=0.8):
    """Draws a right-angle marker at the corner defined by p1, p2, and p3."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    v1 = v1 / np.linalg.norm(v1) * size
    v2 = v2 / np.linalg.norm(v2) * size
    marker = patches.Polygon(
        [p2, p2 + v1, p2 + v1 + v2, p2 + v2],
        closed=True,
        facecolor=color,
        edgecolor=None,
        alpha=alpha
    )
    ax.add_patch(marker)

# Add right-angle markers
draw_right_angle(ax, A, B, C)
draw_right_angle(ax, B, C, D)
draw_right_angle(ax, B, F, G)
draw_right_angle(ax, F, G, C)
draw_right_angle(ax, G, C, B)
draw_right_angle(ax, C, B, F)

# Add labels
ax.text(A[0], A[1], "A", fontsize=16, ha="right", va="top")
ax.text(B[0], B[1], "B", fontsize=16, ha="right", va="bottom")
ax.text(C[0], C[1], "C", fontsize=16, ha="left", va="bottom")
ax.text(D[0], D[1], "D", fontsize=16, ha="left", va="top")
ax.text(E[0], E[1], "E", fontsize=16, ha="center", va="bottom")
ax.text(F[0], F[1], "F", fontsize=16, ha="right", va="bottom")
ax.text(G[0], G[1], "G", fontsize=16, ha="left", va="bottom")

# Configure the plot
ax.set_xlim(-5, 25)
ax.set_ylim(-5, 45)
ax.set_aspect('equal')
ax.axis('off')

# Show the plot
plt.show()
```
you must solve the following question: [If the ABCDE shape is a rectangle where an equilateral triangle has been removed from one side of it, the perimeter of the ABCDE shape is 90 and the diagonal of the BFGC square is 24, compute the length of the AB side of the ABCDE shape.]
# RESULT #:
THOUGHT 0:
To solve for the length of the AB side of the ABCDE shape, we need to find the length of the side of the square BFGC.
Let's draw the auxiliary line BC. AUX-END.
ACTION 0:
```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Coordinates
A = (0, 0)
B = (0, 19.54416)
C = (16.97056, 19.54416)
D = (16.97056, 0)
E = (8.48528, 14.69694)
F = (0, 36.51472)
G = (16.97056, 36.51472)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 16))

# Draw the lime polygon (A, B, C, D, E)
lime_polygon = patches.Polygon([A, B, C, D, E], closed=True, color='lime', alpha=0.6)
ax.add_patch(lime_polygon)

# Draw the cyan polygon (B, F, G, C)
cyan_polygon = patches.Polygon([B, F, G, C], closed=True, color='cyan', alpha=0.6)
ax.add_patch(cyan_polygon)

# Draw right-angle markers
def draw_right_angle(ax, p1, p2, p3, size=1, color="gray", alpha=0.8):
    """Draws a right-angle marker at the corner defined by p1, p2, and p3."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    v1 = v1 / np.linalg.norm(v1) * size
    v2 = v2 / np.linalg.norm(v2) * size
    marker = patches.Polygon(
        [p2, p2 + v1, p2 + v1 + v2, p2 + v2],
        closed=True,
        facecolor=color,
        edgecolor=None,
        alpha=alpha
    )
    ax.add_patch(marker)

# Add right-angle markers
draw_right_angle(ax, A, B, C)
draw_right_angle(ax, B, C, D)
draw_right_angle(ax, B, F, G)
draw_right_angle(ax, F, G, C)
draw_right_angle(ax, G, C, B)
draw_right_angle(ax, C, B, F)

# Add labels
ax.text(A[0], A[1], "A", fontsize=16, ha="right", va="top")
ax.text(B[0], B[1], "B", fontsize=16, ha="right", va="bottom")
ax.text(C[0], C[1], "C", fontsize=16, ha="left", va="bottom")
ax.text(D[0], D[1], "D", fontsize=16, ha="left", va="top")
ax.text(E[0], E[1], "E", fontsize=16, ha="center", va="bottom")
ax.text(F[0], F[1], "F", fontsize=16, ha="right", va="bottom")
ax.text(G[0], G[1], "G", fontsize=16, ha="left", va="bottom")

# Configure the plot
ax.set_xlim(-5, 25)
ax.set_ylim(-5, 45)
ax.set_aspect('equal')
ax.axis('off')

# Draw the auxiliary line
auxiliary_lines = [(B, C)]
for (p1, p2) in auxiliary_lines:
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--', linewidth=3)  # Draw in red dashed line for clarity

# Show the plot
plt.show()
```
OBSERVATION: Execution success. The output is as follows:
<the image outputs of the previous code is here.> 

THOUGHT 1:
To solve for the length of the AB side of the ABCDE shape, we will proceed as follows:
1. **Compute the side length of the square BFGC** using the diagonal. The diagonal of the square is given as 24. Using the Pythagorean theorem:
   \[
   \text{Diagonal} = \sqrt{2} \times \text{Side}
   \]
   Solve for the side length.
2. **Use the perimeter of the ABCDE shape** to find the length of AB. The perimeter is given as 90:
   We have already computed the side length of the square BFGC, and we know that the perimeter of the ABCDE shape is 90. Let's compute the AB side length based on them.
ACTION 1:
```python
equation_diagnal = f"{math.sqrt(2)} * x = 24"
side_of_BFGC = solve_equation(equation_diagnal)

# we know the length of the side opposite to the shared side, so we call the tool like this:
equation_primeter = primeter_of_rectangle_with_equilateral_triangle(length_of_shared_side=side_of_BFGC, length_of_other_side='x', result=90)
side_of_AB = solve_equation(equation_primeter)
side_of_AB
```
OBSERVATION: Execution success. The output is as follows:
19.47
Please generate the next THOUGHT and ACTION. If you can get the answer, please also reply with ANSWER: <your answer> and ends with TERMINATE.
THOUGHT 2:
The computations are complete, and we have found the length of the AB side of the ABCDE shape.
ANSWER: The length of the AB side of the ABCDE shape is 19.47. TERMINATE.

'''


class GeoPromptVisuoThink:

    """
    This prompt generator is to generate auxiliary lines step by step.
    """
    def __init__(self) -> None:
        self.solve_prompt = """
In your THOUGHT, you must gather the information, find the relationships, but not solve the problem diretly. In your ACTION, you should try to use the given tools or directly write equations, then call the `solve_equation` function once or more times to solve the problem. 
If this problem has provide primeter or area of:
- rectangle with a equilateral triangle removed
- rectangle with a equilateral triangle combined
- rectangle with a semicircle removed
- rectangle with a semicircle combined
please use the tools like `primeter_of_rectangle_with_equilateral_triangle`,`area_of_rectangle_with_equilateral_triangle_removed` to construct the equations, and refer to the # REQUIREMENTS # [3] to use the tools correctly. Please not use the coordinates of points in the python code because they are not accurate.
If you can get the answer, please reply with ANSWER: <your answer> and ends with TERMINATE. If not, then please only provide your next THOUGHT and ACTION (You should not provide the OBSERVATION). 
"""
        self.auxiliary_prompt = """
Propose matplotlib code to draw one additional auxiliary line. You need to draw at least one auxiliary line. If you think no more auxiliary lines needed after drawing this one, please end your THOUGHT with AUX-END.
Please only provide your next THOUGHT and ACTION (You should not provide the OBSERVATION). 
"""
    
    def initial_prompt(self, ex, n_images: int) -> str:
        prompt = INIT_PROMPT
        
        # test example
        question = ex["problem_text"]
        diagram_logic_form = ex["logic_form"]["diagram_logic_form"]
        image_path_code = ex["image_path_code"]
        code = ex["code"]

        prompt += f"USER REQUEST #: Given the geometry diagram <img src='{image_path_code}'> and the diagram logic form {diagram_logic_form}\n" + \
        f"Below is the matplotlib code of the geometry: \n```python\n{code}\n```\n Only the angles marked with right-angle marker are right angles. You must solve the following question: [{question}]\n"
        prompt += self.auxiliary_prompt + "\n# RESULT #:\n"

        return prompt
    
    def get_parsing_feedback(self, error_message: str, error_code: str) -> str:
        return f"OBSERVATION: Parsing error. Error code: {error_code}, Error message:\n{error_message}\nPlease fix the error and generate the fixed code, in the next THOUGHT and ACTION."
    
    def get_exec_feedback(self, exit_code: int, output: str, aux_end: bool) -> str:
        # if execution fails
        if exit_code != 0:
           return f"OBSERVATION: Execution error. Exit code: {exit_code}, Output:\n{output}\nPlease fix the error and generate the fixed code, in the next THOUGHT and ACTION."
        else:
            prompt = f"OBSERVATION: Execution success. The output is as follows:\n{output}\n"
            if aux_end:
                prompt += self.solve_prompt # solve the problem
            else:
                prompt += self.auxiliary_prompt # draw auxiliary lines
            return prompt

