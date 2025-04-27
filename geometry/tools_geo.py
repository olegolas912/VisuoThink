import numpy as np
import sympy
import math

def find_perpendicular_intersection(A, B, C):
    # Convert coordinates to numpy arrays for easier computation
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    
    # Calculate the direction vector of line BC
    BC = C - B
    
    # Compute the slope of BC if not vertical
    if BC[0] != 0:
        slope_BC = BC[1] / BC[0]
        # Slope of the perpendicular line from A to BC
        slope_perpendicular = -1 / slope_BC
    else:
        # If line BC is vertical, then perpendicular line is horizontal
        slope_perpendicular = 0
    
    # Calculate the equation of the line passing through A and perpendicular to BC
    # y - y_A = slope_perpendicular * (x - x_A)
    # Rearrange to standard form Ax + By + C = 0
    if BC[0] != 0:
        A_coeff = -slope_perpendicular
        B_coeff = 1
        C_coeff = -A_coeff * A[0] - B_coeff * A[1]
    else:
        # If BC is vertical, AE must be horizontal
        A_coeff = 1
        B_coeff = 0
        C_coeff = -A[0]
    
    # Equation of line BC: (y - y_B) = slope_BC * (x - x_B)
    # Convert to Ax + By + C = 0 for line intersection calculation
    if BC[0] != 0:
        A_BC = -slope_BC
        B_BC = 1
        C_BC = -A_BC * B[0] - B_BC * B[1]
    else:
        # BC is vertical, so x = constant
        A_BC = 1
        B_BC = 0
        C_BC = -B[0]
    
    # Solve the linear system of equations representing the two lines
    # [A_coeff B_coeff] [x] = [-C_coeff]
    # [A_BC    B_BC   ] [y]   [-C_BC  ]
    matrix = np.array([[A_coeff, B_coeff], [A_BC, B_BC]])
    constants = np.array([-C_coeff, -C_BC])
    
    # Use numpy to solve the linear system
    intersection = np.linalg.solve(matrix, constants)
    return intersection.tolist()


# this function takes a coordinate A, start and end points of a line BC, and returns the coordinates of the point E on BC such that AE is parallel to BC
def find_parallel_intersection(A, B, C):
    # Convert coordinates to numpy arrays for vector operations
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    
    # Calculate the direction vector of line BC
    direction_BC = C - B
    
    # Since AE is parallel to BC, its direction vector is the same as BC
    direction_AE = direction_BC
    
    # To find a reasonable "point E", you can just extend AE from A by some length.
    # For visualization, let's extend by the length of BC
    length_BC = np.linalg.norm(direction_BC)
    
    # Normalize the direction vector of AE
    direction_AE_normalized = direction_AE / np.linalg.norm(direction_AE)
    
    # Point E can be found by moving from A in the direction of AE by the length of BC
    E = A + direction_AE_normalized * length_BC
    
    return E.tolist()


def solve_equation(equation: str) -> float:
    """
    Solve a math equation about variable x and return the value of x.
    """
    if isinstance(equation, float) or isinstance(equation, int):
        return equation

    assert isinstance(equation, str), "Equation must be a string"
    assert "x" in equation, "Equation must contain the variable x"
    # only one equal sign
    assert equation.count("=") == 1, "Equation must contain, and contain only one equal sign"

    equation = equation.replace("=", "-") # for sympy, the equation is in the form of "x = 1 -> x - 1"

    # solve the equation with sympy
    x = sympy.symbols('x')
    expr = sympy.sympify(equation)
    solution = sympy.solve(expr, x)
    # first, only keep the real number solution
    solution = [s for s in solution if s.is_real]

    # if there are multiple solutions, return the maximum one (the positive one)
    return float(max(solution))


def primeter_of_rectangle_with_equilateral_triangle(length_of_shared_side: float | str, length_of_other_side: float | str, result=None) -> float | str:
    """
    Given a rectangle with a equilateral triangle sharing a side. 

    If result is not None, return the primeter equation, otherwise return the value of the primeter.
    """
    if isinstance(length_of_shared_side, int):
        length_of_shared_side = float(length_of_shared_side)
    if isinstance(length_of_other_side, int):
        length_of_other_side = float(length_of_other_side)

    if result is not None:
        assert isinstance(result, float) or isinstance(result, int), "Result must be a number if you provide it."
        assert length_of_shared_side == 'x' or length_of_other_side == 'x', "Length of shared side or length of other side must be 'x' if you provide result."
        equation = f"3 * {length_of_shared_side} + 2 * {length_of_other_side} = {result}"
        return equation
    else:
        assert isinstance(length_of_shared_side, float) and isinstance(length_of_other_side, float), "Length of shared side and length of other side must be numbers if you don't provide result."
        return 3 * length_of_shared_side + 2 * length_of_other_side


def area_of_rectangle_with_equilateral_triangle_removed(length_of_shared_side: float | str, length_of_other_side: float | str, result=None) -> float | str:
    """
    Given a rectangle with a equilateral triangle sharing a side. 

    If result is not None, return the area equation, otherwise return the value of the area.
    """
    if isinstance(length_of_shared_side, int):
        length_of_shared_side = float(length_of_shared_side)
    if isinstance(length_of_other_side, int):
        length_of_other_side = float(length_of_other_side)

    if result is not None:
        assert isinstance(result, float) or isinstance(result, int), "Result must be a number if you provide it."
        assert length_of_shared_side == 'x' or length_of_other_side == 'x', "Length of shared side or length of other side must be 'x' if you provide result."
        equation = f"{length_of_shared_side} * {length_of_other_side} - {math.sqrt(3)} / 4 * {length_of_shared_side} ** 2 = {result}"
        return equation
    else:
        assert isinstance(length_of_shared_side, float) and isinstance(length_of_other_side, float), "Length of shared side and length of other side must be numbers if you don't provide result."
        return length_of_shared_side * length_of_other_side - math.sqrt(3) / 4 * length_of_shared_side ** 2


def area_of_rectangle_with_equilateral_triangle_combined(length_of_shared_side: float | str, length_of_other_side: float | str, result=None) -> float | str:
    """
    Given a rectangle with a equilateral triangle sharing a side. 

    If result is not None, return the area equation, other wise return the value of the area.
    """
    if isinstance(length_of_shared_side, int):
        length_of_shared_side = float(length_of_shared_side)
    if isinstance(length_of_other_side, int):
        length_of_other_side = float(length_of_other_side)

    if result is not None:
        assert isinstance(result, float) or isinstance(result, int), "Result must be a number if you provide it."
        assert length_of_shared_side == 'x' or length_of_other_side == 'x', "Length of shared side or length of other side must be 'x' if you provide result."
        equation = f"{length_of_shared_side} * {length_of_other_side} + {math.sqrt(3)} / 4 * {length_of_shared_side} ** 2 = {result}"
        return equation
    else:
        assert isinstance(length_of_shared_side, float) and isinstance(length_of_other_side, float), "Length of shared side and length of other side must be numbers if you don't provide result."
        return length_of_shared_side * length_of_other_side + math.sqrt(3) / 4 * length_of_shared_side ** 2


def primeter_of_rectangle_with_semicircle(length_of_shared_side: float | str, length_of_other_side: float | str, result=None) -> float | str:
    """
    Given a rectangle with a semicircle sharing a side. 

    If result is not None, return the primeter equation, other wise return the value of the primeter.
    """
    if isinstance(length_of_shared_side, int):
        length_of_shared_side = float(length_of_shared_side)
    if isinstance(length_of_other_side, int):
        length_of_other_side = float(length_of_other_side)

    if result is not None:
        assert isinstance(result, float) or isinstance(result, int), "Result must be a number if you provide it."
        assert length_of_shared_side == 'x' or length_of_other_side == 'x', "Length of shared side or length of other side must be 'x' if you provide result."
        equation = f"2 * {length_of_other_side} + {length_of_shared_side} + {math.pi} * {length_of_shared_side} / 2 = {result}"
        return equation
    else:
        assert isinstance(length_of_shared_side, float) and isinstance(length_of_other_side, float), "Length of shared side and length of other side must be numbers if you don't provide result."
        return 2 * length_of_other_side + length_of_shared_side + math.pi * length_of_shared_side / 2


def area_of_rectangle_with_semicircle_removed(length_of_shared_side: float | str, length_of_other_side: float | str, result=None) -> float | str:
    """
    Given a rectangle with a semicircle sharing a side. 

    If result is not None, return the area equation, other wise return the value of the area.
    """
    if isinstance(length_of_shared_side, int):
        length_of_shared_side = float(length_of_shared_side)
    if isinstance(length_of_other_side, int):
        length_of_other_side = float(length_of_other_side)
    
    if result is not None:
        assert isinstance(result, float) or isinstance(result, int), "Result must be a number if you provide it."
        assert length_of_shared_side == 'x' or length_of_other_side == 'x', "Length of shared side or length of other side must be 'x' if you provide result."
        equation = f"{length_of_shared_side} * {length_of_other_side} - {math.pi} / 8 * {length_of_shared_side} ** 2 = {result}"
        return equation
    else:
        assert isinstance(length_of_shared_side, float) and isinstance(length_of_other_side, float), "Length of shared side and length of other side must be numbers if you don't provide result."
        # the area of shape: rectangle - semicircle = rectangle - 1/8 * pi * r^2
        return length_of_shared_side * length_of_other_side - math.pi / 8 * length_of_shared_side ** 2


def area_of_rectangle_with_semicircle_combined(length_of_shared_side: float | str, length_of_other_side: float | str, result=None) -> float | str:
    """
    Given a rectangle with a semicircle sharing a side. 

    If result is not None, return the area equation, other wise return the value of the area.
    """
    if isinstance(length_of_shared_side, int):
        length_of_shared_side = float(length_of_shared_side)
    if isinstance(length_of_other_side, int):
        length_of_other_side = float(length_of_other_side)

    if result is not None:
        assert isinstance(result, float) or isinstance(result, int), "Result must be a number if you provide it."
        assert length_of_shared_side == 'x' or length_of_other_side == 'x', "Length of shared side or length of other side must be 'x' if you provide result."
        equation = f"{length_of_shared_side} * {length_of_other_side} + {math.pi} / 8 * {length_of_shared_side} ** 2 = {result}"
        return equation
    else:
        assert isinstance(length_of_shared_side, float) and isinstance(length_of_other_side, float), "Length of shared side and length of other side must be numbers if you don't provide result."
        return length_of_shared_side * length_of_other_side + math.pi / 8 * length_of_shared_side ** 2


if __name__ == '__main__':
    eq = primeter_of_rectangle_with_equilateral_triangle(10, 'x', 56)
    # eq = area_of_rectangle_with_equilateral_triangle_removed(10, 'x', 100)
    # eq = area_of_rectangle_with_semicircle_combined('x', 10, 100)
    # print(eq)
    result = solve_equation(eq)
    print(result)

