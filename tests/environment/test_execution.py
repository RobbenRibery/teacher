import sys
import importlib.util
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

# Import using importlib to ensure it works
spec = importlib.util.spec_from_file_location(
    "environment.execution",
    project_root / "environment" / "execution.py"
)
execution_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(execution_module)
verify_answer = execution_module.verify_answer
extract_answer = execution_module.extract_answer
execute_answer = execution_module.execute_answer

def test_verify_answer_success_simple():
    """Test simple boolean constraints that should pass."""
    # x > 5
    curriculum = "x > 5"
    result = verify_answer(curriculum, 6, "x")
    assert result.correct is True
    assert result.error is None

def test_verify_answer_failure_simple():
    """Test simple boolean constraints that should fail."""
    # x > 5
    curriculum = "x > 5"
    result = verify_answer(curriculum, 4, "x")
    assert result.correct is False
    assert result.error is None

def test_verify_answer_complex_logic():
    """Test complex boolean logic."""
    # (x > 0) and (x < 10 or x == 20)
    curriculum = "(x > 0) and (x < 10 or x == 20)"
    
    # Case 1: x = 5 (True and True) -> True
    result = verify_answer(curriculum, 5, "x")
    assert result.correct is True

    # Case 2: x = 20 (True and True) -> True
    result = verify_answer(curriculum, 20, "x")
    assert result.correct is True

    # Case 3: x = 15 (True and False) -> False
    result = verify_answer(curriculum, 15, "x")
    assert result.correct is False
    
    # Case 4: x = -1 (False and ...) -> False
    result = verify_answer(curriculum, -1, "x")
    assert result.correct is False

def test_verify_answer_different_variable_name():
    """Test with a custom variable name."""
    curriculum = "y * 2 == 10"
    result = verify_answer(curriculum, 5, variable_name="y")
    assert result.correct is True
    
    result = verify_answer(curriculum, 4, variable_name="y")
    assert result.correct is False

def test_verify_answer_types():
    """Test with different types (float, list, string)."""
    # Float
    curriculum = "x > 3.14"
    result = verify_answer(curriculum, 3.15, "x")
    assert result.correct is True

    # String
    curriculum = "'hello' in x"
    result = verify_answer(curriculum, "hello world", "x")
    assert result.correct is True

    # List
    curriculum = "len(x) == 3"
    result = verify_answer(curriculum, [1, 2, 3], "x")
    assert result.correct is True

def test_verify_answer_syntax_error():
    """Test handling of syntax errors in curriculum."""
    curriculum = "x > " # Incomplete
    result = verify_answer(curriculum, 5, "x")
    assert result.correct is False
    assert result.error is not None
    assert "SyntaxError" in result.error

def test_verify_answer_runtime_error():
    """Test handling of runtime errors (e.g. division by zero)."""
    curriculum = "1 / x > 0"
    result = verify_answer(curriculum, 0, "x")
    assert result.correct is False
    assert result.error is not None
    assert "ZeroDivisionError" in result.error

def test_verify_answer_undefined_variable():
    """Test when curriculum uses variables not in context."""
    curriculum = "x > y" # y is undefined
    result = verify_answer(curriculum, 5, "x")
    assert result.correct is False
    assert result.error is not None
    assert "NameError" in result.error


# Test cases for extract_answer function
# Regex format: matches ```python followed by anything until ```
def test_extract_answer_simple_code_block():
    """Test extracting a simple Python code block."""
    regex_format = r'```python.*?```'
    solver_generation = "Here is the answer:\n```python\nx = 5\n```\nThat's it!"
    result = extract_answer(solver_generation, regex_format)
    assert result is not None
    assert "x = 5" in result
    assert result.startswith("```python")
    assert result.endswith("```")

def test_extract_answer_multiline_code():
    """Test extracting multiline Python code."""
    regex_format = r'```python.*?```'
    solver_generation = """
    The solution is:
    ```python
    x = 10
    y = x * 2
    result = y + 5
    ```
    That's the answer.
    """
    result = extract_answer(solver_generation, regex_format)
    assert result is not None
    assert "x = 10" in result
    assert "y = x * 2" in result
    assert "result = y + 5" in result

def test_extract_answer_first_match():
    """Test that extract_answer returns the first match."""
    regex_format = r'```python.*?```'
    solver_generation = """
    First code block:
    ```python
    x = 1
    ```
    Second code block:
    ```python
    x = 2
    ```
    """
    result = extract_answer(solver_generation, regex_format)
    assert result is not None
    assert "x = 1" in result
    assert "x = 2" not in result

def test_extract_answer_no_match():
    """Test when no code block is found."""
    regex_format = r'```python.*?```'
    solver_generation = "This is just plain text with no code blocks."
    result = extract_answer(solver_generation, regex_format)
    assert result is None

def test_extract_answer_with_other_code_blocks():
    """Test that it only matches Python code blocks, not other languages."""
    regex_format = r'```python.*?```'
    solver_generation = """
    Here's some JavaScript:
    ```javascript
    var x = 5;
    ```
    And here's Python:
    ```python
    x = 5
    ```
    """
    result = extract_answer(solver_generation, regex_format)
    assert result is not None
    assert "x = 5" in result
    assert "javascript" not in result

def test_extract_answer_with_newlines_in_code():
    """Test extracting code that contains multiple newlines."""
    regex_format = r'```python.*?```'
    solver_generation = """
    ```python
    def calculate():
        x = 5
        y = 10
        return x + y
    ```
    """
    result = extract_answer(solver_generation, regex_format)
    assert result is not None
    assert "def calculate():" in result
    assert "x = 5" in result
    assert "y = 10" in result
    assert "return x + y" in result

def test_extract_answer_empty_code_block():
    """Test extracting an empty code block."""
    regex_format = r'```python.*?```'
    solver_generation = "```python\n\n```"
    result = extract_answer(solver_generation, regex_format)
    assert result is not None
    assert "```python" in result
    assert "```" in result

def test_extract_answer_with_text_before_and_after():
    """Test extracting code block with extensive text before and after."""
    regex_format = r'```python.*?```'
    solver_generation = """
    This is a long explanation of the problem.
    We need to solve for x.
    After careful consideration, here's my solution:
    
    ```python
    x = 42
    ```
    
    I hope this helps! Let me know if you have questions.
    """
    result = extract_answer(solver_generation, regex_format)
    assert result is not None
    assert "x = 42" in result

def test_extract_answer_dotall_flag():
    """Test that re.DOTALL flag works correctly (matches across newlines)."""
    regex_format = r'```python.*?```'
    solver_generation = "```python\nx = 5\ny = 10\n```"
    result = extract_answer(solver_generation, regex_format)
    assert result is not None
    assert "x = 5" in result
    assert "y = 10" in result


# Test cases for execute_answer function
def test_execute_answer_simple_assignment():
    """Test executing simple variable assignment."""
    extracted_answer = "x = 5"
    result = execute_answer(extracted_answer)
    assert result == 5

def test_execute_answer_computation():
    """Test executing code that computes the value."""
    extracted_answer = "x = 10 * 2"
    result = execute_answer(extracted_answer)
    assert result == 20

def test_execute_answer_complex_computation():
    """Test executing complex computation."""
    extracted_answer = "x = (5 + 3) * 2 - 1"
    result = execute_answer(extracted_answer)
    assert result == 15

def test_execute_answer_with_function():
    """Test executing a function that computes x."""
    extracted_answer = """
def compute_x():
    return 42

x = compute_x()
"""
    result = execute_answer(extracted_answer)
    assert result == 42

def test_execute_answer_multiple_statements():
    """Test executing multiple statements that compute x."""
    extracted_answer = """
a = 5
b = 10
x = a + b
"""
    result = execute_answer(extracted_answer)
    assert result == 15

def test_execute_answer_with_loops():
    """Test executing code with loops that compute x."""
    extracted_answer = """
total = 0
for i in range(1, 6):
    total += i
x = total
"""
    result = execute_answer(extracted_answer)
    assert result == 15

def test_execute_answer_with_conditionals():
    """Test executing code with conditionals that compute x."""
    extracted_answer = """
value = 10
if value > 5:
    x = value * 2
else:
    x = value
"""
    result = execute_answer(extracted_answer)
    assert result == 20

def test_execute_answer_different_variable_name():
    """Test executing code with a different variable name."""
    extracted_answer = "y = 100"
    result = execute_answer(extracted_answer, variable_name="y")
    assert result == 100

def test_execute_answer_variable_not_found():
    """Test when the variable is not found (should return None)."""
    extracted_answer = "y = 5"  # x is not defined
    result = execute_answer(extracted_answer)
    assert result is None

def test_execute_answer_string_value():
    """Test executing code that assigns a string value."""
    extracted_answer = 'x = "hello"'
    result = execute_answer(extracted_answer)
    assert result == "hello"

def test_execute_answer_list_value():
    """Test executing code that assigns a list value."""
    extracted_answer = "x = [1, 2, 3, 4, 5]"
    result = execute_answer(extracted_answer)
    assert result == [1, 2, 3, 4, 5]

def test_execute_answer_float_value():
    """Test executing code that assigns a float value."""
    extracted_answer = "x = 3.14159"
    result = execute_answer(extracted_answer)
    assert result == 3.14159

def test_execute_answer_boolean_value():
    """Test executing code that assigns a boolean value."""
    extracted_answer = "x = True"
    result = execute_answer(extracted_answer)
    assert result is True

def test_execute_answer_none_value():
    """Test executing code that assigns None."""
    extracted_answer = "x = None"
    result = execute_answer(extracted_answer)
    assert result is None

def test_execute_answer_with_imports():
    """Test executing code that uses imports."""
    extracted_answer = """
import math
x = math.pi
"""
    result = execute_answer(extracted_answer)
    assert abs(result - 3.141592653589793) < 1e-10

def test_execute_answer_nested_function():
    """Test executing code with nested function definitions."""
    extracted_answer = """
def outer():
    def inner():
        return 7
    return inner()

x = outer()
"""
    result = execute_answer(extracted_answer)
    assert result == 7

def test_execute_answer_empty_code():
    """Test executing empty code (should return None)."""
    extracted_answer = ""
    result = execute_answer(extracted_answer)
    assert result is None

def test_execute_answer_only_comments():
    """Test executing code with only comments (should return None)."""
    extracted_answer = "# This is just a comment\n# Another comment"
    result = execute_answer(extracted_answer)
    assert result is None
