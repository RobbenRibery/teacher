import sys
import textwrap
import time
import pytest
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from environment.execution import verify, execute_answer, time_limit, TimeoutException
from environment.format import extract_format as extract_answer, CODE_BLOCK_REGEX


def test_verify_success_simple():
    """Test simple boolean constraints that should pass."""
    # x > 5
    curriculum = "constraints = [lambda x: x > 5]"
    result = verify(curriculum, 6)
    assert result.correct is True
    assert result.error is None
    assert result.feedback is not None
    assert "All constraints are satisfied" in result.feedback


def test_verify_failure_simple():
    """Test simple boolean constraints that should fail."""
    # x > 5
    curriculum = "constraints = [lambda x: x > 5]"
    result = verify(curriculum, 4)
    assert result.correct is False
    assert result.error is None
    assert result.feedback is not None
    assert "not satisfied" in result.feedback
    assert "lambda x: x > 5" in result.feedback


def test_verify_complex_logic():
    """Test complex boolean logic."""
    # (x > 0) and (x < 10 or x == 20)
    curriculum = "constraints = [lambda x: (x > 0) and (x < 10 or x == 20)]"

    # Case 1: x = 5 (True and True) -> True
    result = verify(curriculum, 5)
    assert result.correct is True

    # Case 2: x = 20 (True and True) -> True
    result = verify(curriculum, 20)
    assert result.correct is True

    # Case 3: x = 15 (True and False) -> False
    result = verify(curriculum, 15)
    assert result.correct is False

    # Case 4: x = -1 (False and ...) -> False
    result = verify(curriculum, -1)
    assert result.correct is False


def test_verify_types():
    """Test with different types (float, list, string)."""
    # Float
    curriculum = "constraints = [lambda x: x > 3.14]"
    result = verify(curriculum, 3.15)
    assert result.correct is True

    # String
    curriculum = "constraints = [lambda x: 'hello' in x]"
    result = verify(curriculum, "hello world")
    assert result.correct is True

    # List
    curriculum = "constraints = [lambda x: len(x) == 3]"
    result = verify(curriculum, [1, 2, 3])
    assert result.correct is True


def test_verify_syntax_error():
    """Test handling of syntax errors in curriculum."""
    curriculum = "constraints = [lambda x: x > ]"  # Incomplete
    result = verify(curriculum, 5)
    assert result.correct is False
    assert result.error is not None
    assert "SyntaxError" in result.error


def test_verify_runtime_error():
    """Test handling of runtime errors (e.g. division by zero)."""
    curriculum = "constraints = [lambda x: 1 / x > 0]"
    result = verify(curriculum, 0)
    assert result.correct is False
    assert result.error is not None
    assert "ZeroDivisionError" in result.error


def test_verify_undefined_variable():
    """Test when curriculum uses variables not in context."""
    curriculum = "constraints = [lambda x: x > y]"  # y is undefined
    result = verify(curriculum, 5)
    assert result.correct is False
    assert result.error is not None
    assert "NameError" in result.error


def test_verify_multiple_constraints():
    """Test with multiple constraints."""
    curriculum = "constraints = [lambda x: x > 0, lambda x: x < 10]"
    result = verify(curriculum, 5)
    assert result.correct is True
    assert result.feedback is not None
    assert "All constraints are satisfied" in result.feedback

    result = verify(curriculum, 15)
    assert result.correct is False
    assert result.feedback is not None
    assert "not satisfied" in result.feedback


def test_verify_variable_name_validation():
    """Test validation of variable names in constraints."""
    # Invalid: constraint argument named 'y' instead of 'x'
    curriculum = "constraints = [lambda y: y > 0]"
    result = verify(curriculum, 5)
    assert result.correct is False
    assert result.error is not None
    assert "Constraint argument must be named 'x'" in result.error
    assert "found 'y'" in result.error

    # Invalid: multiple arguments
    curriculum = "constraints = [lambda x, y: x > y]"
    result = verify(curriculum, 5)
    assert result.correct is False
    assert result.error is not None
    assert "Each constraint must take exactly one argument" in result.error

    # Invalid: mixed valid and invalid variable names
    curriculum = "constraints = [lambda x: x > 0, lambda y: y < 10]"
    result = verify(curriculum, 5)
    assert result.correct is False
    assert result.error is not None
    assert "Constraint argument must be named 'x'" in result.error


# Test cases for extract_answer function
# Regex format: matches ```python followed by anything until ```
def test_extract_answer_simple_code_block():
    """Test extracting a simple Python code block."""
    regex_format = r"```python\n(.*?)\n```"
    solver_generation = "Here is the answer:\n```python\nx = 5\n```\nThat's it!"
    result = extract_answer(solver_generation, regex_format)
    assert result is not None
    assert "x = 5" in result
    assert "```" not in result


def test_extract_answer_multiline_code():
    """Test extracting multiline Python code."""
    regex_format = r"```python\n(.*?)\n```"
    solver_generation = textwrap.dedent("""
    The solution is:
    ```python
    x = 10
    y = x * 2
    result = y + 5
    ```
    That's the answer.
    """)
    result = extract_answer(solver_generation, regex_format)
    assert result is not None
    assert "x = 10" in result
    assert "y = x * 2" in result
    assert "result = y + 5" in result


def test_extract_answer_last_match():
    """Test that extract_answer returns the last match."""
    regex_format = r"```python\n(.*?)\n```"
    solver_generation = textwrap.dedent("""
    First code block:
    ```python
    x = 1
    ```
    Second code block:
    ```python
    x = 2
    ```
    """)
    result = extract_answer(solver_generation, regex_format)
    assert result is not None
    assert "x = 2" in result
    assert "x = 1" not in result


def test_extract_answer_no_match():
    """Test when no code block is found."""
    regex_format = r"```python\n(.*?)\n```"
    solver_generation = "This is just plain text with no code blocks."
    result = extract_answer(solver_generation, regex_format)
    assert result is None


def test_extract_answer_with_other_code_blocks():
    """Test that it only matches Python code blocks, not other languages."""
    regex_format = r"```python\n(.*?)\n```"
    solver_generation = textwrap.dedent("""
    Here's some JavaScript:
    ```javascript
    var x = 5;
    ```
    And here's Python:
    ```python
    x = 5
    ```
    """)
    result = extract_answer(solver_generation, regex_format)
    assert result is not None
    assert "x = 5" in result
    assert "javascript" not in result


def test_extract_answer_dotall_flag():
    """Test that re.DOTALL flag works correctly (matches across newlines)."""
    regex_format = r"```python\n(.*?)\n```"
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
    assert result.value == 5
    assert result.error is None


def test_execute_answer_computation():
    """Test executing code that computes the value."""
    extracted_answer = "x = 10 * 2"
    result = execute_answer(extracted_answer)
    assert result.value == 20


def test_execute_answer_complex_computation():
    """Test executing complex computation."""
    extracted_answer = "x = (5 + 3) * 2 - 1"
    result = execute_answer(extracted_answer)
    assert result.value == 15


def test_execute_answer_with_function():
    """Test executing a function that computes x."""
    extracted_answer = """
def compute_x():
    return 42

x = compute_x()
"""
    result = execute_answer(extracted_answer)
    assert result.value == 42


def test_execute_answer_multiple_statements():
    """Test executing multiple statements that compute x."""
    extracted_answer = """
a = 5
b = 10
x = a + b
"""
    result = execute_answer(extracted_answer)
    assert result.value == 15


def test_execute_answer_with_loops():
    """Test executing code with loops that compute x."""
    extracted_answer = """
total = 0
for i in range(1, 6):
    total += i
x = total
"""
    result = execute_answer(extracted_answer)
    assert result.value == 15


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
    assert result.value == 20


def test_execute_answer_different_variable_name():
    """Test executing code with a different variable name."""
    extracted_answer = "y = 100"
    result = execute_answer(extracted_answer, variable_name="y")
    assert result.value == 100


def test_execute_answer_variable_not_found():
    """Test when the variable is not found (should return None value and error)."""
    extracted_answer = "y = 5"  # x is not defined
    result = execute_answer(extracted_answer)
    assert result.value is None
    assert result.error is not None
    assert "not found" in result.error


def test_execute_answer_string_value():
    """Test executing code that assigns a string value."""
    extracted_answer = 'x = "hello"'
    result = execute_answer(extracted_answer)
    assert result.value == "hello"


def test_execute_answer_list_value():
    """Test executing code that assigns a list value."""
    extracted_answer = "x = [1, 2, 3, 4, 5]"
    result = execute_answer(extracted_answer)
    assert result.value == [1, 2, 3, 4, 5]


def test_execute_answer_float_value():
    """Test executing code that assigns a float value."""
    extracted_answer = "x = 3.14159"
    result = execute_answer(extracted_answer)
    assert result.value == 3.14159


def test_execute_answer_boolean_value():
    """Test executing code that assigns a boolean value."""
    extracted_answer = "x = True"
    result = execute_answer(extracted_answer)
    assert result.value is True


def test_execute_answer_none_value():
    """Test executing code that assigns None."""
    extracted_answer = "x = None"
    result = execute_answer(extracted_answer)
    assert result.value is None
    # If x is explicitly None, it's found, so error should be None
    assert result.error is None


def test_execute_answer_with_imports():
    """Test executing code that uses imports."""
    extracted_answer = """
import math
x = math.pi
"""
    result = execute_answer(extracted_answer)
    assert abs(result.value - 3.141592653589793) < 1e-10


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
    assert result.value == 7


def test_execute_answer_empty_code():
    """Test executing empty code (should return None value and error)."""
    extracted_answer = ""
    result = execute_answer(extracted_answer)
    assert result.value is None
    assert result.error is not None


def test_execute_answer_only_comments():
    """Test executing code with only comments (should return None value and error)."""
    extracted_answer = "# This is just a comment\n# Another comment"
    result = execute_answer(extracted_answer)
    assert result.value is None
    assert result.error is not None


def test_execute_answer_runtime_error():
    """Test executing code that raises runtime error."""
    extracted_answer = "x = 1 / 0"
    result = execute_answer(extracted_answer)
    assert result.value is None
    assert result.error is not None
    assert "ZeroDivisionError" in result.error


def test_exec_globals_available():
    """Test that math, random etc are available via default globals."""
    code = "x = math.pi"
    result = execute_answer(code)
    assert abs(result.value - 3.141592653589793) < 1e-10

    code = "import collections; x = collections.Counter([1,1,2])"
    result = execute_answer(code)
    assert result.value[1] == 2


def test_timeout_mechanism():
    """Test the timeout mechanism via context manager directly."""
    # Test that it raises TimeoutException
    with pytest.raises(TimeoutException):
        with time_limit(1):
            time.sleep(2)

    # Test that it doesn't raise if faster
    with time_limit(2):
        time.sleep(0.1)


def test_execute_answer_timeout():
    """Test that execute_answer times out on infinite loops."""
    # Note: execute_answer has hardcoded 30s timeout.
    # We can't easily test the 30s timeout without waiting 30s.
    # But we can verify it doesn't crash immediately or fail for valid code.
    pass
