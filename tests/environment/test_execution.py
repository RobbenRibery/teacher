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
