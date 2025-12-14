import sys
import importlib.util
import pytest
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
verify = execution_module.verify


# The curriculum to test - contains a bug: .isdigit() is called on an int
# sum() returns an int, but isdigit() is a string method
CURRICULUM_CODE = """
constraints = [
    lambda x: x % 7 == 6,
    lambda x: bin(x).count('1') == 2,
    lambda x: sum(int(d) for d in str(x)).isdigit() and sum(int(d) for d in str(x)) % 3 == 0
]
"""


def test_curriculum_returns_error_due_to_isdigit_on_int():
    """Test that the curriculum returns an error because .isdigit() is called on an int.
    
    The third constraint calls sum(...).isdigit(), but sum() returns an int,
    and int doesn't have an isdigit() method - that's a string method.
    """
    result = verify(CURRICULUM_CODE, 6)
    assert result.correct is False
    assert result.error is not None
    assert "AttributeError" in result.error
    assert "isdigit" in result.error


def test_curriculum_error_for_value_that_passes_first_two_constraints():
    """Test that even values passing first two constraints fail on the buggy third one.
    
    x=6 passes:
    - 6 % 7 == 6 ✓
    - bin(6) = '110' has 2 ones ✓
    But fails on third constraint due to AttributeError.
    """
    result = verify(CURRICULUM_CODE, 6)
    assert result.correct is False
    assert result.error is not None
    assert "AttributeError" in result.error


def test_curriculum_error_for_another_valid_candidate():
    """Test with x=48 which would pass first two constraints but fails on buggy third.
    
    x=48:
    - 48 % 7 == 6 ✓
    - bin(48) = '110000' has 2 ones ✓
    But fails on third constraint due to AttributeError.
    """
    result = verify(CURRICULUM_CODE, 48)
    assert result.correct is False
    assert result.error is not None
    assert "AttributeError" in result.error


def test_curriculum_fails_early_on_first_constraint():
    """Test with x=5 which fails the first constraint (no error, just incorrect).
    
    x=5: 5 % 7 == 5 ✗ (needs to be 6)
    Fails before reaching the buggy third constraint.
    """
    result = verify(CURRICULUM_CODE, 5)
    assert result.correct is False
    # No error because it fails on first constraint check (returns False, not exception)
    assert result.error is None
    # Feedback should indicate which constraint failed
    assert result.feedback is not None
    assert "not satisfied" in result.feedback


def test_curriculum_fails_early_on_second_constraint():
    """Test with x=13 which passes first but fails second constraint.
    
    x=13:
    - 13 % 7 == 6 ✓
    - bin(13) = '1101' has 3 ones ✗ (needs exactly 2)
    Fails before reaching the buggy third constraint.
    """
    result = verify(CURRICULUM_CODE, 13)
    assert result.correct is False
    # No error because it fails on second constraint check (returns False, not exception)
    assert result.error is None
    # Feedback should indicate which constraint failed
    assert result.feedback is not None
    assert "not satisfied" in result.feedback


def test_curriculum_no_solutions_due_to_bug():
    """Test that no values pass because the third constraint always raises an error."""
    passing_values = []
    error_values = []
    
    for x in range(0, 100):
        result = verify(CURRICULUM_CODE, x)
        if result.correct:
            passing_values.append(x)
        elif result.error is not None and "AttributeError" in result.error:
            error_values.append(x)
    
    # No values should pass due to the buggy constraint
    assert len(passing_values) == 0, f"Expected no passing values due to bug, but found: {passing_values}"
    
    # Values that pass first two constraints should hit the AttributeError
    # These are values where x % 7 == 6 AND bin(x).count('1') == 2
    assert len(error_values) > 0, "Expected some values to hit the AttributeError"


# Test with a valid curriculum to verify feedback on success
VALID_CURRICULUM_CODE = """
# Write Python code to derive a particular value of "x" which meets all constraints imposed:
constraints = [
    lambda x: x % 7 == 6,
    lambda x: bin(x).count('1') == 2,
]
"""


def test_valid_curriculum_passes_with_feedback():
    """Test that a valid curriculum passes and provides success feedback."""
    result = verify(VALID_CURRICULUM_CODE, 6)
    assert result.correct is True
    assert result.error is None
    assert result.feedback is not None
    assert "All constraints are satisfied" in result.feedback


def test_valid_curriculum_fails_with_feedback():
    """Test that a valid curriculum fails with appropriate feedback."""
    result = verify(VALID_CURRICULUM_CODE, 5)
    assert result.correct is False
    assert result.error is None
    assert result.feedback is not None
    assert "not satisfied" in result.feedback
