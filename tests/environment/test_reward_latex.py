import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from environment.reward import reward_answer
from environment.format import STUDENT_PATTERN

def test_reward_answer_latex_fraction():
    """Test that \frac{26}{1} is correctly handled and satisfies constraints."""
    curriculum_code = "constraints = [lambda x: x == 26, lambda x: x % 2 == 0]"
    # Student outputs \boxed{\frac{26}{1}}
    answer_generation = "<think>t</think>The answer is \\boxed{\\frac{26}{1}}"
    
    result = reward_answer(
        extracted_curriculum=curriculum_code,
        answer_generation=answer_generation,
    )
    
    assert result.passed is True
    assert result.reward == 1.0
    assert result.error is None

def test_reward_answer_latex_sqrt():
    """Test that \sqrt{25} is correctly handled."""
    curriculum_code = "constraints = [lambda x: x == 5]"
    answer_generation = "<think>t</think>\\boxed{\\sqrt{25}}"
    
    result = reward_answer(
        extracted_curriculum=curriculum_code,
        answer_generation=answer_generation,
    )
    
    assert result.passed is True
    assert result.reward == 1.0

def test_reward_answer_latex_nested_sqrt():
    """Test that \sqrt{5} satisfies x^2 == 5 (symbolic comparison or float)."""
    curriculum_code = "constraints = [lambda x: abs(x**2 - 5) < 1e-9]"
    answer_generation = "<think>t</think>\\boxed{\\sqrt{5}}"
    
    result = reward_answer(
        extracted_curriculum=curriculum_code,
        answer_generation=answer_generation,
    )
    
    assert result.passed is True
    assert result.reward == 1.0

def test_reward_answer_boolean():
    """Test that standard eval still works for non-math things like True/False."""
    curriculum_code = "constraints = [lambda x: x is True]"
    answer_generation = "<think>t</think>\\boxed{True}"
    
    result = reward_answer(
        extracted_curriculum=curriculum_code,
        answer_generation=answer_generation,
    )
    
    assert result.passed is True
    assert result.reward == 1.0

def test_reward_answer_latex_already_wrapped():
    """Test that $ \sqrt{25} $ inside \boxed is correctly handled."""
    curriculum_code = "constraints = [lambda x: x == 5]"
    # Student outputs \boxed{$\sqrt{25}$}
    answer_generation = "<think>t</think>\\boxed{$\\sqrt{25}$}"
    
    result = reward_answer(
        extracted_curriculum=curriculum_code,
        answer_generation=answer_generation,
    )
    
    assert result.passed is True
    assert result.reward == 1.0

def test_reward_answer_sympy_constraint():
    """Test that sympy functions can be used in constraints."""
    # Constraint uses sympy.sqrt which is available in the exec environment
    curriculum_code = "constraints = [lambda x: x == sqrt(5)]"
    answer_generation = "<think>t</think>\\boxed{\\sqrt{5}}"
    
    result = reward_answer(
        extracted_curriculum=curriculum_code,
        answer_generation=answer_generation,
    )
    
    assert result.passed is True
    assert result.reward == 1.0

def test_reward_answer_sympy_simplify_in_constraint():
    """Test that sympy.simplify works in constraints."""
    curriculum_code = "constraints = [lambda x: simplify(x**2 - 5) == 0]"
    answer_generation = "<think>t</think>\\boxed{\\sqrt{5}}"
    
    result = reward_answer(
        extracted_curriculum=curriculum_code,
        answer_generation=answer_generation,
    )
    
    assert result.passed is True
    assert result.reward == 1.0

