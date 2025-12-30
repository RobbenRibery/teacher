import pathlib
import re
import sys
import pytest

project_root = pathlib.Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from environment.reward import reward_answer
from environment.format import TEACHER_PATTERN, STUDENT_PATTERN, CODE_BLOCK_REGEX


FIXTURES_DIR = pathlib.Path(__file__).resolve().parent.parent
VALID_PATH = FIXTURES_DIR / "valid_curriculum_generation.txt"
INVALID_PATH = FIXTURES_DIR / "invalid_curriculum_generation.txt"
CODE_REGEX = re.compile(CODE_BLOCK_REGEX, re.DOTALL)


def _read_fixture(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_curriculum_from_valid() -> str:
    match = CODE_REGEX.search(_read_fixture(VALID_PATH))
    assert match, "Valid fixture must contain python code block"
    return match.group(1)


def _make_answer_generation(value: str) -> str:
    """Create a student answer in the \\boxed{} format."""
    return f"<think>Let me solve this step by step...</think>\nThe answer is \\boxed{{{value}}}"


def test_reward_answer_success():
    curriculum_code = _extract_curriculum_from_valid()
    # x = 26 satisfies: even, (26+1)%3==0, (26-1)%5==0
    answer_generation = _make_answer_generation("26")

    result = reward_answer(
        extracted_curriculum=curriculum_code,
        answer_generation=answer_generation,
    )

    assert result.passed is True
    assert result.error is None
    assert result.extracted_answer == "26"


def test_reward_answer_format_failure():
    curriculum_code = _extract_curriculum_from_valid()
    # Invalid format - no \boxed{}
    answer_generation = "<think>thinking</think>\nx = 26"

    result = reward_answer(
        extracted_curriculum=curriculum_code,
        answer_generation=answer_generation,
    )

    assert result.passed is False
    assert result.reward == 0.0
    assert result.error is not None


def test_reward_answer_verification_failure():
    curriculum_code = _extract_curriculum_from_valid()
    # x = 4 does NOT satisfy the constraints (not even divisible requirement met)
    answer_generation = _make_answer_generation("4")

    result = reward_answer(
        extracted_curriculum=curriculum_code,
        answer_generation=answer_generation,
    )

    assert result.passed is False
    assert result.reward == 0.0
    # verification failure - constraint not satisfied
    assert result.error is None  # No execution error, just wrong answer


def test_reward_answer_negative_number():
    """Test that negative numbers are handled correctly."""
    curriculum_code = """constraints = [
    lambda x: x < 0,
    lambda x: x > -10
]"""
    answer_generation = _make_answer_generation("-5")

    result = reward_answer(
        extracted_curriculum=curriculum_code,
        answer_generation=answer_generation,
    )

    assert result.passed is True
    assert result.extracted_answer == "-5"


def test_reward_answer_float():
    """Test that float values are handled correctly."""
    curriculum_code = """constraints = [
    lambda x: x > 3,
    lambda x: x < 4
]"""
    answer_generation = _make_answer_generation("3.5")

    result = reward_answer(
        extracted_curriculum=curriculum_code,
        answer_generation=answer_generation,
    )

    assert result.passed is True
    assert result.extracted_answer == "3.5"
