import pathlib
import sys
import pytest

project_root = pathlib.Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from environment.reward import reward_curriculum
from environment.format import ANSWER_FORMAT_REGEX


FIXTURES_DIR = pathlib.Path(__file__).resolve().parent.parent
VALID_PATH = FIXTURES_DIR / "valid_curriculum_generation.txt"
INVALID_PATH = FIXTURES_DIR / "invalid_curriculum_generation.txt"


def _read_fixture(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def test_reward_curriculum_valid_format_only():
    curriculum_generation = _read_fixture(VALID_PATH)

    result = reward_curriculum(
        curriculum_generation=curriculum_generation,
        format_regex=ANSWER_FORMAT_REGEX,
        check_format_only=True,
    )

    assert result.passed is True
    assert result.extracted_curriculum is not None
    expected_curriculum = """# Write Python code to derive a particular value of "x" which meets all constraints imposed:
constraints = [
    lambda x: x % 2 == 0,
    lambda x: (x + 1) % 3 == 0,
    lambda x: (x - 1) % 5 == 0
]"""
    assert result.extracted_curriculum == expected_curriculum
    assert result.error is None
    assert result.format_only is True
    assert result.reward == 1.0


def test_reward_curriculum_invalid_format():
    curriculum_generation = _read_fixture(INVALID_PATH)

    result = reward_curriculum(
        curriculum_generation=curriculum_generation,
        format_regex=ANSWER_FORMAT_REGEX,
        check_format_only=True,
    )

    assert result.passed is False
    assert result.reward == 0.0
    assert result.extracted_curriculum is None or result.extracted_curriculum == ""
    assert result.error is not None
    assert result.reward == 0.0
