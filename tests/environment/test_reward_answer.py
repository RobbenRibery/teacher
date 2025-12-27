import pathlib
import re
import sys
import pytest

project_root = pathlib.Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from environment.reward import reward_answer
from environment.format import BASE_PATTERN, CODE_BLOCK_REGEX


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


def _make_answer_generation(code: str) -> str:
    return f"<think>t</think><answer>\n```python\n{code}\n```</answer>"


def test_reward_answer_success():
    curriculum_code = _extract_curriculum_from_valid()
    answer_generation = _make_answer_generation("x = 26")  # satisfies valid constraints

    result = reward_answer(
        extracted_curriculum=curriculum_code,
        answer_generation=answer_generation,
    )

    assert result.passed is True
    assert result.error is None
    assert result.extracted_answer is not None


def test_reward_answer_format_failure():
    curriculum_code = _extract_curriculum_from_valid()
    answer_generation = _read_fixture(
        INVALID_PATH
    )  # missing proper answer block for our regex

    result = reward_answer(
        extracted_curriculum=curriculum_code,
        answer_generation=answer_generation,
    )

    assert result.passed is False
    assert result.reward == 0.0
    assert result.error is not None


def test_reward_answer_verification_failure():
    curriculum_code = _extract_curriculum_from_valid()
    # Constraints require x to satisfy curriculum; choose a value that fails
    answer_generation = _make_answer_generation("x = 4")

    result = reward_answer(
        extracted_curriculum=curriculum_code,
        answer_generation=answer_generation,
    )

    assert result.passed is False
    # verification failure returns error=None but reward=0.0
    assert result.reward == 0.0
    assert result.error is None
