import asyncio
import pathlib
import re
import sys
import importlib.util

import pytest

project_root = pathlib.Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Ensure environment package is loadable even in isolated test execution
env_init_spec = importlib.util.spec_from_file_location(
    "environment", project_root / "environment" / "__init__.py"
)
env_module = importlib.util.module_from_spec(env_init_spec)
sys.modules.setdefault("environment", env_module)
if env_init_spec.loader is not None:
    env_init_spec.loader.exec_module(env_module)

reward_spec = importlib.util.spec_from_file_location(
    "environment.reward", project_root / "environment" / "reward.py"
)
reward_module = importlib.util.module_from_spec(reward_spec)
sys.modules["environment.reward"] = reward_module

execution_spec = importlib.util.spec_from_file_location(
    "environment.execution", project_root / "environment" / "execution.py"
)
execution_module = importlib.util.module_from_spec(execution_spec)
sys.modules["environment.execution"] = execution_module
if execution_spec.loader is not None:
    execution_spec.loader.exec_module(execution_module)

if reward_spec.loader is not None:
    reward_spec.loader.exec_module(reward_module)

reward_answer_group = reward_module.reward_answer_group


FIXTURES_DIR = pathlib.Path(__file__).resolve().parent.parent
VALID_PATH = FIXTURES_DIR / "valid_curriculum_generation.txt"
INVALID_PATH = FIXTURES_DIR / "invalid_curriculum_generation.txt"
CODE_REGEX = re.compile(r"```python\n(.*?)\n```", re.DOTALL)


def _read_fixture(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_curriculum_from_valid() -> str:
    match = CODE_REGEX.search(_read_fixture(VALID_PATH))
    assert match, "Valid fixture must contain python code block"
    return match.group(1)


def _make_answer_generation(value: str) -> str:
    """Create a student answer in the \\boxed{} format."""
    return f"<think>Let me solve this...</think>\nThe answer is \\boxed{{{value}}}"


def test_reward_answer_group_mixed_results():
    curriculum_code = _extract_curriculum_from_valid()
    # x = 26 satisfies: even, (26+1)%3==0, (26-1)%5==0
    valid_answer = _make_answer_generation("26")
    # Invalid format - no \boxed{}
    invalid_format_answer = "<think>thinking</think>\nx = 26"
    # x = 4 does NOT satisfy the constraints
    verification_fail_answer = _make_answer_generation("4")

    answers = [valid_answer, invalid_format_answer, verification_fail_answer]

    results = asyncio.run(
        reward_answer_group(
            extracted_curriculum=curriculum_code,
            answer_generations=answers,
        )
    )

    assert len(results) == 3

    assert results[0].passed is True
    assert results[0].reward == 1.0
    assert results[0].error is None

    assert results[1].passed is False
    assert results[1].reward == 0.0
    assert results[1].error is not None

    assert results[2].passed is False
    assert results[2].reward == 0.0
    assert results[2].error is None  # No execution error, just wrong answer
