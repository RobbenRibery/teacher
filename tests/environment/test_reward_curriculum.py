import pathlib
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

reward_curriculum = reward_module.reward_curriculum


FIXTURES_DIR = pathlib.Path(__file__).resolve().parent.parent
VALID_PATH = FIXTURES_DIR / "valid_curriculum_generation.txt"
INVALID_PATH = FIXTURES_DIR / "invalid_curriculum_generation.txt"


def _read_fixture(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def test_reward_curriculum_valid_format_only():
    curriculum_generation = _read_fixture(VALID_PATH)

    result = reward_curriculum(
        curriculum_generation=curriculum_generation,
        check_format_only=True,
    )

    assert result.passed is True
    assert result.extracted_curriculum is not None
    expected_curriculum = """constraints = [
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
        check_format_only=True,
    )

    assert result.passed is False
    assert result.reward == 0.0
    assert result.extracted_curriculum is None or result.extracted_curriculum == ""
    assert result.error is not None
    assert result.reward == 0.0

