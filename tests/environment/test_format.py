import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from environment.format import TEACHER_PATTERN, STUDENT_PATTERN, extract_format


# =============================================================================
# Teacher Pattern Tests (```python block after thinking, must end with \n```)
# =============================================================================


@pytest.mark.parametrize(
    "text, expected_code",
    [
        # Positive case: standard teacher format
        (
            """<think>
some thoughts
</think>
```python
x = 1
```""",
            """x = 1""",
        ),
        # Positive case: minimal whitespace
        (
            """<think></think>
```python
x=1
```""",
            """x=1""",
        ),
        # Positive case: multiline code
        (
            """<think>thinking here</think>
```python
x = 1
y = 2
z = x + y
```""",
            """x = 1
y = 2
z = x + y""",
        ),
        # Positive case: with comment inside python block
        (
            """<think>t</think>
```python
# This is a comment
x = 1
```""",
            """# This is a comment
x = 1""",
        ),
    ],
)
def test_teacher_pattern_positive(text, expected_code):
    result = extract_format(text, TEACHER_PATTERN)
    assert result == expected_code


@pytest.mark.parametrize(
    "text",
    [
        # Negative case: missing <think>
        """```python
x = 1
```""",
        # Negative case: <think> not at start
        """some text
<think>t</think>
```python
x = 1
```""",
        # Negative case: text after closing fence
        """<think>t</think>
```python
x = 1
```
some text after""",
        # Negative case: trailing whitespace after closing fence (not allowed)
        """<think>t</think>
```python
x = 1
```   """,
        # Negative case: missing python code block
        """<think>t</think>
x = 1""",
        # Negative case: uses <answer> tags (old format)
        """<think>t</think><answer>```python
x = 1
```</answer>""",
        # Negative case: code block not closed
        """<think>t</think>
```python
x = 1""",
    ],
)
def test_teacher_pattern_negative(text):
    result = extract_format(text, TEACHER_PATTERN)
    assert result is None


# =============================================================================
# Student Pattern Tests (\boxed{} at end after thinking)
# =============================================================================


@pytest.mark.parametrize(
    "text, expected_answer",
    [
        # Positive case: standard student format
        (
            r"""<think>
Let me solve this step by step.
x must be even and greater than 5.
So x = 8.
</think>
The answer is \boxed{8}""",
            "8",
        ),
        # Positive case: minimal
        (r"<think>t</think>\boxed{42}", "42"),
        # Positive case: with decimal
        (r"<think>calculating...</think>The value is \boxed{3.14}", "3.14"),
        # Positive case: negative number
        (r"<think>t</think>\boxed{-5}", "-5"),
        # Positive case: expression result
        (r"<think>t</think>\boxed{26}", "26"),
        # Positive case: text after boxed (newly allowed)
        (r"<think>t</think>\boxed{8} more text", "8"),
        # Positive case: trailing whitespace after boxed (newly allowed)
        (r"<think>t</think>\boxed{8}   ", "8"),
        # Positive case: multiple boxes, last one captured
        (r"<think>t</think>\boxed{1} and \boxed{2}", "2"),
        # Positive case: text before the first box
        (r"<think>t</think>Initial guess \boxed{1}, final answer \boxed{2}", "2"),
        # Positive case: nested braces (LaTeX)
        (r"<think>t</think>\boxed{\sqrt{5}}", r"\sqrt{5}"),
        (r"<think>t</think>\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
    ],
)
def test_student_pattern_positive(text, expected_answer):
    result = extract_format(text, STUDENT_PATTERN)
    assert result == expected_answer


@pytest.mark.parametrize(
    "text",
    [
        # Negative case: missing <think>
        r"\boxed{8}",
        # Negative case: <think> not at start
        r"text before<think>t</think>\boxed{8}",
        # Negative case: missing boxed
        "<think>t</think>x = 8",
    ],
)
def test_student_pattern_negative(text):
    result = extract_format(text, STUDENT_PATTERN)
    assert result is None


def test_student_pattern_multiple_boxed():
    """When multiple \\boxed{} exist, the last one is matched, even with text after."""
    text = r"<think>t</think>\boxed{8} and also \boxed{9} plus text"
    result = extract_format(text, STUDENT_PATTERN)
    # The pattern matches the last \boxed{}
    assert result == "9"
