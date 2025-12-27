import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from environment.format import BASE_PATTERN, extract_format

@pytest.mark.parametrize("text, expected_code", [
    # Positive case: standard format (no trailing newline after closing fence)
    ("""<think>
some thoughts
</think>
<answer>
```python
x = 1
```</answer>""", """x = 1"""),
    
    # Positive case: with preamble only (no postamble allowed after closing fence)
    ("""<think>t</think><answer>preamble
```python
print('hi')
```</answer>""", """print('hi')"""),
    
    # Positive case: with comment inside python block
    ("""<think>t</think><answer>```python
# This is a comment
x = 1
```</answer>""", """# This is a comment
x = 1"""),
    
    # Positive case: minimal whitespace
    ("""<think></think><answer>```python
x=1
```</answer>""", """x=1"""),
    
    # Positive case: multiline code with no trailing content
    ("""<think>t</think><answer>
# this a python code block
```python
x = 1
y = 2
z = x + y
```</answer>""", """x = 1
y = 2
z = x + y"""),
])
def test_base_pattern_positive(text, expected_code):
    result = extract_format(text, BASE_PATTERN)
    assert result == expected_code

@pytest.mark.parametrize("text", [
    # Negative case: missing <think>
    """<answer>```python
x = 1
```</answer>""",
    
    # Negative case: <think> not at start
    """some text
<think>t</think><answer>```python
x = 1
```</answer>""",
    
    # Negative case: </answer> not at end
    """<think>t</think><answer>```python
x = 1
```</answer>
some text""",
    
    # Negative case: nested <answer>
    """<think>t</think><answer><answer>```python
x = 1
```</answer></answer>""",
    
    # Negative case: multiple answer blocks
    """<think>t</think><answer>```python
x = 1
```</answer><answer>```python
x = 2
```</answer>""",
    
    # Negative case: missing python code block
    """<think>t</think><answer>x = 1</answer>""",
    
    # Negative case: postamble after closing fence (new pattern disallows this)
    """<think>t</think><answer>```python
x = 1
```
postamble</answer>""",
    
    # Negative case: newline after closing fence
    """<think>t</think><answer>```python
x = 1
```
</answer>""",
])
def test_base_pattern_negative(text):
    result = extract_format(text, BASE_PATTERN)
    assert result is None

