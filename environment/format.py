"""Unified regex patterns for extracting code from teacher and student generations.

This module provides centralized regex patterns and extraction utilities used across
the codebase for parsing model outputs that follow the 
<think>...<answer>```python...```</answer> format.
"""

from __future__ import annotations

import re
from re import Match
from functools import partial
from typing import Optional


# =============================================================================
# Core Regex Patterns
# =============================================================================

# Regex building blocks for pattern composition
_NO_ANSWER_TAGS = r"(?:(?!</?answer>).)*?"  # Match any content without answer tags
_NO_ANSWER_OR_CODE = r"(?:(?!</?answer>|```).)*?"  # Match any content without answer tags or code fences

# The unified format regex for both teacher and student outputs.
# Matches:
#   - Mandatory <think>...</think> block at the very start
#   - Optional whitespace after </think>
#   - <answer> tag
#   - Optional preamble (cannot contain <answer> or </answer>)
#   - ```python code fence
#   - Captured code content (group 1)
#   - Closing ``` fence
#   - Optional postamble (cannot contain <answer> or </answer>)
#   - </answer> tag at the very end
BASE_PATTERN = (
    r"^<think>.*?</think>\s*"      # Mandatory thinking block at start
    rf"<answer>{_NO_ANSWER_TAGS}"   # Opening tag + preamble (no nested answer tags)
    r"```python\n"                  # Python code fence
    rf"({_NO_ANSWER_OR_CODE})"      # Captured code content (no answer tags or code fences)
    r"\n```"                        # Closing fence
    r"</answer>$"                   # Closing tag at absolute end
)

# Simpler pattern that only matches the code block (no answer tags)
# Useful for testing or when answer tags are already stripped
CODE_BLOCK_REGEX = r"```python\n(.*?)\n```"


# =============================================================================
# Role-specific aliases (same pattern, different names for clarity)
# =============================================================================

# Teacher output format - expects curriculum definition with constraints
TEACHER_PATTERN = BASE_PATTERN
STUDENT_PATTERN = BASE_PATTERN

# Legacy alias for compatibility with existing code
ANSWER_FORMAT_REGEX = BASE_PATTERN

# =============================================================================
# Extraction Utilities
# =============================================================================

def extract_format(text: str, regex_format: str) -> Optional[str]:
    """
    Extract content from text using a regex pattern.

    Args:
        text: Text to extract from.
        regex_format: Regex pattern with at least one capture group for the content.
    Returns:
        Return the match inside the code block. 
    """
    matches = list[Match[str]](re.finditer(regex_format, text, re.DOTALL))
    if matches:
        match = matches[0]
        if match.groups():
            return match.group(1)
        return match.group(0)   
    return None


# Convenience functions using the unified regex
extract_teacher_code = partial(
    extract_format, regex_format=TEACHER_PATTERN
)
extract_student_code = partial(
    extract_format, regex_format=STUDENT_PATTERN
)
