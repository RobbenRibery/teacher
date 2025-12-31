"""Regex patterns for extracting content from teacher and student generations.

This module provides centralized regex patterns and extraction utilities used across
the codebase for parsing model outputs.

Teacher format: <think>...</think>```python...```
Student format: <think>...</think>...\boxed{...}
"""

from __future__ import annotations

import re
from re import Match
from functools import partial
from typing import Optional


# =============================================================================
# Role-specific patterns
# =============================================================================

# Teacher output format - thinking followed by python code block
# MUST end with \n``` (newline + closing fence)
# Matches:
#   - Mandatory <think>...</think> block at the very start
#   - Optional whitespace after </think>
#   - ```python\n code fence opening
#   - Captured code content (group 1)
#   - \n``` at the END of completion (required)
TEACHER_PATTERN = (
    r"^<think>.*?</think>\s*"      # Mandatory thinking block at start
    r"```python\n"                  # Python code fence opening
    r"(.*?)"                        # Captured code content (non-greedy)
    r"\n```$"                       # \n``` MUST be at END of completion (no trailing whitespace)
)

# Student output format - thinking followed by answer in \boxed{}
# MUST have exactly one \boxed{...} after the <think> block.
# Matches:
#   - Mandatory <think>...</think> block at the very start
#   - Text that does NOT contain any other \boxed{
#   - The unique \boxed{...} (group 1)
#   - Text that does NOT contain any other \boxed{ until the end
STUDENT_PATTERN = (
    r"^<think>.*?</think>"          # Mandatory thinking block at start
    r"(?:(?!\\boxed\{).)*"          # No other boxes allowed before the answer
    r"\\boxed\{([^}]*)\}"           # The answer box
    r"\s*$"                         # MUST be the end of the string (only whitespace allowed)
)

# Simpler pattern that only matches the code block (no thinking required)
# Useful for testing or when thinking is already stripped
CODE_BLOCK_REGEX = r"```python\n(.*?)\n```"

# Legacy alias for compatibility with existing code
ANSWER_FORMAT_REGEX = TEACHER_PATTERN


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


# Convenience functions using role-specific patterns
extract_teacher_code = partial(
    extract_format, regex_format=TEACHER_PATTERN
)
extract_student_answer = partial(
    extract_format, regex_format=STUDENT_PATTERN
)

# Legacy alias
extract_student_code = extract_student_answer
