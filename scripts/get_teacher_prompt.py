#!/usr/bin/env python3
"""Script to extract and save the teacher's prompt to a markdown file."""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from curriculum.prompt_template import TeacherPrompt
from curriculum.tasks import BooleanConstraintsSatisfaction


def format_prompt_messages(messages: list[dict[str, str]]) -> str:
    """Format prompt messages as markdown without needing a tokenizer."""
    formatted = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        formatted.append(f"## {role.upper()}\n\n{content}\n")
    return "\n".join(formatted)


def main():
    """Load teacher prompt and save to markdown file without loading model."""
    # Create task and prompt template (no model needed)
    print("Creating task and prompt template...")
    task = BooleanConstraintsSatisfaction()
    teacher_prompt_template = TeacherPrompt()
    
    # Build the prompt messages (without chat template formatting)
    print("Building teacher prompt...")
    prompt_messages = teacher_prompt_template.prompt(
        seed_task=task.render(),
        num_instances=1,
        include_system=False,
    )
    

    # Save to markdown file
    output_file = "teacher_prompt.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(prompt_messages[0]["content"])
    
    print(f"Teacher prompt saved to {output_file}")


if __name__ == "__main__":
    main()

