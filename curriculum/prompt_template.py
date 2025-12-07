from __future__ import annotations

from textwrap import dedent
from pydantic import BaseModel, Field


class PromptTemplate(BaseModel):
    """
    A prompt template
    """
    system:str = Field(
        ...,
        description="The system prompt for the teacher."
    )
    task:str = Field(
        ...,
        description="The seed for the task, must contain a 'seed_task' to be filled."
    )
    def render(self, seed_task: str) -> str:
        return self.system + "\n\n" + self.task.format(seed_task=seed_task)


class TeacherPrompt(PromptTemplate):
    """
    A prompt template for a teacher.
    """
    system:str = Field(
        default=dedent(
        """
        # Role
        You are a teacher who is motivated to design diverse and high learning potential curriculums for students, given a particular task. 

        # Thinking Process
        You think step by step to design the curriculum.

        # Output Format
        You must put your thinking process inside the <think>\n...\n</think> tag, and the final curriculum should be inside the <answer>\n...\n</answer> tag.
        """
        ).strip(),
        description="The system prompt for the teacher."
    )

    task: str = Field(
        default=dedent(
        """
        {seed_task}

        Now start designing the curriculum which are instances of the task!
        DO NOT solve the task or provide any hint about the solution.
        """
        ).strip(),
        description="The task prompt for the teacher."
    )