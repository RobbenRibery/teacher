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
    def render(self, seed_task: str, num_instances: int) -> str:
        return self.system + "\n\n" + self.task.format(seed_task=seed_task, num_instances=num_instances)

    def prompt(self, seed_task: str, num_instances: int) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": self.system
            },
            {
                "role": "user",
                "content": self.task.format(seed_task=seed_task, num_instances=num_instances)
            }
        ]


class TeacherPrompt(PromptTemplate):
    """
    A prompt template for a teacher.
    """
    system:str = Field(
        default=dedent(
        """
        # Role
        You are a professor who's exceptionally skilled at designing diverse problems with high learning potential for students, given the following task.
        """
        ).strip(),
        description="The system prompt for the teacher."
    )

    task: str = Field(
        default=dedent(
        """
        # Task
        {seed_task}

        Now design {num_instances} problems originated from the above task.
        """
        ).strip(),
        description="The task prompt for the teacher."
    )