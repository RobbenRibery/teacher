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

    def prompt(self, seed_task: str, num_instances: int, include_system: bool = False) -> list[dict[str, str]]:
        if include_system:
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
        else:
            return [
                {
                    "role": "user",
                    "content": self.system + \
                                "\n\n" + \
                                self.task.format(
                                    seed_task=seed_task, 
                                    num_instances=num_instances
                                )
                }
            ]


class TeacherPrompt(PromptTemplate):
    """
    A prompt template for a teacher.
    """
    system:str = Field(
        default=dedent(
        """
        You are a professor exceptionally skilled at crafting problem with high learning potential for students.
        """
        ).strip(),
        description="The system prompt for the teacher."
    )

    task: str = Field(
        default=dedent(
        """
        # Your Task
        Design exactly {num_instances} problem from the below curriculum.

        # Curriculum Description
        {seed_task}
        """
        ).strip(),
        description="The task prompt for the teacher."
    )